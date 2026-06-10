"""Turn stored trajectories into mixed-modality token sequences for TrajectoryGPT.

A *modality registry* assigns each stream a stable id and dim, derived from the
trajectory store's manifest:

    state   : obs_dim      action : act_dim
    reward  : 1            value  : 1            failing : 1

A ``SequenceSpec`` is a list of per-timestep **slots**, tiled across the episode.
Each slot is either

    "state"               -> an input token, no supervised target
    ["value", "action"]   -> input token "value"; target "action" predicted from
                             THIS token's hidden state (same timestep)

So the layout [s_1, v_1, s_2, v_2, ...] with action regressed from each v_t is::

    SequenceSpec(pattern=["state", ["value", "action"]])

and behavior cloning from [s,a,s,a,...] (action from the state token) is::

    SequenceSpec(pattern=[["state", "action"], "action"])

For non-tiled layouts (e.g. [s,s,s,a]) pass a custom
``token_fn(episode) -> list of (in_name, in_vec, target_name|None, target_vec|None)``.

The dataset yields per-episode dicts; ``collate`` right-pads a batch and emits the
attention mask (1=real token) and the loss mask (1 where a target is supervised).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from torch.utils.data import Dataset

from src.smerl.trajectory_store import load_manifest, load_episode


# map modality name -> stored array key in the episode npz.
# "skill" is special: it has no per-step array; its value is the episode's scalar
# skill index z (constant over t), used both as an input embedding and a target.
_STREAM_KEY = {"state": "states", "action": "actions", "reward": "rewards",
               "value": "values", "failing": "failing"}


def make_modalities(meta: dict) -> list[tuple[str, int]]:
    """Ordered (name, dim) registry. Order fixes the modality ids. ``skill`` is a
    discrete index, so its dim is 1 (the index lives in slot 0)."""
    return [("state", int(meta["obs_dim"])), ("action", int(meta["act_dim"])),
            ("reward", 1), ("value", 1), ("failing", 1), ("skill", 1)]


def _as_vec(x) -> np.ndarray:
    a = np.asarray(x, dtype=np.float32)
    return a.reshape(-1) if a.ndim else a.reshape(1)


def _slot_vec(ep: dict, name: str, t: int, is_target: bool = False) -> np.ndarray:
    """Per-timestep raw vector for a modality.

    For ``skill`` the INPUT token is the per-step EXECUTED skill (``skills``, the
    skill whose action is in the sequence), while the TARGET is the per-step
    EXPERT label (``skill_target`` — decoupled: it flips to the good skill as soon
    as failure is observed, even while z0 is still executing). Both fall back to
    the constant episode label z for single-skill demos."""
    if name == "skill":
        if is_target:
            tgt = ep.get("skill_target")
            if tgt is not None and len(tgt):
                return np.asarray([tgt[min(t, len(tgt) - 1)]], dtype=np.float32)
        sk = ep.get("skills")
        if sk is not None and len(sk):
            return np.asarray([sk[min(t, len(sk) - 1)]], dtype=np.float32)
        return np.asarray([ep["z"]], dtype=np.float32)
    stream = ep[_STREAM_KEY[name]]
    return _as_vec(stream[min(t, len(stream) - 1)])


def _split_slot(slot):
    """Return (input_name, target_name|None) for a pattern slot."""
    if isinstance(slot, (list, tuple)):
        return slot[0], slot[1]
    return slot, None


@dataclass
class SequenceSpec:
    pattern: list = field(default_factory=lambda: ["state", ["value", "action"]])
    token_fn: object = None        # optional: episode -> token/target tuples
    balance_switch: bool = True    # data-driven switch reweighting: per episode, scale
                                   # the decision tokens so they contribute switch_share
                                   # of the skill gradient (0.5 = equal switch/non-switch)
    switch_share: float = 0.5      # target gradient fraction for the decision group
    switch_weight: float = 1.0     # FIXED fallback weight on switch tokens, used only
                                   # when balance_switch is False (legacy ablation)
    decouple: bool = True          # skill TARGET = decoupled skill_target (True) or
                                   # the executed skill / coupled (False) — for ablation
    expert_gated: bool = False     # HG-DAgger: supervise a step ONLY where the expert
                                   # acted (ep["expert_mask"]); learner steps are context

    def tokens(self, ep: dict):
        """List of (in_name, in_vec, tgt_name|None, tgt_vec|None, weight, switch,
        stay, new).

        A SWITCH (DECISION) token is a skill-target token whose supervised target
        disagrees with the skill currently executing (``skill_target[t] != skills[t]``):
        the target says switch while the old skill is still running. Under decoupling
        the target leads the executed switch, so this is the whole WINDOW of tokens, not
        just the one executed-transition step (coupled mode falls back to that step).
        The ``switch`` flag marks EVERY such token, so the logic-gate BCE is supervised
        on the entire window; ``stay`` = executing skill, ``new`` = the target skill.

        ``balance_switch`` (default) scales the decision tokens so they contribute
        ``switch_share`` of the skill gradient: ``w = (s/(1-s)) * n_nondecision /
        n_decision`` (s=0.5 -> equal switch/non-switch summed gradient)."""
        if self.token_fn is not None:
            return [tuple(x) for x in self.token_fn(ep)]
        T = int(len(ep["actions"]))
        em = ep.get("expert_mask") if self.expert_gated else None
        out = []
        skill_pos = []                  # (index in out, is_decision) per skill token
        n_dec = 0
        for t in range(T):
            sup = em is None or bool(em[min(t, len(em) - 1)])   # HG-DAgger loss mask
            for slot in self.pattern:
                in_name, tgt_name = _split_slot(slot)
                in_vec = _slot_vec(ep, in_name, t, is_target=False)
                if tgt_name is None or not sup:                 # learner step -> context only
                    out.append((in_name, in_vec, None, None, 1.0, 0, 0, 0))
                    continue
                # skill TARGET: decoupled skill_target only if self.decouple, else
                # the executed skill (coupled)
                tgt_is_target = (tgt_name != "skill") or self.decouple
                tgt_vec = _slot_vec(ep, tgt_name, t, is_target=tgt_is_target)
                w, switch, stay, new = 1.0, 0, 0, 0
                if tgt_name == "skill":
                    cur = int(_slot_vec(ep, "skill", t, is_target=False)[0])  # executing
                    nxt = (int(_slot_vec(ep, "skill", t + 1, is_target=False)[0])
                           if t + 1 < T else cur)
                    tgt_cur = int(tgt_vec[0])                 # supervised target skill
                    # decision token: target disagrees with the executing skill. Under
                    # decoupling the target leads the executed switch, so this is the
                    # whole window (multiple tokens); coupled -> the transition step.
                    is_dec = (tgt_cur != cur) if self.decouple else (nxt != cur)
                    if is_dec:
                        switch = 1                           # logic-gate target on EVERY
                        stay = cur                           # decision-window token
                        new = tgt_cur if self.decouple else nxt
                    skill_pos.append((len(out), is_dec))
                    n_dec += int(is_dec)
                out.append((in_name, in_vec, tgt_name, tgt_vec, w, switch, stay, new))
        # per-episode reweighting: scale decision tokens so their summed loss weight
        # equals that of the non-decision skill tokens (equal gradient contribution).
        if n_dec > 0:
            n_non = len(skill_pos) - n_dec
            if self.balance_switch:
                s = self.switch_share
                w_dec = ((s / (1.0 - s)) * (n_non / n_dec)) if n_non > 0 else 1.0
            else:
                w_dec = self.switch_weight
            for j, is_dec in skill_pos:
                if is_dec:
                    tok = out[j]
                    out[j] = tok[:4] + (w_dec,) + tok[5:]
        return out


class SeqDataset(Dataset):
    """Tokenized episodes from a trajectory store, per a SequenceSpec."""

    def __init__(self, store_dir, spec: SequenceSpec,
                 modalities: list[tuple[str, int]] | None = None,
                 cache: bool = True):
        # store_dir may be a single path or a list of paths (DAgger merge): the
        # datasets are concatenated, modalities taken from the first store.
        dirs = [store_dir] if isinstance(store_dir, str) else list(store_dir)
        self.spec = spec
        self.man = load_manifest(dirs[0])
        # flat list of (store_dir, record) so episodes from all stores interleave
        self.items = [(d, rec) for d in dirs for rec in load_manifest(d)["episodes"]]
        self.modalities = modalities or make_modalities(self.man["meta"])
        self.id_of = {n: i for i, (n, _) in enumerate(self.modalities)}
        self.max_dim = max(d for _, d in self.modalities)
        # in-RAM memoization of fully tokenized items: npz decompression +
        # tokenization run ONCE per episode (first epoch), ~0.3 MB/episode.
        self._cache: list[dict | None] | None = [None] * len(self.items) if cache else None

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        if self._cache is not None:
            it = self._cache[i]
            if it is None:
                it = self._build(i)
                self._cache[i] = it
            return it
        return self._build(i)

    def _build(self, i):
        store_dir, rec = self.items[i]
        ep = load_episode(store_dir, rec)
        toks = self.spec.tokens(ep)
        L = len(toks)
        D = self.max_dim
        ids = np.zeros(L, dtype=np.int64)
        vals = np.zeros((L, D), dtype=np.float32)
        tgt_ids = np.zeros(L, dtype=np.int64)
        tgt_vals = np.zeros((L, D), dtype=np.float32)
        loss = np.zeros(L, dtype=np.float32)
        weight = np.ones(L, dtype=np.float32)
        switch = np.zeros(L, dtype=np.float32)     # 1 at skill switch-decision steps
        stay = np.zeros(L, dtype=np.int64)         # current skill at those
        new = np.zeros(L, dtype=np.int64)          # the skill switched to
        for j, tok in enumerate(toks):
            in_name, in_vec, tgt_name, tgt_vec = tok[:4]
            w = tok[4] if len(tok) > 4 else 1.0
            ids[j] = self.id_of[in_name]
            vals[j, :len(in_vec)] = in_vec
            if tgt_name is not None:
                tgt_ids[j] = self.id_of[tgt_name]
                tgt_vals[j, :len(tgt_vec)] = tgt_vec
                loss[j] = 1.0
                weight[j] = w
                if len(tok) > 7:
                    switch[j] = tok[5]
                    stay[j] = tok[6]
                    new[j] = tok[7]
        # within-window position (0-based) of each switch token among skill tokens,
        # resetting at each contiguous run — lets us log P(switch) vs how many failure
        # feedback tokens have accrued in context.
        switch_pos = np.full(L, -1, dtype=np.int64)
        sk_id = self.id_of.get("skill")
        if sk_id is not None:
            k, run = 0, False
            for j in range(L):
                if loss[j] > 0 and tgt_ids[j] == sk_id:
                    if switch[j] > 0:
                        k = 0 if not run else k
                        switch_pos[j] = k
                        k += 1
                        run = True
                    else:
                        run = False
        return {"token_ids": torch.from_numpy(ids),
                "values": torch.from_numpy(vals),
                "target_ids": torch.from_numpy(tgt_ids),
                "target_values": torch.from_numpy(tgt_vals),
                "loss_mask": torch.from_numpy(loss),
                "loss_weight": torch.from_numpy(weight),
                "switch_mask": torch.from_numpy(switch),
                "switch_pos": torch.from_numpy(switch_pos),
                "stay_id": torch.from_numpy(stay),
                "new_id": torch.from_numpy(new),
                "length": L, "z": int(ep["z"]), "success": bool(ep["success"])}


def collate(batch: list[dict]) -> dict:
    L = max(b["length"] for b in batch)
    B = len(batch)
    D = batch[0]["values"].shape[1]
    token_ids = torch.zeros(B, L, dtype=torch.long)
    values = torch.zeros(B, L, D, dtype=torch.float32)
    target_ids = torch.zeros(B, L, dtype=torch.long)
    target_values = torch.zeros(B, L, D, dtype=torch.float32)
    loss_mask = torch.zeros(B, L, dtype=torch.float32)
    loss_weight = torch.ones(B, L, dtype=torch.float32)
    switch_mask = torch.zeros(B, L, dtype=torch.float32)
    switch_pos = torch.full((B, L), -1, dtype=torch.long)
    stay_id = torch.zeros(B, L, dtype=torch.long)
    new_id = torch.zeros(B, L, dtype=torch.long)
    attn = torch.zeros(B, L, dtype=torch.float32)
    for i, b in enumerate(batch):
        n = b["length"]
        token_ids[i, :n] = b["token_ids"]
        values[i, :n] = b["values"]
        target_ids[i, :n] = b["target_ids"]
        target_values[i, :n] = b["target_values"]
        loss_mask[i, :n] = b["loss_mask"]
        loss_weight[i, :n] = b["loss_weight"]
        switch_mask[i, :n] = b["switch_mask"]
        switch_pos[i, :n] = b["switch_pos"]
        stay_id[i, :n] = b["stay_id"]
        new_id[i, :n] = b["new_id"]
        attn[i, :n] = 1.0
    return {"token_ids": token_ids, "values": values, "target_ids": target_ids,
            "target_values": target_values, "loss_mask": loss_mask,
            "loss_weight": loss_weight, "switch_mask": switch_mask,
            "switch_pos": switch_pos,
            "stay_id": stay_id, "new_id": new_id, "attention_mask": attn,
            "z": torch.tensor([b["z"] for b in batch]),
            "success": torch.tensor([b["success"] for b in batch])}
