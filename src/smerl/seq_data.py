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
    switch_weight: float = 1.0     # layer-1 weight on skill SWITCH-decision steps
    decouple: bool = True          # skill TARGET = decoupled skill_target (True) or
                                   # the executed skill / coupled (False) — for ablation

    def tokens(self, ep: dict):
        """List of (in_name, in_vec, tgt_name|None, tgt_vec|None, weight, switch,
        stay, new).

        A SWITCH step is a skill-target (state) token whose executed skill changes
        at the NEXT step (``skills[t+1] != skills[t]``) — the pre-switch DECISION
        state (usable at test: fire here, then switch). ``stay`` = current skill,
        ``new`` = the skill switched to. ``switch`` is the binary logic-gate target
        and (when switch_weight!=1) the layer-1 loss-weight flag."""
        if self.token_fn is not None:
            return [tuple(x) for x in self.token_fn(ep)]
        T = int(len(ep["actions"]))
        out = []
        for t in range(T):
            for slot in self.pattern:
                in_name, tgt_name = _split_slot(slot)
                in_vec = _slot_vec(ep, in_name, t, is_target=False)
                if tgt_name is None:
                    out.append((in_name, in_vec, None, None, 1.0, 0, 0, 0))
                    continue
                # skill TARGET: decoupled skill_target only if self.decouple, else
                # the executed skill (coupled)
                tgt_is_target = (tgt_name != "skill") or self.decouple
                tgt_vec = _slot_vec(ep, tgt_name, t, is_target=tgt_is_target)
                w, switch, stay, new = 1.0, 0, 0, 0
                if tgt_name == "skill":
                    cur = int(_slot_vec(ep, "skill", t, is_target=False)[0])
                    nxt = (int(_slot_vec(ep, "skill", t + 1, is_target=False)[0])
                           if t + 1 < T else cur)
                    stay, new = cur, nxt
                    if nxt != cur:                   # temporal switch-decision step
                        switch = 1
                        w = self.switch_weight
                out.append((in_name, in_vec, tgt_name, tgt_vec, w, switch, stay, new))
        return out


class SeqDataset(Dataset):
    """Tokenized episodes from a trajectory store, per a SequenceSpec."""

    def __init__(self, store_dir, spec: SequenceSpec,
                 modalities: list[tuple[str, int]] | None = None):
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

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
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
        return {"token_ids": torch.from_numpy(ids),
                "values": torch.from_numpy(vals),
                "target_ids": torch.from_numpy(tgt_ids),
                "target_values": torch.from_numpy(tgt_vals),
                "loss_mask": torch.from_numpy(loss),
                "loss_weight": torch.from_numpy(weight),
                "switch_mask": torch.from_numpy(switch),
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
        stay_id[i, :n] = b["stay_id"]
        new_id[i, :n] = b["new_id"]
        attn[i, :n] = 1.0
    return {"token_ids": token_ids, "values": values, "target_ids": target_ids,
            "target_values": target_values, "loss_mask": loss_mask,
            "loss_weight": loss_weight, "switch_mask": switch_mask,
            "stay_id": stay_id, "new_id": new_id, "attention_mask": attn,
            "z": torch.tensor([b["z"] for b in batch]),
            "success": torch.tensor([b["success"] for b in batch])}
