"""Stateful inference wrapper around the multimodal TrajectoryGPT.

Owns the running [state, skill, value] token sequence and exposes the skill and
action heads, so callers never hand-build token ids/values. Per timestep:

    at.update(state)        # append the state token
    z = at.sample_skill()   # read the skill head, append the skill token
    a = at.sample_action()  # read the action head (no token appended)
    at.push_value(v)        # append the value token V(s, z)

Asserts enforce that ordering, so the context each head reads is never malformed.
``revise_state`` overwrites the just-appended state (e.g. after a reset-on-switch
teleport, where the skill is decided on the old state but acted from the new one).

Inference is incrementally KV-cached: each head read encodes only the tokens
appended since the last read (the backbone keeps per-layer (k, v) for the prefix),
so a rollout costs O(T) full-width attention rows instead of O(T^2) re-encodes.
``revise_state`` rolls the cache back past the revised state token and re-encodes.
Set cache=False to force the legacy full-recompute path (equivalence testing).
"""

from __future__ import annotations

import numpy as np
import torch


class AdaptiveTransformer:
    def __init__(self, model, device, cache=True):
        self.model = model
        self.model.eval()
        self.device = device
        self.cache = bool(cache)
        self.sid = model.id_of["state"]
        self.zid = model.id_of["skill"]
        self.vid = model.id_of["value"]
        self.D = model.max_dim
        self._ids: list[int] = []
        self._vals: list[np.ndarray] = []
        self._stage = "state"          # next legal call: state -> skill -> action -> value
        self._past = None              # per-layer (k, v) covering _ids[:_cached_len]
        self._cached_len = 0

    def reset(self):
        self._ids.clear()
        self._vals.clear()
        self._stage = "state"
        self._past = None
        self._cached_len = 0

    def _vec(self, x):
        v = np.zeros(self.D, np.float32)
        x = np.asarray(x, np.float32).reshape(-1)
        v[: len(x)] = x
        return v

    @torch.no_grad()
    def _last_hidden(self):
        if not self.cache:                         # legacy full recompute
            tid = torch.as_tensor(self._ids, device=self.device)[None]
            tval = torch.as_tensor(np.stack(self._vals), device=self.device)[None]
            attn = torch.ones_like(tid, dtype=torch.float32)
            return self.model.backbone(self.model.embed(tid, tval), attn)[0, -1]
        new_ids = self._ids[self._cached_len:]
        assert new_ids, "no new tokens since the last head read"
        # embed the (1-3) new tokens directly — modalities are known python-side,
        # so skip model.embed's per-modality mask scatter (it costs a GPU sync per
        # modality via .any())
        m = self.model
        embs = []
        for j, mid in enumerate(new_ids):
            name = m.names[mid]
            val = self._vals[self._cached_len + j]
            if name in m.discrete:
                e = m.encoders[name](torch.tensor([int(val[0])], device=self.device))
            else:
                d = m.dims[name]
                e = m.encoders[name](
                    torch.as_tensor(val[None, :d], device=self.device))
            embs.append(e + m.type_emb.weight[mid])
        x = torch.cat(embs, 0)[None]                      # [1, L_new, E]
        h, self._past = m.backbone(x, None, past_kvs=self._past, use_cache=True)
        self._cached_len = len(self._ids)
        return h[0, -1]

    def update(self, state):
        assert self._stage == "state", f"update() out of order (stage={self._stage})"
        self._ids.append(self.sid)
        self._vals.append(self._vec(state))
        self._stage = "skill"

    def sample_skill(self, force=None, rng=None, temperature=1.0):
        """Decide the skill from the last state token and append its token.

        force: commit to this skill (skip the head). rng: sample from the
        softmax at ``temperature`` instead of taking the argmax."""
        assert self._stage == "skill", f"sample_skill() out of order (stage={self._stage})"
        if force is not None:
            z = int(force)
        else:
            logits = self.model.head_logits("skill", self._last_hidden())
            if rng is None:
                z = int(logits.argmax())
            else:
                p = logits.detach().cpu().numpy().astype(np.float64)
                p = np.exp((p - p.max()) / temperature)
                p /= p.sum()
                z = int(rng.choice(len(p), p=p))
        self._ids.append(self.zid)
        self._vals.append(self._vec(z))
        self._stage = "action"
        return z

    def revise_state(self, state):
        """Overwrite the current step's state token (skill already chosen on the old one)."""
        assert self._stage == "action" and self._ids[-1] == self.zid \
            and self._ids[-2] == self.sid, f"revise_state() out of order (stage={self._stage})"
        self._vals[-2] = self._vec(state)
        keep = len(self._ids) - 2              # tokens before the revised state
        if self._cached_len > keep:            # state token already in the cache:
            if keep == 0:                      # roll back so it re-encodes revised
                self._past = None
            else:
                self._past = [(k[:, :, :keep], v[:, :, :keep])
                              for k, v in self._past]
            self._cached_len = keep

    def sample_action(self):
        assert self._stage == "action", f"sample_action() out of order (stage={self._stage})"
        a = torch.clamp(self.model.head_mean("action", self._last_hidden()), -1, 1)
        self._stage = "value"
        return a.cpu().numpy()

    def skip_action(self):
        """Advance past the action slot when an external policy supplies the action."""
        assert self._stage == "action", f"skip_action() out of order (stage={self._stage})"
        self._stage = "value"

    def push_value(self, value):
        assert self._stage == "value", f"push_value() out of order (stage={self._stage})"
        self._ids.append(self.vid)
        self._vals.append(self._vec(value))
        self._stage = "state"
