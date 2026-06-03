"""On-disk store for collected (state, action) trajectories.

A *store* is a directory holding one compressed ``.npz`` per episode plus a
``manifest.json`` describing the dataset. Each episode file holds the aligned
arrays of a single rollout; the manifest carries dataset-level metadata and the
per-episode scalar labels (skill, latent theta, outcome, failure step/mode).

Episode arrays (action[t] is taken at state[t] and leads to state[t+1]):

    states  : float32 [T+1, obs_dim]   base env observations
    actions : float32 [T,   act_dim]
    rewards : float32 [T]
    values  : float32 [T+1]            exposed task value (corrupted post-failure)
    failing : bool    [T+1]            absorbing-failure flag

For transformer training, load with ``TrajectoryDataset``:

    ds = TrajectoryDataset("path/to/store", mode="sequence")   # per-episode
    ds = TrajectoryDataset("path/to/store", mode="pairs")      # flat (s, a) rows
"""

from __future__ import annotations

import json
import os

import numpy as np
import torch
from torch.utils.data import Dataset


# ----------------------------- writing -----------------------------------


class TrajectoryWriter:
    """Incrementally writes episodes to a store directory."""

    def __init__(self, out_dir: str, meta: dict):
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)
        self.meta = dict(meta)
        self.records: list[dict] = []

    def add(self, ep: dict) -> None:
        """Append one episode. ``ep`` must contain the arrays above plus the
        scalar labels z, theta, success, fail_step, fail_mode. Optionally a
        per-step ``skills`` array (the active skill at each step; constant for a
        single-skill demo, switching for an intervention/DAgger episode) and the
        intervention labels (intervened, intervene_step, z_good)."""
        idx = len(self.records)
        fname = f"ep_{idx:06d}.npz"
        arrs = dict(
            states=np.asarray(ep["states"], dtype=np.float32),
            actions=np.asarray(ep["actions"], dtype=np.float32),
            rewards=np.asarray(ep["rewards"], dtype=np.float32),
            values=np.asarray(ep["values"], dtype=np.float32),
            failing=np.asarray(ep["failing"], dtype=bool),
        )
        if ep.get("skills") is not None:
            arrs["skills"] = np.asarray(ep["skills"], dtype=np.int64)
        if ep.get("skill_target") is not None:
            arrs["skill_target"] = np.asarray(ep["skill_target"], dtype=np.int64)
        np.savez_compressed(os.path.join(self.out_dir, fname), **arrs)
        self.records.append({
            "file": fname,
            "length": int(len(ep["actions"])),
            "z": int(ep["z"]),
            "theta": np.asarray(ep["theta"], dtype=float).tolist(),
            "success": bool(ep["success"]),
            "fail_step": (None if ep["fail_step"] is None
                          else int(ep["fail_step"])),
            "fail_mode": ep["fail_mode"],
            "intervened": bool(ep.get("intervened", False)),
            "intervene_step": (None if ep.get("intervene_step") is None
                               else int(ep["intervene_step"])),
            "z_good": (None if ep.get("z_good") is None else int(ep["z_good"])),
        })

    def close(self) -> str:
        path = os.path.join(self.out_dir, "manifest.json")
        with open(path, "w") as f:
            json.dump({"meta": self.meta, "episodes": self.records}, f, indent=2)
        return path


# ----------------------------- reading -----------------------------------


def load_manifest(out_dir: str) -> dict:
    with open(os.path.join(out_dir, "manifest.json")) as f:
        return json.load(f)


def load_episode(out_dir: str, rec: dict) -> dict:
    blob = np.load(os.path.join(out_dir, rec["file"]))
    return {k: blob[k] for k in blob.files} | {
        "z": rec["z"], "theta": np.asarray(rec["theta"]),
        "success": rec["success"], "fail_step": rec["fail_step"],
        "fail_mode": rec["fail_mode"],
        "intervened": rec.get("intervened", False),
        "intervene_step": rec.get("intervene_step"),
        "z_good": rec.get("z_good")}


def load_trajectories(out_dir: str):
    """Eagerly load every episode. Returns (episodes, meta)."""
    man = load_manifest(out_dir)
    eps = [load_episode(out_dir, r) for r in man["episodes"]]
    return eps, man["meta"]


class TrajectoryDataset(Dataset):
    """Torch dataset over a trajectory store.

    mode="sequence": item i is a dict of per-episode tensors (ragged lengths;
        use ``pad_collate`` in a DataLoader).
    mode="pairs":    flattened (state, action) rows across all episodes; item i
        is (state[obs_dim], action[act_dim]) for behavior-cloning style training.
    """

    def __init__(self, out_dir: str, mode: str = "sequence"):
        self.out_dir = out_dir
        self.mode = mode
        self.man = load_manifest(out_dir)
        self.records = self.man["episodes"]
        if mode == "pairs":
            S, A = [], []
            for r in self.records:
                ep = load_episode(out_dir, r)
                S.append(ep["states"][:-1])     # align action[t] with state[t]
                A.append(ep["actions"])
            self._S = torch.as_tensor(np.concatenate(S), dtype=torch.float32)
            self._A = torch.as_tensor(np.concatenate(A), dtype=torch.float32)
        elif mode != "sequence":
            raise ValueError(f"unknown mode {mode!r}")

    def __len__(self):
        return self._S.shape[0] if self.mode == "pairs" else len(self.records)

    def __getitem__(self, i):
        if self.mode == "pairs":
            return self._S[i], self._A[i]
        ep = load_episode(self.out_dir, self.records[i])
        return {
            "states": torch.as_tensor(ep["states"], dtype=torch.float32),
            "actions": torch.as_tensor(ep["actions"], dtype=torch.float32),
            "rewards": torch.as_tensor(ep["rewards"], dtype=torch.float32),
            "values": torch.as_tensor(ep["values"], dtype=torch.float32),
            "failing": torch.as_tensor(ep["failing"], dtype=torch.bool),
            "z": int(ep["z"]),
            "success": bool(ep["success"]),
        }


def pad_collate(batch: list[dict]) -> dict:
    """Right-pad a batch of sequence items to the max length; adds a key_padding
    ``mask`` (True = padded) suitable for transformer attention masking."""
    T = max(b["actions"].shape[0] for b in batch)
    out = {"z": torch.tensor([b["z"] for b in batch]),
           "success": torch.tensor([b["success"] for b in batch]),
           "length": torch.tensor([b["actions"].shape[0] for b in batch])}
    mask = torch.ones(len(batch), T, dtype=torch.bool)
    for key in ("states", "actions", "rewards", "values", "failing"):
        ref = batch[0][key]
        shape = (len(batch), T + (1 if key in ("states", "values", "failing")
                                  else 0)) + ref.shape[1:]
        buf = torch.zeros(shape, dtype=ref.dtype)
        for i, b in enumerate(batch):
            v = b[key]
            buf[i, :v.shape[0]] = v
            if key == "actions":
                mask[i, :v.shape[0]] = False
        out[key] = buf
    out["mask"] = mask
    return out
