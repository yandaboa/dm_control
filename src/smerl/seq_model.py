"""GPT2 backbone + trajectory transformer for mixed-modality token sequences.

Self-contained GPT2 (minGPT-style: pre-LN blocks, GELU MLP, learned positional
embeddings, causal multi-head attention) operating on **continuous** input
embeddings, so we can feed heterogeneous trajectory tokens (states, actions,
rewards, values, ...) rather than discrete vocab ids.

``TrajectoryGPT`` is modality-generic:

  * one linear **encoder** per modality maps its raw vector -> n_embd
  * a learned **type embedding** per modality is added so the model knows the
    token kind regardless of where it sits in the sequence
  * one linear **head** per modality decodes a hidden state back to that modality

It is trained as a causal next-token predictor: the hidden state at position i
predicts the token at i+1, decoded by *that token's* modality head. A per-token
**loss mask** selects which next-token predictions contribute (e.g. only actions
for behavior cloning, or every token for full sequence modeling). This makes the
same model train on [s,a,s,a], [s,s,s,a], [s,r,s,r,a], ... unchanged — only the
token layout (see seq_data.SequenceSpec) and the predicted-modality set differ.

Padding is handled with an attention mask; padded positions never attend and
never contribute to the loss.
"""

from __future__ import annotations

from dataclasses import dataclass

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPT2Config:
    n_embd: int = 128
    n_layer: int = 4
    n_head: int = 4
    n_positions: int = 1024
    dropout: float = 0.1


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: GPT2Config):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd)
        self.attn_drop = nn.Dropout(cfg.dropout)
        self.resid_drop = nn.Dropout(cfg.dropout)

    def forward(self, x, key_padding_mask=None):
        B, L, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        h = self.n_head
        q = q.view(B, L, h, C // h).transpose(1, 2)   # [B,h,L,d]
        k = k.view(B, L, h, C // h).transpose(1, 2)
        v = v.view(B, L, h, C // h).transpose(1, 2)
        # additive mask: causal + padding, broadcast to [B,h,L,L]
        mask = torch.zeros(B, 1, L, L, device=x.device, dtype=x.dtype)
        causal = torch.triu(torch.ones(L, L, device=x.device, dtype=torch.bool), 1)
        mask = mask.masked_fill(causal, float("-inf"))
        if key_padding_mask is not None:   # [B,L] True=valid
            pad = (~key_padding_mask.bool())[:, None, None, :]
            mask = mask.masked_fill(pad, float("-inf"))
        att = (q @ k.transpose(-2, -1)) / math.sqrt(C // h) + mask
        att = self.attn_drop(F.softmax(att, dim=-1))
        y = (att @ v).transpose(1, 2).contiguous().view(B, L, C)
        return self.resid_drop(self.proj(y))


class Block(nn.Module):
    def __init__(self, cfg: GPT2Config):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.n_embd, 4 * cfg.n_embd), nn.GELU(),
            nn.Linear(4 * cfg.n_embd, cfg.n_embd), nn.Dropout(cfg.dropout))

    def forward(self, x, key_padding_mask=None):
        x = x + self.attn(self.ln1(x), key_padding_mask)
        x = x + self.mlp(self.ln2(x))
        return x


class GPT2Backbone(nn.Module):
    """Consumes continuous inputs_embeds [B,L,n_embd] + attention_mask [B,L]."""

    def __init__(self, cfg: GPT2Config):
        super().__init__()
        self.cfg = cfg
        self.wpe = nn.Embedding(cfg.n_positions, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd)

    def forward(self, inputs_embeds, attention_mask=None):
        B, L, _ = inputs_embeds.shape
        pos = torch.arange(L, device=inputs_embeds.device)
        x = self.drop(inputs_embeds + self.wpe(pos)[None, :, :])
        for blk in self.blocks:
            x = blk(x, attention_mask)
        return self.ln_f(x)


class TrajectoryGPT(nn.Module):
    """Mixed-modality causal trajectory transformer.

    modalities: ordered dict-like list of (name, dim). The id of a modality is
    its index in this list (must match seq_data's registry)."""

    def __init__(self, modalities: list[tuple[str, int]], cfg: GPT2Config,
                 head: str = "mse", discrete: dict | None = None,
                 head_types: dict | None = None, loss_weights: dict | None = None,
                 n_bins: int = 21, logic: bool = False):
        """modalities: ordered (name, dim) registry.

        discrete:   {name: n_classes} for discrete modalities (e.g. a skill index)
                    -> embedding-lookup encoder + categorical (logits) head.
        head_types: {name: "mse"|"gaussian"|"categorical"|"disc"} per-target head.
                    "disc": per-dim categorical over `n_bins` bins on [-1,1] (CE),
                    so a continuous action becomes K independent classifications.
        loss_weights: {name: lambda} per-modality loss multiplier (layer-2 class
                    balancing — e.g. lift skill so the action NLL doesn't dominate
                    the shared backbone). Default 1.0 each.
        n_bins:     bins per dim for "disc" heads.
        """
        super().__init__()
        self.cfg = cfg
        self.default_head = head         # fallback for continuous modalities
        self.log_std_min, self.log_std_max = -5.0, 2.0
        self.n_bins = int(n_bins)
        self.names = [n for n, _ in modalities]
        self.dims = {n: d for n, d in modalities}
        self.id_of = {n: i for i, n in enumerate(self.names)}
        self.max_dim = max(self.dims.values())
        self.discrete = dict(discrete or {})    # name -> n_classes
        self.loss_weights = dict(loss_weights or {})   # name -> lambda
        # resolve a head type per modality
        self.head_types = {}
        for n in self.names:
            if head_types and n in head_types:
                self.head_types[n] = head_types[n]
            elif n in self.discrete:
                self.head_types[n] = "categorical"
            else:
                self.head_types[n] = head
        # encoders: embedding-lookup for discrete, linear for continuous modalities
        self.encoders = nn.ModuleDict({
            n: (nn.Embedding(self.discrete[n], cfg.n_embd) if n in self.discrete
                else nn.Linear(d, cfg.n_embd))
            for n, d in modalities})
        # heads sized by their head type
        heads = {}
        for n, d in modalities:
            ht = self.head_types[n]
            if ht == "categorical":
                heads[n] = nn.Linear(cfg.n_embd, self.discrete[n])
            elif ht == "disc":
                heads[n] = nn.Linear(cfg.n_embd, d * self.n_bins)
            else:
                heads[n] = nn.Linear(cfg.n_embd, (2 if ht == "gaussian" else 1) * d)
        self.heads = nn.ModuleDict(heads)
        # logic gate: a binary "does a switch happen here?" head on the state-token
        # hidden (the same representation the skill head reads). Training-dynamics
        # probe only — not used at inference.
        self.has_logic = bool(logic)
        if self.has_logic:
            self.logic_head = nn.Linear(cfg.n_embd, 1)
        self.type_emb = nn.Embedding(len(self.names), cfg.n_embd)
        self.backbone = GPT2Backbone(cfg)

    def logic_terms(self, hidden, target_ids, attention_mask, switch_mask,
                    pos_weight=None):
        """Weighted-BCE logic-gate loss + stats at the state-token (skill-target)
        positions. ``switch_mask`` is the 0/1 switch label; ``pos_weight`` upweights
        the rare positives. Returns (loss, stats) or (None, {})."""
        if not self.has_logic:
            return None, {}
        # only score the logic gate on episodes that CONTAIN a switch — the pure
        # no-switch (DART) episodes are all-negative and would bias it to "never
        # switch". Restrict to state-token positions in switch-containing rows.
        has_switch = switch_mask.sum(dim=1, keepdim=True) > 0      # [B,1]
        sel = ((target_ids == self.id_of["skill"]) & (attention_mask > 0)
               & has_switch)
        if not sel.any():
            return None, {}
        logit = self.logic_head(hidden[sel]).squeeze(-1)
        label = switch_mask[sel].float()
        pw = (torch.tensor(float(pos_weight), device=logit.device)
              if pos_weight else None)
        loss = F.binary_cross_entropy_with_logits(logit, label, pos_weight=pw)
        with torch.no_grad():
            pred = (logit > 0).float()
            tp = float(((pred == 1) & (label == 1)).sum())
            fp = float(((pred == 1) & (label == 0)).sum())
            fn = float(((pred == 0) & (label == 1)).sum())
            pos = label == 1
            stats = {"bce": float(loss),
                     "acc": float((pred == label).float().mean()),
                     "precision": tp / (tp + fp + 1e-8),
                     "recall": tp / (tp + fn + 1e-8),
                     "pos_rate_pred": float(pred.mean()),
                     "p_switch_at_switch": (float(torch.sigmoid(logit[pos]).mean())
                                            if bool(pos.any()) else 0.0)}
        return loss, stats

    def _bin_centers(self, idx):
        """Map per-dim bin indices -> continuous values at bin centers on [-1,1]."""
        return -1.0 + (idx.float() + 0.5) * (2.0 / self.n_bins)

    def head_mean(self, name, h):
        """Point prediction / Gaussian mean / disc bin-center for rollout."""
        out = self.heads[name](h)
        ht = self.head_types[name]
        if ht == "gaussian":
            return out[..., :self.dims[name]]
        if ht == "disc":
            d = self.dims[name]
            logits = out.reshape(*out.shape[:-1], d, self.n_bins)
            return self._bin_centers(logits.argmax(-1))
        return out

    def head_logits(self, name, h):
        """Class logits for a categorical (discrete) modality head."""
        return self.heads[name](h)

    @torch.no_grad()
    def target_loss_per_position(self, name, token_ids, values, attention_mask,
                                 target_ids, target_values, loss_mask):
        """Per-position loss for target modality ``name`` with NO reduction over
        positions. Returns (loss [B,L], sel [B,L] bool) where sel marks the
        supervised positions whose target is ``name`` — for grouped diagnostics
        (e.g. action NLL broken down per skill)."""
        hidden = self.backbone(self.embed(token_ids, values), attention_mask)
        sel = (loss_mask > 0) & (attention_mask > 0) & (target_ids == self.id_of[name])
        loss = hidden.new_zeros(token_ids.shape)
        if sel.any():
            loss[sel] = self._loss_per_pos(name, hidden[sel], target_values[sel])
        return loss, sel

    def embed(self, token_ids, values):
        """token_ids [B,L] (modality id), values [B,L,max_dim] -> embeds [B,L,E].

        Discrete modalities read a class index from values[...,0]; continuous ones
        read their first `dim` entries."""
        B, L = token_ids.shape
        out = torch.zeros(B, L, self.cfg.n_embd, device=token_ids.device)
        for name in self.names:
            mid = self.id_of[name]
            sel = token_ids == mid
            if not sel.any():
                continue
            if name in self.discrete:
                out[sel] = self.encoders[name](values[sel][:, 0].long())
            else:
                d = self.dims[name]
                out[sel] = self.encoders[name](values[sel][:, :d])
        return out + self.type_emb(token_ids)

    def _loss_per_pos(self, name, h_sel, tgt_sel):
        """Per-position loss [k] for modality ``name`` over selected hidden states
        `h_sel` [k,n_embd] and targets `tgt_sel` [k,max_dim]. Differentiable."""
        d = self.dims[name]
        ht = self.head_types[name]
        out = self.heads[name](h_sel)
        if ht == "categorical":
            return F.cross_entropy(out, tgt_sel[:, 0].long(), reduction="none")
        if ht == "disc":
            # per-dim categorical over n_bins on [-1,1]: continuous target -> bins
            logits = out.reshape(-1, d, self.n_bins)
            tgt = tgt_sel[:, :d].clamp(-1, 1)
            idx = ((tgt + 1.0) * 0.5 * self.n_bins).long().clamp(0, self.n_bins - 1)
            ce = F.cross_entropy(logits.reshape(-1, self.n_bins), idx.reshape(-1),
                                 reduction="none").reshape(-1, d)
            return ce.sum(-1)                    # sum CE over action dims
        if ht == "gaussian":
            tgt = tgt_sel[:, :d]
            mu = out[:, :d]
            log_std = out[:, d:2 * d].clamp(self.log_std_min, self.log_std_max)
            nll = (0.5 * ((tgt - mu) ** 2) * torch.exp(-2 * log_std)
                   + log_std + 0.5 * math.log(2 * math.pi))
            return nll.sum(-1)
        return ((out - tgt_sel[:, :d]) ** 2).mean(-1)

    def _loss_for(self, name, h_sel, tgt_sel):
        """Mean per-position loss for modality ``name``."""
        return self._loss_per_pos(name, h_sel, tgt_sel).mean()

    def modality_loss(self, name, hidden, target_ids, target_values,
                      loss_mask, attention_mask):
        """Differentiable mean loss for one target modality from precomputed
        ``hidden`` (shares the forward pass). Returns None if no targets in this
        batch. Used by the per-term gradient-norm diagnostic."""
        sel = (loss_mask > 0) & (attention_mask > 0) & (target_ids == self.id_of[name])
        if not sel.any():
            return None
        return self._loss_for(name, hidden[sel], target_values[sel])

    def forward(self, token_ids, values, attention_mask,
                target_ids, target_values, loss_mask, loss_weight=None,
                switch_mask=None, logic_pos_weight=None):
        """Per-position prediction with masking and class balancing.

        Same masked/parallel scheme as before (per-position loss_weight + per-
        modality lambda). If the logic gate is enabled, also returns its weighted-
        BCE loss + stats from the shared hidden (training-dynamics probe).

        Returns (token_loss, per_modality_loss_dict, n_terms, logic_loss,
        logic_stats); logic_loss is None when the gate is off."""
        hidden = self.backbone(self.embed(token_ids, values), attention_mask)
        mask = (loss_mask > 0) & (attention_mask > 0)
        if loss_weight is None:
            loss_weight = torch.ones_like(loss_mask)

        total = hidden.new_zeros(())
        n_terms = 0
        per_mod = {}
        for name in self.names:
            sel = mask & (target_ids == self.id_of[name])
            cnt = int(sel.sum())
            if cnt == 0:
                continue
            l = self._loss_per_pos(name, hidden[sel], target_values[sel])  # [k]
            w = loss_weight[sel]
            loss_m = (w * l).sum() / w.sum().clamp_min(1e-8)   # weighted mean
            lam = self.loss_weights.get(name, 1.0)
            total = total + lam * loss_m
            n_terms += cnt
            per_mod[name] = float(loss_m)
        logic_loss, logic_stats = (self.logic_terms(
            hidden, target_ids, attention_mask, switch_mask, logic_pos_weight)
            if (self.has_logic and switch_mask is not None) else (None, {}))
        return total, per_mod, n_terms, logic_loss, logic_stats
