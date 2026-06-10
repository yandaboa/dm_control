"""Train TrajectoryGPT (GPT2 backbone) on a trajectory store, driven by a YAML config.

All knobs live in the config file (see configs/bc_skill0.yaml): the token layout
(``pattern`` of per-timestep slots, each an input modality or [input, target]
pair), the GPT2 backbone size, and the optimizer. Loss is masked per-position
regression, so padded positions and untargeted tokens cost nothing.

    python -m src.smerl.train_seq --config src/smerl/configs/bc_skill0.yaml
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler, random_split

from src.smerl.seq_data import SeqDataset, SequenceSpec, collate
from src.smerl.seq_model import TrajectoryGPT, GPT2Config


def _resolve(path: str) -> str:
    return path if os.path.isabs(path) else os.path.join("src/smerl", path)


def evaluate(model, loader, device):
    model.eval()
    tot, n = 0.0, 0
    per_tot = {}
    with torch.no_grad():
        for b in loader:
            loss, per, k, _, _ = model(
                b["token_ids"].to(device), b["values"].to(device),
                b["attention_mask"].to(device), b["target_ids"].to(device),
                b["target_values"].to(device), b["loss_mask"].to(device),
                b["loss_weight"].to(device))
            if k:
                tot += float(loss) * k
                n += k
                for name, v in per.items():
                    per_tot[name] = per_tot.get(name, 0.0) + v * k
    model.train()
    per_avg = {f"val_{name}": v / max(n, 1) for name, v in per_tot.items()}
    return tot / max(n, 1), per_avg


def per_skill_action_loss(model, loader, device, n_skills):
    """Open-loop (teacher-forced) action NLL broken down by the episode's skill z.
    Every action target in an episode belongs to that episode's skill, so we group
    per-position action losses by the batch's per-row z."""
    sums = [0.0] * n_skills
    cnts = [0] * n_skills
    model.eval()
    with torch.no_grad():
        for b in loader:
            loss, sel = model.target_loss_per_position(
                "action", b["token_ids"].to(device), b["values"].to(device),
                b["attention_mask"].to(device), b["target_ids"].to(device),
                b["target_values"].to(device), b["loss_mask"].to(device))
            loss = loss.cpu(); sel = sel.cpu()
            z = b["z"]
            for i in range(len(z)):
                m = sel[i]
                k = int(m.sum())
                if k:
                    sums[int(z[i])] += float(loss[i][m].sum())
                    cnts[int(z[i])] += k
    model.train()
    return {f"act_z{zz}": sums[zz] / max(cnts[zz], 1) for zz in range(n_skills)}


def switch_metrics(model, loader, device, max_pos=12):
    """Offline analytic for whether the SKILL HEAD anticipates a switch, measured
    over the FULL decoupled decision window (every switch_mask position):
        acc    : argmax skill == new skill (the one switched to)
        p_new  : mean P(new skill)        (want UP if the skill head learns it)
        p_stay : mean P(current skill)    (want DOWN)
    Also breaks P(new) and the logic-gate switch probability down by within-window
    position (pnew_pos{k}, gate_pos{k}), so we can watch the switch probability climb
    as more failure-feedback tokens accrue in context (k = 0,1,2,... into the window).
    """
    if "skill" not in model.discrete:
        return {}
    has_logic = model.has_logic
    model.eval()
    n = acc = p_new = p_stay = 0
    pnew_sum = [0.0] * max_pos
    gate_sum = [0.0] * max_pos
    pos_cnt = [0] * max_pos
    with torch.no_grad():
        for b in loader:
            sm = b["switch_mask"].to(device)
            if int(sm.sum()) == 0:
                continue
            hidden = model.backbone(
                model.embed(b["token_ids"].to(device), b["values"].to(device)),
                b["attention_mask"].to(device))
            sel = sm > 0
            hsel = hidden[sel]
            probs = model.head_logits("skill", hsel).softmax(-1)
            new = b["new_id"].to(device)[sel]
            stay = b["stay_id"].to(device)[sel]
            spos = b["switch_pos"].to(device)[sel]
            rows = torch.arange(len(new), device=device)
            pnew_tok = probs[rows, new]
            acc += int((probs.argmax(-1) == new).sum())
            p_new += float(pnew_tok.sum())
            p_stay += float(probs[rows, stay].sum())
            n += int(sel.sum())
            gate_tok = (torch.sigmoid(model.logic_head(hsel).squeeze(-1))
                        if has_logic else None)
            sp = spos.cpu().numpy()
            pn = pnew_tok.cpu().numpy()
            gt = gate_tok.cpu().numpy() if has_logic else None
            for i in range(len(sp)):
                p = int(sp[i])
                if 0 <= p < max_pos:
                    pnew_sum[p] += float(pn[i])
                    pos_cnt[p] += 1
                    if has_logic:
                        gate_sum[p] += float(gt[i])
    model.train()
    if n == 0:
        return {}
    out = {"acc": acc / n, "p_new": p_new / n, "p_stay": p_stay / n}
    for p in range(max_pos):
        if pos_cnt[p]:
            out[f"pnew_pos{p}"] = pnew_sum[p] / pos_cnt[p]
            if has_logic:
                out[f"gate_pos{p}"] = gate_sum[p] / pos_cnt[p]
    return out


def _global_norm(grads):
    sq = sum(float((g ** 2).sum()) for g in grads if g is not None)
    return sq ** 0.5


def _flat(grads):
    import torch as _t
    parts = [g.reshape(-1) for g in grads if g is not None]
    return _t.cat(parts) if parts else None


def grad_norm_diag(model, batch, device, terms=None):
    """Per-term gradient-norm diagnostic — ALWAYS run periodically by default.

    Computes one shared forward, then uses torch.autograd.grad (functional — never
    touches .grad) once per term so the norms stay isolated (sequential .backward()
    would accumulate). Reports each term's total norm, its backbone-restricted norm
    (`_bb` — the shared trunk the objectives actually compete over), and the cosine
    between term backbone gradients (negative => objectives conflict over the trunk).

    ``terms`` defaults to whatever target modalities are actually supervised in the
    batch, so it never silently drops a head if the token layout changes."""
    import torch
    model.eval()
    if terms is None:                       # auto-derive supervised target modalities
        tids = batch["target_ids"][batch["loss_mask"] > 0]
        id2name = {i: n for n, i in model.id_of.items()}
        terms = [id2name[int(i)] for i in torch.unique(tids).tolist()]
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    names = [n for n, _ in named]
    ps = [p for _, p in named]
    bb = [("backbone" in n) for n in names]
    hidden = model.backbone(
        model.embed(batch["token_ids"].to(device), batch["values"].to(device)),
        batch["attention_mask"].to(device))
    out, gbb = {}, {}
    for name in terms:
        L = model.modality_loss(name, hidden, batch["target_ids"].to(device),
                                batch["target_values"].to(device),
                                batch["loss_mask"].to(device),
                                batch["attention_mask"].to(device))
        if L is None:
            continue
        g = torch.autograd.grad(L, ps, retain_graph=True, allow_unused=True)
        out[f"gradnorm_{name}"] = _global_norm(g)
        out[f"gradnorm_{name}_bb"] = _global_norm(
            [gg for gg, is_bb in zip(g, bb) if is_bb])
        gbb[name] = _flat([gg for gg, is_bb in zip(g, bb) if is_bb])
    # pairwise backbone-gradient cosine (keep the bare key for the common 2-term case)
    keys = list(gbb)
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            a, b = gbb[keys[i]], gbb[keys[j]]
            if a is None or b is None:
                continue
            cos = float(torch.dot(a, b) / (a.norm() * b.norm()).clamp_min(1e-12))
            out["grad_cos_bb" if len(keys) == 2
                else f"grad_cos_{keys[i]}_{keys[j]}_bb"] = cos
    model.zero_grad(set_to_none=True)
    model.train()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--limit", type=int, default=None,
                    help="cap # episodes used (nested subset for data ablation)")
    ap.add_argument("--device", type=str, default=None, help="override config")
    ap.add_argument("--out", type=str, default=None, help="override config")
    ap.add_argument("--run-name", type=str, default=None, help="wandb run name")
    ap.add_argument("--wandb-project", type=str, default="2d")
    ap.add_argument("--wandb-group", type=str, default=None)
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--switch-share", type=float, default=None,
                    help="override balance_switch target gradient share for the "
                         "decision group (0.5 = equal switch/non-switch)")
    ap.add_argument("--batch", type=int, default=None, help="override optim.batch")
    ap.add_argument("--seed", type=int, default=None, help="override config seed")
    ap.add_argument("--warmup-frac", type=float, default=None,
                    help="override optim.warmup_frac (fraction of total steps for "
                         "linear LR warmup before cosine decay)")
    ap.add_argument("--rope", action="store_true",
                    help="use rotary (relative) positions instead of learned wpe")
    ap.add_argument("--save-every", type=int, default=0,
                    help="also save a raw checkpoint every N epochs (for "
                         "closed-loop checkpoint selection)")
    ap.add_argument("--no-bf16", action="store_true",
                    help="disable bf16 autocast (default: ON for cuda)")
    ap.add_argument("--no-data-cache", action="store_true",
                    help="disable in-RAM memoization of tokenized episodes")
    args = ap.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    seed = int(args.seed if args.seed is not None else cfg.get("seed", 0))
    torch.manual_seed(seed)
    dev_str = args.device or cfg.get(
        "device", "cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(dev_str)
    use_bf16 = device.type == "cuda" and not args.no_bf16
    import contextlib as _ctxlib
    def _amp():
        return (torch.autocast("cuda", dtype=torch.bfloat16) if use_bf16
                else _ctxlib.nullcontext())

    # data.store may be a single path or a list of paths (DAgger merge)
    raw_store = cfg["data"]["store"]
    store = ([_resolve(s) for s in raw_store] if isinstance(raw_store, list)
             else _resolve(raw_store))
    switch_share = float(args.switch_share if args.switch_share is not None
                         else cfg.get("switch_share", 0.5))
    spec = SequenceSpec(pattern=cfg["pattern"],
                        balance_switch=bool(cfg.get("balance_switch", True)),
                        switch_share=switch_share,
                        switch_weight=float(cfg.get("switch_weight", 1.0)),
                        decouple=bool(cfg.get("decouple", True)),
                        expert_gated=bool(cfg.get("expert_gated", False)))
    full_ds = SeqDataset(store, spec, cache=not args.no_data_cache)
    modalities = full_ds.modalities
    n_skills = int(full_ds.man["meta"]["n_skills"])
    # the "skill" modality is a discrete index over n_skills (embedding + categorical)
    discrete = {"skill": n_skills} if "skill" in dict(modalities) else None
    # nested subset for the data-scale ablation: fixed permutation, take a prefix
    if args.limit is not None and args.limit < len(full_ds):
        perm = torch.randperm(len(full_ds),
                              generator=torch.Generator().manual_seed(123))
        ds = Subset(full_ds, perm[:args.limit].tolist())
    else:
        ds = full_ds

    opt_cfg = cfg.get("optim", {})
    batch = int(args.batch if args.batch is not None else opt_cfg.get("batch", 64))
    epochs = int(opt_cfg.get("epochs", 50))
    lr = float(opt_cfg.get("lr", 3e-4))
    wd = float(opt_cfg.get("weight_decay", 1e-2))
    grad_clip = float(opt_cfg.get("grad_clip", 1.0))
    val_frac = float(opt_cfg.get("val_frac", 0.1))

    n_val = max(1, int(val_frac * len(ds)))
    g = torch.Generator().manual_seed(seed)
    tr, va = random_split(ds, [len(ds) - n_val, n_val], generator=g)
    seq_len = max(ds[i]["length"] for i in range(min(64, len(ds))))
    print(f"[data] store={store}")
    print(f"[data] {len(ds)} episodes  pattern={cfg['pattern']}  "
          f"~seq_len>={seq_len}  modalities={[n for n,_ in modalities]}")

    pin = device.type == "cuda"
    # optional per-store sampling frequencies (DAgger recency weighting): config
    # data.store_weights = [w_0, ..., w_{S-1}] aligned with data.store. Each
    # epoch draws len(tr) episodes WITH replacement; an episode's probability is
    # w_s / n_s(train split), so each store contributes a w_s share of gradient
    # steps regardless of its size.
    store_weights = cfg["data"].get("store_weights")
    sampler = None
    if store_weights is not None:
        dirs = store if isinstance(store, list) else [store]
        assert len(store_weights) == len(dirs), \
            f"store_weights has {len(store_weights)} entries for {len(dirs)} stores"
        s_of = {d: si for si, d in enumerate(dirs)}

        def _full_idx(subset_i):                 # tr index -> full_ds index
            i, d = subset_i, tr
            while isinstance(d, Subset):
                i, d = d.indices[i], d.dataset
            return i
        tr_store = [s_of[full_ds.items[_full_idx(i)][0]] for i in range(len(tr))]
        n_per = np.bincount(tr_store, minlength=len(dirs))
        w_ep = [store_weights[s] / max(int(n_per[s]), 1) for s in tr_store]
        sampler = WeightedRandomSampler(
            w_ep, num_samples=len(tr), replacement=True,
            generator=torch.Generator().manual_seed(seed))
        print(f"[data] store_weights={store_weights}  train eps/store={n_per.tolist()}")
    dl_tr = DataLoader(tr, batch_size=batch, shuffle=sampler is None, sampler=sampler,
                       collate_fn=collate, pin_memory=pin)
    dl_va = DataLoader(va, batch_size=batch, shuffle=False, collate_fn=collate, pin_memory=pin)
    # stable loader for the heavy diagnostics — SUBSAMPLED (they're metrics,
    # and full-corpus passes stall the step stream for minutes at 5-10k eps)
    n_diag = min(len(ds), 512)
    diag_idx = torch.randperm(len(ds), generator=torch.Generator().manual_seed(11))[:n_diag].tolist()
    dl_diag = DataLoader(Subset(ds, diag_idx), batch_size=batch, shuffle=False,
                         collate_fn=collate, pin_memory=pin)

    m_cfg = cfg.get("model", {})
    gcfg = GPT2Config(n_embd=int(m_cfg.get("n_embd", 128)),
                      n_layer=int(m_cfg.get("n_layer", 4)),
                      n_head=int(m_cfg.get("n_head", 4)),
                      n_positions=int(m_cfg.get("n_positions", max(1024, 4 * seq_len))),
                      dropout=float(m_cfg.get("dropout", 0.1)),
                      rope=bool(args.rope or m_cfg.get("rope", False)))
    head = m_cfg.get("head", "mse")
    head_types = m_cfg.get("heads", None)   # {modality: mse|gaussian|categorical|disc}
    loss_weights = m_cfg.get("loss_weights", None)   # per-modality lambda (layer 2)
    n_bins = int(m_cfg.get("n_bins", 21))
    use_logic = bool(cfg.get("logic", False))         # logic-gate aux probe
    logic_weight = float(cfg.get("logic_weight", 1.0))
    model = TrajectoryGPT(modalities, gcfg, head=head, discrete=discrete,
                          head_types=head_types, loss_weights=loss_weights,
                          n_bins=n_bins, logic=use_logic).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    # LR schedule: linear warmup -> cosine decay to lr_min_frac*lr, stepped per batch.
    # config optim.lr_schedule = "cosine" (default) | "constant"; optim.warmup_frac;
    # optim.lr_min_frac. warmup_frac=0 -> no warmup (cosine from step 0).
    lr_kind = str(opt_cfg.get("lr_schedule", "constant")).lower()
    warmup_frac = float(args.warmup_frac if args.warmup_frac is not None
                        else opt_cfg.get("warmup_frac", 0.05))
    lr_min_frac = float(opt_cfg.get("lr_min_frac", 0.0))
    total_steps = max(1, epochs * len(dl_tr))
    warmup_steps = int(warmup_frac * total_steps) if warmup_frac > 0 else 0

    def _lr_lambda(step):
        if lr_kind in ("constant", "none"):
            return 1.0
        if warmup_steps and step < warmup_steps:
            return (step + 1) / warmup_steps
        prog = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return lr_min_frac + (1.0 - lr_min_frac) * 0.5 * (1.0 + float(np.cos(np.pi * min(1.0, prog))))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, _lr_lambda)
    print(f"[lr] schedule={lr_kind} peak={lr} warmup={warmup_steps}/{total_steps} steps "
          f"min_frac={lr_min_frac}")

    # logic-gate pos_weight = mean length (in state tokens) of switch-containing
    # episodes, so the rare per-episode positive is upweighted accordingly
    logic_pos_weight = None
    if use_logic:
        if cfg.get("logic_pos_weight") is not None:
            logic_pos_weight = float(cfg["logic_pos_weight"])
            print(f"[logic] pos_weight={logic_pos_weight} (hardcoded)  "
                  f"logic_weight={logic_weight}  (loss on switch episodes only)")
        else:
            sk_id = model.id_of["skill"]
            # estimate on a SUBSAMPLE: it's a token-class ratio, and a full-corpus
            # pass costs minutes of silent single-thread CPU at 5-10k episodes
            n_est = min(len(ds), 512)
            est_idx = torch.randperm(
                len(ds), generator=torch.Generator().manual_seed(7))[:n_est].tolist()
            dl_est = DataLoader(Subset(ds, est_idx), batch_size=batch,
                                shuffle=False, collate_fn=collate, pin_memory=pin)
            n_pos = n_neg = 0
            for b in dl_est:
                st = (b["target_ids"] == sk_id) & (b["attention_mask"] > 0)
                sw = b["switch_mask"] > 0
                has_sw = b["switch_mask"].sum(1) > 0
                for i in range(len(has_sw)):
                    if bool(has_sw[i]):
                        npos = int(sw[i].sum())
                        n_pos += npos
                        n_neg += int(st[i].sum()) - npos
            logic_pos_weight = (n_neg / n_pos) if n_pos else 1.0
            print(f"[logic] pos_weight={logic_pos_weight:.1f} (neg/pos over switch "
                  f"episodes, {n_est}-episode estimate; {n_pos} switch tokens)  "
                  f"logic_weight={logic_weight}")

    print(f"[model] GPT2  n_embd={gcfg.n_embd} n_layer={gcfg.n_layer} "
          f"n_head={gcfg.n_head} n_pos={gcfg.n_positions}  "
          f"heads={model.head_types}  loss_weights={model.loss_weights}  "
          f"logic={use_logic}  params={n_params/1e6:.2f}M  device={device}")
    print(f"[optim] bf16={use_bf16} data_cache={not args.no_data_cache} "
          f"epochs={epochs} batch={batch} lr={lr} wd={wd}")

    n_data = len(ds)
    wb = None
    if not args.no_wandb:
        import wandb
        wb = wandb.init(
            project=args.wandb_project,
            name=args.run_name or f"bc_skill0_n{n_data}",
            group=args.wandb_group,
            config={"n_data": n_data, "n_train": len(tr), "n_val": len(va),
                    "pattern": cfg["pattern"], "params": n_params, "head": head,
                    "n_embd": gcfg.n_embd, "n_layer": gcfg.n_layer,
                    "n_head": gcfg.n_head, "epochs": epochs, "batch": batch,
                    "lr": lr, "weight_decay": wd, "store": store,
                    "store_weights": store_weights,
                    "device": str(device), "seed": seed})

    out_cfg = args.out or cfg.get("out")
    if out_cfg is not None:
        out = _resolve(out_cfg)
    else:                                    # default beside the (first) store
        base = store[0] if isinstance(store, list) else store
        out = os.path.join(base, "seq_gpt2.pt")

    def save_ckpt(path):
        torch.save({"state_dict": model.state_dict(), "modalities": modalities,
                    "cfg": vars(gcfg), "pattern": cfg["pattern"], "head": head,
                    "discrete": model.discrete, "head_types": model.head_types,
                    "loss_weights": model.loss_weights, "n_bins": model.n_bins,
                    "logic": model.has_logic}, path)

    ckpt_dir = None
    if args.save_every > 0:
        ckpt_dir = os.path.splitext(out)[0] + "_ckpts"
        os.makedirs(ckpt_dir, exist_ok=True)
    # one fixed batch for the per-term gradient-norm diagnostic
    diag_batch = next(iter(dl_diag))
    # per-skill action loss needs the skill modality; grad-norm logging is always on
    do_perskill = "skill" in model.discrete and "action" in model.names

    best = float("inf")
    for ep in range(epochs):
        run, n = 0.0, 0
        logic_run = {}
        for b in dl_tr:
            with _amp():
                tok_loss, per, k, logic_loss, logic_stats = model(
                    b["token_ids"].to(device), b["values"].to(device),
                    b["attention_mask"].to(device), b["target_ids"].to(device),
                    b["target_values"].to(device), b["loss_mask"].to(device),
                    b["loss_weight"].to(device), b["switch_mask"].to(device),
                    logic_pos_weight)
            if k == 0:
                continue
            loss = tok_loss
            if logic_loss is not None:
                loss = loss + logic_weight * logic_loss   # separate aux term
                for kk, vv in logic_stats.items():
                    logic_run[kk] = logic_run.get(kk, 0.0) + vv * k
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
            sched.step()
            run += float(tok_loss) * k; n += k
        val, val_per = evaluate(model, dl_va, device)
        per_skill = (per_skill_action_loss(model, dl_diag, device, n_skills)
                     if do_perskill else {})
        gnorm = grad_norm_diag(model, diag_batch, device)   # always on
        swm = switch_metrics(model, dl_diag, device)        # skill-head switch analytic
        logic_avg = {kk: vv / max(n, 1) for kk, vv in logic_run.items()}
        tr_loss = run / max(n, 1)
        flag = ""
        if val < best:
            best = val
            save_ckpt(out)
            flag = "  *"
        if ckpt_dir is not None and (ep % args.save_every == 0 or ep == epochs - 1):
            save_ckpt(os.path.join(ckpt_dir, f"ep{ep:03d}.pt"))
        if wb is not None:
            # category prefixes so wandb groups the metrics
            logd = {"epoch": ep, "loss/train": tr_loss, "loss/val": val,
                    "loss/best_val": best, "optim/lr": sched.get_last_lr()[0]}
            logd.update({f"loss/{k}": v for k, v in val_per.items()})
            logd.update({f"loss/{k}": v for k, v in per_skill.items()})
            logd.update({f"grad_norm/{k.replace('gradnorm_', '').replace('grad_', '')}"
                         : v for k, v in gnorm.items()})
            logd.update({f"switch/{k}": v for k, v in swm.items()})
            logd.update({f"logic/{k}": v for k, v in logic_avg.items()})
            wb.log(logd)
        per_str = "  ".join(f"{k}={v:.4f}" for k, v in val_per.items())
        sw_str = "  ".join(f"{k}={v:.3f}" for k, v in swm.items())
        lg_str = "  ".join(f"{k}={v:.3f}" for k, v in logic_avg.items())
        print(f"  epoch {ep:3d}  train={tr_loss:.5f}  val={val:.5f}  {per_str}  | "
              f"switch: {sw_str}  | logic: {lg_str}{flag}")

    if wb is not None:
        wb.summary["best_val"] = best
        wb.finish()
    print(f"[done] best val={best:.5f}  saved {out}")


if __name__ == "__main__":
    main()
