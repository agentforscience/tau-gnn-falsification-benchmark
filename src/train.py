"""Training loops for the GNN models.

Protocol (identical for every GNN variant, so ablations are comparable):
  * The held-out experiment is never touched during fitting.
  * One *training* experiment is set aside as an inner validation fold for epoch selection,
    rotated by fold index so no single experiment is always the validation set.
  * Adam, cosine-decayed learning rate, fixed epoch budget, best-val-epoch checkpointing.
  * Repeated over `n_seeds` random initialisations; we report mean +- SD across seeds, and
    seed variance is itself a reported result (data this small makes it non-negligible).
"""
from __future__ import annotations

import copy

import numpy as np
import torch

from gnn import TauODE, corr_loss, set_seed
from metrics import pearson, transform


def _to_t(x, dtype=torch.float32):
    return torch.tensor(np.asarray(x), dtype=dtype)


def groups_to_tensors(groups, genes_np, A_ant, A_ret):
    """Pre-convert a list of Groups plus the shared graph into torch tensors."""
    g = _to_t(genes_np) if (genes_np is not None and genes_np.shape[1] > 0) else None
    return {
        "A_ant": _to_t(A_ant), "A_ret": _to_t(A_ret), "g": g,
        "groups": [{
            "seeds": _to_t(gr.seeds), "times": gr.times,
            "targets": [_to_t(e.path) for e in gr.exps],
            "idx": [torch.tensor(e.idx, dtype=torch.long) for e in gr.exps],
        } for gr in groups],
    }


def _loss_over(pack, model, scale):
    """Mean (1 - Pearson R) across every experiment x timepoint in `pack`."""
    losses = []
    for gr in pack["groups"]:
        P = model(gr["seeds"], pack["A_ant"], pack["A_ret"], gr["times"], pack["g"])
        for b, (tgt, idx) in enumerate(zip(gr["targets"], gr["idx"])):
            for k in range(tgt.shape[0]):
                losses.append(corr_loss(P[k][idx, b], tgt[k], scale))
    return torch.stack(losses).mean()


def inv_softplus(y: float) -> float:
    """Inverse of softplus, used to warm-start TauODE at a fitted classical solution."""
    y = float(max(y, 1e-6))
    return float(np.log(np.expm1(min(y, 30.0)))) if y < 30 else y


def train_tauode(train_groups, val_groups, genes_np, A_ant, A_ret, *, seed: int = 0,
                 epochs: int = 150, lr: float = 0.05, scale: str = "raw",
                 model_kwargs: dict | None = None, weight_decay: float = 0.0,
                 warm_start: dict | None = None, verbose: bool = False):
    """Fit one TauODE. Returns (model, history dict).

    `warm_start` optionally initialises the diffusion rate / growth rate / direction blend at
    the values fitted for the classical FKPP-NexIS baseline on the SAME training data. This is
    deliberate: because TauODE strictly contains FKPP, warm-starting means the GNN begins at
    the classical solution and any subsequent difference is attributable to the learned terms
    rather than to optimiser luck on a non-convex surface. Ablations that share the warm start
    are therefore directly comparable.
    """
    set_seed(seed)
    n_genes = 0 if genes_np is None else genes_np.shape[1]
    model = TauODE(n_genes=n_genes, **(model_kwargs or {}))
    if warm_start:
        with torch.no_grad():
            if "rate" in warm_start:
                model.log_gamma.fill_(inv_softplus(warm_start["rate"]))
            if "growth" in warm_start:
                model.log_alpha.fill_(inv_softplus(warm_start["growth"]))
            if "dir_logit" in warm_start:
                model.dir_logit.fill_(float(warm_start["dir_logit"]))
    tr = groups_to_tensors(train_groups, genes_np, A_ant, A_ret)
    va = groups_to_tensors(val_groups, genes_np, A_ant, A_ret) if val_groups else None
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    best = {"val": np.inf, "state": copy.deepcopy(model.state_dict()), "epoch": -1}
    hist = {"train": [], "val": []}
    for ep in range(epochs):
        model.train()
        opt.zero_grad()
        loss = _loss_over(tr, model, scale)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        sched.step()
        hist["train"].append(float(loss.detach()))
        if va is not None and (ep % 3 == 0 or ep == epochs - 1):
            model.eval()
            with torch.no_grad():
                v = float(_loss_over(va, model, scale))
            hist["val"].append((ep, v))
            if v < best["val"] - 1e-5:
                best = {"val": v, "state": copy.deepcopy(model.state_dict()), "epoch": ep}
        if verbose and ep % 25 == 0:
            print(f"    ep{ep:4d} train={float(loss):.4f}"
                  + (f" val={hist['val'][-1][1]:.4f}" if hist["val"] else ""))
    if va is not None:
        model.load_state_dict(best["state"])
    hist["best_epoch"] = best["epoch"]
    hist["best_val"] = best["val"]
    return model, hist


@torch.no_grad()
def predict_groups(model, groups, genes_np, A_ant, A_ret):
    """Return {experiment_name: (T, n_measured) prediction} for a list of Groups."""
    model.eval()
    pack = groups_to_tensors(groups, genes_np, A_ant, A_ret)
    out = {}
    for gr_t, gr in zip(pack["groups"], groups):
        P = model(gr_t["seeds"], pack["A_ant"], pack["A_ret"], gr_t["times"], pack["g"])
        for b, e in enumerate(gr.exps):
            out[e.name] = P[:, e.idx, b].numpy()
    return out


def score_predictions(preds: dict, exps, scale: str = "raw") -> float:
    """Mean spatial Pearson R across experiments x timepoints (same metric as the baselines)."""
    rs = []
    for e in exps:
        P = preds[e.name]
        for k in range(P.shape[0]):
            r = pearson(transform(e.path[k], scale), transform(P[k], scale))
            if np.isfinite(r):
                rs.append(r)
    return float(np.mean(rs)) if rs else np.nan
