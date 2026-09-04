"""E0-E3: the mouse longitudinal tauopathy arm -- the primary test of H1.

Two prediction tasks, both evaluated leave-one-experiment-out (LOEO) over the 8 Kaufman
experiments (identical 194-region measurement set, identical 4/8/12-month schedule, so the
folds are exchangeable):

  TASK A  seed -> trajectory.  Given only the injection seed, predict pathology at 4, 8 and 12
          months. This is the propagation task the NDM/NexIS literature is built around, and
          the strongest available test of "does the connectome predict where tau goes".
          Null: predict the seed itself (no spreading).

  TASK B  one-step forecast.  Given *observed* pathology at t_k, predict t_{k+1}.
          Null: persistence (predict no change). This is where the Schafer trap bites, so we
          report Delta metrics (correlation of predicted change with observed change) as well
          as absolute-scale metrics.

Every model is additionally run on a ladder of null networks (Euclidean-distance graph,
degree-preserving rewired connectome, shuffled seed) to separate "the connectome matters"
from "any smooth spatial operator matters".

Output: results/mouse_taskA_loeo.csv, results/mouse_taskB_loeo.csv (tidy long format).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from baselines import SpreadModel, fit, normalize_adj, rewire_degree_preserving  # noqa: E402
from data import CORE_GENES_MOUSE, gene_features, load_mouse                     # noqa: E402
from metrics import pearson, r2_score, spearman, transform                       # noqa: E402
from simulate import TSCALE, Group, group_by_schedule, norm_times, unit_seed     # noqa: E402
from train import predict_groups, train_tauode                                   # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
os.makedirs(RES, exist_ok=True)


# --------------------------------------------------------------------------------------
def metric_row(y_true, y_pred, y_base=None) -> dict:
    """Metric block for one (model, fold, timepoint). Delta metrics need a baseline map."""
    out = {
        "R": pearson(y_true, y_pred),
        "rho": spearman(y_true, y_pred),
        "R_log": pearson(transform(y_true, "log"), transform(y_pred, "log")),
    }
    if y_base is not None:
        dt, dp = np.asarray(y_true) - y_base, np.asarray(y_pred) - y_base
        out["dR"] = pearson(dt, dp)
        out["drho"] = spearman(dt, dp)
        out["dR2"] = r2_score(dt, dp)
        out["R_persist"] = pearson(y_true, y_base)
        out["dR2_persist_zero"] = r2_score(dt, np.zeros_like(dt))
    return out


def shuffle_seeds(exps, seed: int = 0):
    """Copy experiments with their seed vectors randomly permuted across regions."""
    import copy as _c
    rng = np.random.default_rng(seed)
    out = []
    for e in exps:
        e2 = _c.copy(e)
        perm = rng.permutation(len(e.seed_full))
        e2.seed_full = e.seed_full[perm]
        e2.seed = e2.seed_full[e.idx]
        out.append(e2)
    return out


# --------------------------------------------------------------------------------------
def build_graphs(md, rewire_seed: int = 0) -> dict:
    """All connectomes and null networks, max-normalised, as (A_antero, A_retro) pairs."""
    Aa = normalize_adj(md.conn["antero"])
    Ar = normalize_adj(md.conn["retro"])
    And = normalize_adj(md.conn["nd"])
    Asp = normalize_adj(md.conn["spatial"])
    Arw = normalize_adj(rewire_degree_preserving(md.conn["retro"], seed=rewire_seed))
    return {"connectome": (Aa, Ar), "nondirected": (And, And),
            "spatial": (Asp, Asp), "rewired": (Arw, Arw.T)}


def baseline_specs(graphs, genes3):
    """(name, family, graph_key, SpreadModel kwargs, shuffle_seed) for every classical model."""
    S = []
    S.append(("Persistence (seed only)", "null", "connectome", dict(kind="persist"), False))
    S.append(("NDM retrograde", "diffusion", "connectome",
              dict(kind="ndm", fit_dir=False, dir_fixed=0.0), False))
    S.append(("NDM anterograde", "diffusion", "connectome",
              dict(kind="ndm", fit_dir=False, dir_fixed=1.0), False))
    S.append(("NDM bidirectional", "diffusion", "connectome",
              dict(kind="ndm", fit_dir=True), False))
    S.append(("FKPP / NexIS (no genes)", "diffusion", "connectome",
              dict(kind="fkpp", fit_dir=True), False))
    S.append(("NexIS + genes (MAPT/APOE/TREM2)", "diffusion+genes", "connectome",
              dict(kind="nexis", fit_dir=True, genes=genes3), False))
    # --- null networks -----------------------------------------------------------------
    S.append(("NDM on distance graph", "null-network", "spatial",
              dict(kind="ndm", fit_dir=False), False))
    S.append(("FKPP on distance graph", "null-network", "spatial",
              dict(kind="fkpp", fit_dir=False), False))
    S.append(("NDM on rewired connectome", "null-network", "rewired",
              dict(kind="ndm", fit_dir=True), False))
    S.append(("FKPP on rewired connectome", "null-network", "rewired",
              dict(kind="fkpp", fit_dir=True), False))
    S.append(("FKPP with shuffled seed", "null-network", "connectome",
              dict(kind="fkpp", fit_dir=True), True))
    return S


# --------------------------------------------------------------------------------------
def run_task_a(md, n_seeds: int = 5, epochs: int = 150, n_steps: int = 24,
               out_csv: str = "mouse_taskA_loeo.csv", verbose: bool = True,
               max_folds: int | None = None):
    """TASK A: seed -> trajectory, leave-one-experiment-out over the Kaufman set."""
    kauf = [e for e in md.experiments if e.source == "kaufman"]
    graphs = build_graphs(md)
    genes3 = gene_features(md.genes, CORE_GENES_MOUSE)
    rows = []
    t_start = time.time()

    if max_folds:
        kauf_iter = kauf[:max_folds]
    else:
        kauf_iter = kauf
    for fold, test_exp in enumerate(kauf_iter):
        train_exps = [e for e in kauf if e.name != test_exp.name]
        if verbose:
            print(f"\n[Task A] fold {fold + 1}/{len(kauf_iter)}  held out: {test_exp.name}")

        # ---- classical baselines -------------------------------------------------------
        for name, fam, gkey, kw, shuf in baseline_specs(graphs, genes3):
            Aa, Ar = graphs[gkey]
            tr_e = shuffle_seeds(train_exps, seed=fold) if shuf else train_exps
            te_e = shuffle_seeds([test_exp], seed=100 + fold) if shuf else [test_exp]
            mdl = SpreadModel(A_ant=Aa, A_ret=Ar, **kw)
            res = fit(mdl, group_by_schedule(tr_e), n_restarts=2, seed=fold)
            gr = group_by_schedule(te_e)[0]
            P = mdl.predict(gr, res["params"])
            for k, tp in enumerate(gr.times_raw):
                base = te_e[0].seed if k == 0 else None
                rows.append(dict(task="A", fold=fold, held_out=test_exp.name, model=name,
                                 family=fam, seed_run=0, timepoint=tp,
                                 train_R=res["train_R"], n_params=mdl.n_params,
                                 **metric_row(test_exp.path[k], P[k][test_exp.idx, 0],
                                              y_base=test_exp.seed)))

        # ---- GNN variants ---------------------------------------------------------------
        # Fixed epoch budget, identical for every GNN variant, and no inner validation split.
        # Rationale: with 7 training experiments an inner validation fold is a single
        # experiment, and two of the eight (DS1, DS10) carry almost no predictable signal even
        # under an in-sample fit -- drawing one of those as the validation set stops training
        # at epoch 0. A fixed budget applied identically to every variant keeps the ablation
        # comparison clean; training loss is flat well before the budget ends (see logs/).
        Aa, Ar = graphs["connectome"]
        inner = train_exps
        # warm start every GNN at the FKPP/NexIS solution fitted on this fold's training data
        _fk = SpreadModel(kind="fkpp", A_ant=Aa, A_ret=Ar, fit_dir=True)
        _fkr = fit(_fk, group_by_schedule(inner), n_restarts=2, seed=fold)
        warm = {"rate": float(np.exp(_fkr["params"][0])),
                "growth": float(np.exp(_fkr["params"][1])),
                "dir_logit": float(_fkr["params"][2])}
        gnn_specs = [
            ("TauODE (no genes)", "gnn", None, dict(n_steps=n_steps)),
            ("TauODE + genes (MAPT/APOE/TREM2)", "gnn+genes", genes3, dict(n_steps=n_steps)),
            ("TauODE no-MLP (no genes)", "gnn-ablation", None,
             dict(n_steps=n_steps, use_mlp=False)),
            ("TauODE no-MLP + genes", "gnn-ablation", genes3,
             dict(n_steps=n_steps, use_mlp=False)),
        ]
        for name, fam, gfeat, mk in gnn_specs:
            for s in range(n_seeds):
                model, hist = train_tauode(group_by_schedule(inner), None,
                                           gfeat, Aa, Ar, seed=s, epochs=epochs,
                                           model_kwargs=mk, warm_start=warm)
                pred = predict_groups(model, group_by_schedule([test_exp]), gfeat, Aa, Ar)
                P = pred[test_exp.name]
                npar = sum(p.numel() for p in model.parameters())
                for k, tp in enumerate(test_exp.timepoints):
                    rows.append(dict(task="A", fold=fold, held_out=test_exp.name, model=name,
                                     family=fam, seed_run=s, timepoint=tp,
                                     train_R=1 - hist["train"][-1], n_params=npar,
                                     **metric_row(test_exp.path[k], P[k],
                                                  y_base=test_exp.seed)))
            if verbose:
                sub = [r for r in rows if r["fold"] == fold and r["model"] == name]
                print(f"    {name:38s} testR={np.nanmean([r['R'] for r in sub]):.4f}")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RES, out_csv), index=False)
    if verbose:
        print(f"\n[Task A] done in {time.time() - t_start:.0f}s -> results/{out_csv}")
    return df


# --------------------------------------------------------------------------------------
def _onestep_groups(exps, k: int):
    """Build pseudo-Groups whose 'seed' is observed pathology at timepoint k.

    Pathology is normalised by its own maximum at the *input* timepoint (observed at
    prediction time, so no leakage) to put it on the [0, 1] scale the logistic term assumes.
    Unmeasured regions are set to zero on the full graph.
    """
    import copy as _c
    out, scales = [], {}
    for e in exps:
        e2 = _c.copy(e)
        full = np.zeros(len(e.seed_full))
        full[e.idx] = e.path[k]
        m = full.max()
        scales[e.name] = m if m > 0 else 1.0
        e2.seed_full = full / scales[e.name]
        e2.seed = e2.seed_full[e.idx]
        e2.path = e.path[k + 1: k + 2]
        e2.timepoints = [e.timepoints[k + 1] - e.timepoints[k]]
        out.append(e2)
    return group_by_schedule(out), scales


def run_task_b(md, n_seeds: int = 5, epochs: int = 120, n_steps: int = 24,
               out_csv: str = "mouse_taskB_loeo.csv", verbose: bool = True,
               max_folds: int | None = None):
    """TASK B: one-step forecast x(t_k) -> x(t_{k+1}), LOEO. Persistence is the key null."""
    kauf = [e for e in md.experiments if e.source == "kaufman"]
    graphs = build_graphs(md)
    genes3 = gene_features(md.genes, CORE_GENES_MOUSE)
    rows = []
    n_trans = len(kauf[0].timepoints) - 1
    t_start = time.time()

    kauf_iter = kauf[:max_folds] if max_folds else kauf
    for fold, test_exp in enumerate(kauf_iter):
        train_exps = [e for e in kauf if e.name != test_exp.name]
        if verbose:
            print(f"\n[Task B] fold {fold + 1}/{len(kauf_iter)}  held out: {test_exp.name}")
        for k in range(n_trans):
            tr_groups, _ = _onestep_groups(train_exps, k)
            te_groups, te_scale = _onestep_groups([test_exp], k)
            gr = te_groups[0]
            base = test_exp.path[k]
            targ = test_exp.path[k + 1]
            tp_lab = f"{test_exp.timepoints[k]:.0f}->{test_exp.timepoints[k + 1]:.0f}"

            # persistence: predict no change
            rows.append(dict(task="B", fold=fold, held_out=test_exp.name,
                             model="Persistence (no change)", family="null", seed_run=0,
                             timepoint=tp_lab, train_R=np.nan, n_params=0,
                             **metric_row(targ, base, y_base=base)))

            for name, fam, gkey, kw, shuf in baseline_specs(graphs, genes3):
                if kw["kind"] == "persist" or shuf:
                    continue                                   # already covered / not applicable
                Aa, Ar = graphs[gkey]
                mdl = SpreadModel(A_ant=Aa, A_ret=Ar, **kw)
                res = fit(mdl, tr_groups, n_restarts=2, seed=fold)
                P = mdl.predict(gr, res["params"])
                yp = P[0][test_exp.idx, 0] * te_scale[test_exp.name]
                rows.append(dict(task="B", fold=fold, held_out=test_exp.name, model=name,
                                 family=fam, seed_run=0, timepoint=tp_lab,
                                 train_R=res["train_R"], n_params=mdl.n_params,
                                 **metric_row(targ, yp, y_base=base)))

            Aa, Ar = graphs["connectome"]
            inner_tr, _ = _onestep_groups(train_exps, k)
            inner_va = None
            _fk = SpreadModel(kind="fkpp", A_ant=Aa, A_ret=Ar, fit_dir=True)
            _fkr = fit(_fk, inner_tr, n_restarts=2, seed=fold)
            warm = {"rate": float(np.exp(_fkr["params"][0])),
                    "growth": float(np.exp(_fkr["params"][1])),
                    "dir_logit": float(_fkr["params"][2])}
            for name, fam, gfeat in [("TauODE (no genes)", "gnn", None),
                                     ("TauODE + genes (MAPT/APOE/TREM2)", "gnn+genes", genes3)]:
                for s in range(n_seeds):
                    model, hist = train_tauode(inner_tr, inner_va, gfeat, Aa, Ar, seed=s,
                                               epochs=epochs, warm_start=warm,
                                               model_kwargs=dict(n_steps=n_steps))
                    pred = predict_groups(model, te_groups, gfeat, Aa, Ar)
                    yp = pred[test_exp.name][0] * te_scale[test_exp.name]
                    rows.append(dict(task="B", fold=fold, held_out=test_exp.name, model=name,
                                     family=fam, seed_run=s, timepoint=tp_lab,
                                     train_R=1 - hist["train"][-1],
                                     n_params=sum(p.numel() for p in model.parameters()),
                                     **metric_row(targ, yp, y_base=base)))
        if verbose:
            sub = pd.DataFrame([r for r in rows if r["fold"] == fold])
            print(sub.groupby("model")[["R", "dR"]].mean().round(3).to_string())

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RES, out_csv), index=False)
    if verbose:
        print(f"\n[Task B] done in {time.time() - t_start:.0f}s -> results/{out_csv}")
    return df


# --------------------------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["A", "B", "both"], default="both")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--steps", type=int, default=24)
    ap.add_argument("--folds", type=int, default=None)
    a = ap.parse_args()
    md = load_mouse()
    if a.task in ("A", "both"):
        run_task_a(md, n_seeds=a.seeds, epochs=a.epochs, n_steps=a.steps, max_folds=a.folds)
    if a.task in ("B", "both"):
        run_task_b(md, n_seeds=a.seeds, epochs=max(60, a.epochs - 30), n_steps=a.steps,
                   max_folds=a.folds)
