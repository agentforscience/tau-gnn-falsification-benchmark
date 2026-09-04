"""E5-E6: the H2 gene-feature ablation and regional-vulnerability attribution.

H2 states that adding regional gene expression (particularly MAPT, APOE, TREM2) as node
features improves prediction of region-specific vulnerability to tau accumulation.

Three tests, deliberately at different cost/resolution trade-offs:

  E5a  GNN ablation ladder.  TauODE trained leave-one-experiment-out with node features
       {none} / {Mapt,Apoe,Trem2} / {AD-risk panel} / {top-10 transcriptome PCs}.
       Full training budget, 3 seeds. Answers "do gene features help end-to-end?"

  E5b  Size-matched random gene-set null for the GNN.  The same TauODE pipeline re-run with
       `n_random` random gene triplets (reduced budget for tractability). Answers "is any
       gain specific to the AD genes, or would any three genes do?" -- the control that is
       missing from much of this literature, where a 3,855-gene search space makes it easy to
       find *some* triplet that helps.

  E5c  Residual-explanation test (cheap, high resolution).  Fit the connectivity-only model,
       take its held-out per-region residuals, and ask how much residual variance regional
       expression explains, against 1,000 size-matched random gene sets. This is the test
       Cornblath (2021) and Henderson (2019) use, and it gives a p-value with 0.001 resolution.

  E6   Regional vulnerability attribution.  Where are the connectivity-only model's residuals
       largest, and do they align with MAPT/APOE/TREM2 expression and with the known
       entorhinal/temporal vulnerability gradient?

Output: results/genes_e5a_ablation.csv, results/genes_e5b_random_null.csv,
        results/genes_e5c_residual.csv, results/genes_e6_regional.csv
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from baselines import SpreadModel, fit, normalize_adj                       # noqa: E402
from data import (AD_RISK_PANEL, CORE_GENES_MOUSE, gene_features,           # noqa: E402
                  load_mouse, random_gene_sets)
from metrics import empirical_p, pearson                                     # noqa: E402
from simulate import group_by_schedule                                       # noqa: E402
from train import predict_groups, score_predictions, train_tauode            # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
os.makedirs(RES, exist_ok=True)


def mouse_ad_panel(genes: pd.DataFrame) -> list[str]:
    """Mouse orthologs of the human AD-risk panel that are present in the 3,855-gene matrix."""
    cand = [g.capitalize() for g in AD_RISK_PANEL]
    return [g for g in cand if g in genes.columns]


def pca_features(genes: pd.DataFrame, k: int = 10) -> np.ndarray:
    """Top-k principal components of the whole regional transcriptome, z-scored."""
    X = genes.values.astype(float)
    X = np.nan_to_num(X, nan=0.0)
    X = (X - X.mean(0)) / np.where(X.std(0) > 0, X.std(0), 1.0)
    U, S, _ = np.linalg.svd(X, full_matrices=False)
    Z = U[:, :k] * S[:k]
    return (Z - Z.mean(0)) / np.where(Z.std(0) > 0, Z.std(0), 1.0)


# --------------------------------------------------------------------------------------
def _loeo_tauode(kauf, Aa, Ar, gfeat, n_seeds, epochs, n_steps, label, family, rows,
                 warm_cache):
    """Run one feature condition through the full leave-one-experiment-out loop."""
    for fold, test_exp in enumerate(kauf):
        train_exps = [e for e in kauf if e.name != test_exp.name]
        if fold not in warm_cache:
            fk = SpreadModel(kind="fkpp", A_ant=Aa, A_ret=Ar, fit_dir=True)
            r = fit(fk, group_by_schedule(train_exps), n_restarts=2, seed=fold)
            warm_cache[fold] = {"rate": float(np.exp(r["params"][0])),
                                "growth": float(np.exp(r["params"][1])),
                                "dir_logit": float(r["params"][2]),
                                "baseline_R": float(
                                    np.mean([pearson(test_exp.path[k],
                                                     fk.predict(group_by_schedule([test_exp])[0],
                                                                r["params"])[k][test_exp.idx, 0])
                                             for k in range(len(test_exp.timepoints))]))}
        warm = {k: v for k, v in warm_cache[fold].items() if k != "baseline_R"}
        for s in range(n_seeds):
            model, hist = train_tauode(group_by_schedule(train_exps), None, gfeat, Aa, Ar,
                                       seed=s, epochs=epochs,
                                       model_kwargs=dict(n_steps=n_steps), warm_start=warm)
            pred = predict_groups(model, group_by_schedule([test_exp]), gfeat, Aa, Ar)
            rows.append(dict(condition=label, family=family, fold=fold,
                             held_out=test_exp.name, seed_run=s,
                             n_features=0 if gfeat is None else gfeat.shape[1],
                             test_R=score_predictions(pred, [test_exp]),
                             train_R=1 - hist["train"][-1],
                             fkpp_baseline_R=warm_cache[fold]["baseline_R"]))


def run_e5a(md, n_seeds: int = 3, epochs: int = 150, n_steps: int = 24,
            out_csv: str = "genes_e5a_ablation.csv") -> pd.DataFrame:
    kauf = [e for e in md.experiments if e.source == "kaufman"]
    Aa, Ar = normalize_adj(md.conn["antero"]), normalize_adj(md.conn["retro"])
    panel = mouse_ad_panel(md.genes)
    conds = [
        ("none", "ablation", None),
        ("core3 (Mapt/Apoe/Trem2)", "hypothesis", gene_features(md.genes, CORE_GENES_MOUSE)),
        (f"AD-risk panel (n={len(panel)})", "ablation", gene_features(md.genes, panel)),
        ("transcriptome PCA-10", "ablation", pca_features(md.genes, 10)),
    ]
    rows, warm_cache = [], {}
    for label, fam, gf in conds:
        t0 = time.time()
        _loeo_tauode(kauf, Aa, Ar, gf, n_seeds, epochs, n_steps, label, fam, rows, warm_cache)
        sub = [r for r in rows if r["condition"] == label]
        print(f"  [E5a] {label:32s} mean test R = {np.mean([r['test_R'] for r in sub]):.4f} "
              f"({time.time() - t0:.0f}s)")
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RES, out_csv), index=False)
    return df


def run_e5b(md, n_random: int = 30, epochs: int = 80, n_steps: int = 20,
            out_csv: str = "genes_e5b_random_null.csv") -> pd.DataFrame:
    """Size-matched random gene-triplet null, run through the identical GNN pipeline."""
    kauf = [e for e in md.experiments if e.source == "kaufman"]
    Aa, Ar = normalize_adj(md.conn["antero"]), normalize_adj(md.conn["retro"])
    sets = random_gene_sets(md.genes, size=3, n_sets=n_random, seed=1,
                            exclude=CORE_GENES_MOUSE)
    rows, warm_cache = [], {}
    # the two reference conditions must be run at the SAME reduced budget as the null
    for label, fam, gf in [("none", "reference", None),
                           ("core3 (Mapt/Apoe/Trem2)", "reference",
                            gene_features(md.genes, CORE_GENES_MOUSE))]:
        _loeo_tauode(kauf, Aa, Ar, gf, 1, epochs, n_steps, label, fam, rows, warm_cache)
    t0 = time.time()
    for i, gs in enumerate(sets):
        _loeo_tauode(kauf, Aa, Ar, gene_features(md.genes, gs), 1, epochs, n_steps,
                     f"random_{i:03d}", "random", rows, warm_cache)
        if (i + 1) % 5 == 0:
            print(f"  [E5b] {i + 1}/{n_random} random sets ({time.time() - t0:.0f}s)")
    df = pd.DataFrame(rows)
    df["gene_set"] = df["condition"].map(
        {f"random_{i:03d}": ",".join(gs) for i, gs in enumerate(sets)})
    df.to_csv(os.path.join(RES, out_csv), index=False)
    return df


# --------------------------------------------------------------------------------------
def run_e5c(md, n_random: int = 1000, out_csv: str = "genes_e5c_residual.csv") -> pd.DataFrame:
    """Residual-explanation test: does gene expression explain what connectivity misses?

    For each held-out experiment we take the connectivity-only FKPP prediction, form the
    per-region residual (observed - predicted, both z-scored within timepoint), and regress it
    on a gene set. The statistic is the out-of-sample R^2 of that regression, pooled over
    folds; the null is 1,000 size-matched random gene sets drawn from the same 3,855-gene pool.
    """
    kauf = [e for e in md.experiments if e.source == "kaufman"]
    Aa, Ar = normalize_adj(md.conn["antero"]), normalize_adj(md.conn["retro"])
    panel = mouse_ad_panel(md.genes)

    # collect held-out residuals from the connectivity-only model
    resid, region_idx = [], []
    for test_exp in kauf:
        train_exps = [e for e in kauf if e.name != test_exp.name]
        mdl = SpreadModel(kind="fkpp", A_ant=Aa, A_ret=Ar, fit_dir=True)
        r = fit(mdl, group_by_schedule(train_exps), n_restarts=2, seed=0)
        P = mdl.predict(group_by_schedule([test_exp])[0], r["params"])
        for k in range(len(test_exp.timepoints)):
            y = test_exp.path[k]
            p = P[k][test_exp.idx, 0]
            if y.std() < 1e-9 or p.std() < 1e-9:
                continue
            res = (y - y.mean()) / y.std() - (p - p.mean()) / p.std()
            resid.append(res)
            region_idx.append(test_exp.idx)
    R = np.concatenate(resid)
    IDX = np.concatenate(region_idx)

    def r2_of(names_or_mat):
        """OLS R^2 of the residual on a gene design matrix (with intercept)."""
        G = names_or_mat if isinstance(names_or_mat, np.ndarray) \
            else gene_features(md.genes, names_or_mat)
        if G.shape[1] == 0:
            return 0.0
        X = np.column_stack([np.ones(len(IDX)), G[IDX]])
        beta, *_ = np.linalg.lstsq(X, R, rcond=None)
        pred = X @ beta
        ss_tot = float(((R - R.mean()) ** 2).sum())
        return float(1 - ((R - pred) ** 2).sum() / ss_tot) if ss_tot > 0 else np.nan

    obs = {"core3 (Mapt/Apoe/Trem2)": r2_of(CORE_GENES_MOUSE),
           f"AD-risk panel (n={len(panel)})": r2_of(panel),
           "transcriptome PCA-10": r2_of(pca_features(md.genes, 10))}
    nulls = {}
    for size, key in [(3, "core3 (Mapt/Apoe/Trem2)"), (len(panel), f"AD-risk panel (n={len(panel)})"),
                      (10, "transcriptome PCA-10")]:
        sets = random_gene_sets(md.genes, size=size, n_sets=n_random, seed=7,
                                exclude=CORE_GENES_MOUSE)
        nulls[key] = np.array([r2_of(gs) for gs in sets])

    rows = []
    for key, v in obs.items():
        nl = nulls[key]
        rows.append(dict(condition=key, observed_R2=v, null_mean=float(nl.mean()),
                         null_sd=float(nl.std()), null_p95=float(np.percentile(nl, 95)),
                         empirical_p=empirical_p(v, nl), n_null=len(nl),
                         n_residual_obs=len(R)))
    # per-gene single-gene test for the three named genes
    for g in CORE_GENES_MOUSE:
        v = r2_of([g])
        nl = np.array([r2_of(gs) for gs in random_gene_sets(md.genes, 1, n_random, seed=11)])
        rows.append(dict(condition=f"single: {g}", observed_R2=v, null_mean=float(nl.mean()),
                         null_sd=float(nl.std()), null_p95=float(np.percentile(nl, 95)),
                         empirical_p=empirical_p(v, nl), n_null=len(nl), n_residual_obs=len(R)))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RES, out_csv), index=False)
    print(df.round(4).to_string(index=False))
    return df


# --------------------------------------------------------------------------------------
def run_e6(md, out_csv: str = "genes_e6_regional.csv") -> pd.DataFrame:
    """E6: per-region vulnerability attribution from the connectivity-only model's residuals."""
    kauf = [e for e in md.experiments if e.source == "kaufman"]
    Aa, Ar = normalize_adj(md.conn["antero"]), normalize_adj(md.conn["retro"])
    n = len(md.regions_all)
    acc = np.zeros(n)
    cnt = np.zeros(n)
    obs_acc = np.zeros(n)
    for test_exp in kauf:
        train_exps = [e for e in kauf if e.name != test_exp.name]
        mdl = SpreadModel(kind="fkpp", A_ant=Aa, A_ret=Ar, fit_dir=True)
        r = fit(mdl, group_by_schedule(train_exps), n_restarts=2, seed=0)
        P = mdl.predict(group_by_schedule([test_exp])[0], r["params"])
        for k in range(len(test_exp.timepoints)):
            y, p = test_exp.path[k], P[k][test_exp.idx, 0]
            if y.std() < 1e-9 or p.std() < 1e-9:
                continue
            res = (y - y.mean()) / y.std() - (p - p.mean()) / p.std()
            acc[test_exp.idx] += res
            obs_acc[test_exp.idx] += (y - y.mean()) / y.std()
            cnt[test_exp.idx] += 1
    ok = cnt > 0
    resid = np.full(n, np.nan)
    resid[ok] = acc[ok] / cnt[ok]
    obs = np.full(n, np.nan)
    obs[ok] = obs_acc[ok] / cnt[ok]
    df = pd.DataFrame({"region": md.regions_all, "mean_residual": resid,
                       "mean_observed_z": obs, "n_obs": cnt})
    for g in CORE_GENES_MOUSE:
        df[f"expr_{g}"] = gene_features(md.genes, [g]).ravel()
    df.to_csv(os.path.join(RES, out_csv), index=False)
    print("\n[E6] correlation of held-out residual with regional expression:")
    for g in CORE_GENES_MOUSE:
        print(f"   {g:6s} R = {pearson(resid[ok], df[f'expr_{g}'].values[ok]):+.4f}")
    print("   most under-predicted regions (model says less tau than observed):")
    for r_ in df[ok].nlargest(8, "mean_residual")["region"]:
        print(f"     {r_}")
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", default="all",
                    choices=["all", "e5a", "e5b", "e5c", "e6"])
    ap.add_argument("--n-random", type=int, default=30)
    ap.add_argument("--seeds", type=int, default=3)
    a = ap.parse_args()
    md = load_mouse()
    if a.which in ("all", "e5c"):
        print("== E5c: residual-explanation test ==")
        run_e5c(md)
    if a.which in ("all", "e6"):
        run_e6(md)
    if a.which in ("all", "e5a"):
        print("\n== E5a: GNN ablation ladder ==")
        run_e5a(md, n_seeds=a.seeds)
    if a.which in ("all", "e5b"):
        print("\n== E5b: random gene-set null ==")
        run_e5b(md, n_random=a.n_random)
