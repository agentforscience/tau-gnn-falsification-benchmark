"""E4: the human DK-86 arm.

IMPORTANT SCOPE CAVEAT (stated here and in REPORT.md): no open longitudinal human tau PET
dataset exists (verified exhaustively in the resource_finder phase -- ADNI/OASIS are
access-gated, OpenNeuro's only tau study is 33 young healthy adults). The human trajectory
available here is **stage-stratified** (NC -> EMCI -> LMCI -> AD group means), not
within-subject longitudinal. This is the same design used by Raj et al. (2015) and the
Franchi-Raj AND model, so it is literature-standard, but H1 as literally worded
("longitudinal tau PET") is fully testable only on the mouse arm.

Three evaluations, in increasing strictness:

  H-1  Stage transition, region-held-out CV.  Predict AD-stage tau from NC-stage tau by
       propagation on the connectome; 10-fold cross-validation over regions, so the model's
       parameters never see the held-out regions' targets.
  H-2  Cross-cohort transfer.  Fit on ADNI, predict the independent Lyoo/Cho replication
       cohort (NL -> AD) with no refitting. The strictest available generalisation test.
  H-3  Null-network ladder, identical to the mouse arm.

Metrics emphasise **Delta tau** (AD minus NC), because absolute tau maps autocorrelate so
strongly that absolute R is near-uninformative (Schafer et al. 2020).

Output: results/human_stage_transition.csv, results/human_cohort_transfer.csv
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from baselines import laplacian, normalize_adj, rewire_degree_preserving   # noqa: E402
from data import AD_RISK_PANEL, CORE_GENES_HUMAN, gene_features, load_human  # noqa: E402
from gnn import TauODE, corr_loss, set_seed                                 # noqa: E402
from metrics import pearson, r2_score, spearman                             # noqa: E402
from simulate import integrate                                             # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
os.makedirs(RES, exist_ok=True)

STAGES = ["nc_tau", "emci_tau", "lmci_tau", "ad_tau"]
STAGE_T = {"nc_tau": 0.0, "emci_tau": 1.0, "lmci_tau": 2.0, "ad_tau": 3.0}

# The ADNI group tau values shipped with the Raj-lab data are reference-normalised SUVRs with
# the reference subtracted (range -0.05 to 0.71, mean 0.11). The Lyoo/Cho replication cohort
# ships raw T807 SUVRs (range 0.97 to 1.98). Subtracting 1 puts the two cohorts on the same
# scale; after this correction their group means agree (NL 0.16 vs ADNI NC 0.11;
# AD 0.45 vs ADNI AD 0.31) and their Delta-tau maps correlate at R = 0.93.
REPL_SUVR_OFFSET = 1.0


def scaled(x: np.ndarray) -> tuple[np.ndarray, float]:
    """Normalise a baseline tau map to unit maximum for the logistic dynamics.

    The scale comes only from the *input* map, which is observed at prediction time, so no
    information leaks from the target. Predictions are multiplied back by the same constant
    before scoring.
    """
    x = np.asarray(x, float)
    m = float(np.max(np.abs(x)))
    return (x / m, m) if m > 0 else (x, 1.0)


# --------------------------------------------------------------------------------------
def fit_scalar(objective, grid) -> tuple[float, float]:
    """Exhaustive 1-D grid search. With one free parameter this is more reliable than
    Nelder-Mead and cheap enough at 86 nodes."""
    best_v, best_s = None, -np.inf
    for v in grid:
        s = objective(v)
        if np.isfinite(s) and s > best_s:
            best_s, best_v = s, v
    return best_v, best_s


def ndm_map(x0, L, t, beta):
    """Linear network diffusion applied to a unit-scaled baseline map, rescaled back."""
    xs, sc = scaled(x0)
    return integrate(xs, [t], lambda x: -beta * (L @ x))[0] * sc


def fkpp_map(x0, L, t, beta, alpha, gmod=None):
    """FKPP (optionally gene-modulated) applied to a unit-scaled baseline map, rescaled back."""
    m = 1.0 if gmod is None else np.clip(gmod, 0.05, 20.0)
    xs, sc = scaled(x0)
    return integrate(xs, [t], lambda x: -beta * (L @ x) + alpha * m * x * (1.0 - x))[0] * sc


# --------------------------------------------------------------------------------------
def human_models(conn, genes_z, rewire_seed: int = 0):
    """Return {name: (family, Laplacian, uses_genes)} for the classical model ladder."""
    A = normalize_adj(conn["acs"])
    Af = normalize_adj(conn["fibercount"])
    Ad = normalize_adj(conn["distance"])
    Ar = normalize_adj(rewire_degree_preserving(conn["acs"], seed=rewire_seed))
    return {
        "NDM (ACS connectome)": ("diffusion", laplacian(A), False),
        "NDM (HCP fibercount)": ("diffusion", laplacian(Af), False),
        "FKPP (ACS connectome)": ("diffusion", laplacian(A), False),
        "FKPP + genes (MAPT/APOE/TREM2)": ("diffusion+genes", laplacian(A), True),
        "NDM on distance graph": ("null-network", laplacian(Ad), False),
        "FKPP on distance graph": ("null-network", laplacian(Ad), False),
        "NDM on rewired connectome": ("null-network", laplacian(Ar), False),
        "FKPP on rewired connectome": ("null-network", laplacian(Ar), False),
    }


def _metrics(y_true, y_pred, y_base):
    dt = np.asarray(y_true) - y_base
    dp = np.asarray(y_pred) - y_base
    return dict(R=pearson(y_true, y_pred), rho=spearman(y_true, y_pred),
                dR=pearson(dt, dp), drho=spearman(dt, dp), dR2=r2_score(dt, dp),
                R_persist=pearson(y_true, y_base),
                dR2_persist_zero=r2_score(dt, np.zeros_like(dt)))


# --------------------------------------------------------------------------------------
def train_human_tauode(x0, target, A_ant, A_ret, genes_t, t, fit_mask, seed=0,
                       epochs=400, lr=0.03, warm=None, model_kwargs=None):
    """Train TauODE on the human graph, scoring only on `fit_mask` regions (region-held-out CV).

    The state equation is identical to the mouse arm; only the graph and the time axis change.
    """
    set_seed(seed)
    ng = 0 if genes_t is None else genes_t.shape[1]
    model = TauODE(n_genes=ng, **(model_kwargs or dict(n_steps=30)))
    if warm:
        with torch.no_grad():
            from train import inv_softplus
            model.log_gamma.fill_(inv_softplus(warm.get("rate", 1.0)))
            model.log_alpha.fill_(inv_softplus(warm.get("growth", 1.0)))
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    m = torch.tensor(fit_mask)
    for _ in range(epochs):
        opt.zero_grad()
        P = model(x0, A_ant, A_ret, [t], genes_t)[0]
        loss = corr_loss(P[m], target[m], "raw")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        sched.step()
    with torch.no_grad():
        return model, model(x0, A_ant, A_ret, [t], genes_t)[0].numpy()


# --------------------------------------------------------------------------------------
def run_stage_transition(hd, n_folds: int = 10, n_seeds: int = 5, out_csv: str =
                         "human_stage_transition.csv", verbose: bool = True) -> pd.DataFrame:
    """H-1: NC -> AD transition with region-held-out cross-validation."""
    rng = np.random.default_rng(0)
    regions = hd.regions
    x_nc = hd.tau["nc_tau"].values.astype(float)
    x_ad = hd.tau["ad_tau"].values.astype(float)
    genes3 = gene_features(hd.genes, CORE_GENES_HUMAN)
    models = human_models(hd.conn, genes3)

    folds = np.array_split(rng.permutation(len(regions)), n_folds)
    rows = []
    beta_grid = np.exp(np.linspace(-3, 3, 25))
    alpha_grid = np.exp(np.linspace(-3, 3, 13))

    A_acs = normalize_adj(hd.conn["acs"])
    A_t = torch.tensor(A_acs, dtype=torch.float32)
    x_nc_s, sc_nc = scaled(x_nc)
    x0_t = torch.tensor(x_nc_s, dtype=torch.float32)
    y_t = torch.tensor(x_ad / sc_nc, dtype=torch.float32)
    g3_t = torch.tensor(genes3, dtype=torch.float32)

    for f, test_idx in enumerate(folds):
        tr = np.ones(len(regions), bool)
        tr[test_idx] = False
        # persistence null
        rows.append(dict(fold=f, model="Persistence (no change)", family="null", seed_run=0,
                         **_metrics(x_ad[test_idx], x_nc[test_idx], x_nc[test_idx])))
        for name, (fam, L, use_g) in models.items():
            gm = (1.0 + genes3 @ np.array([0.5, 0.5, 0.5])) if use_g else None
            if name.startswith("NDM"):
                def obj(b, L=L):
                    return pearson(x_ad[tr], ndm_map(x_nc, L, 1.0, b)[tr])
                b, _ = fit_scalar(obj, beta_grid)
                pred = ndm_map(x_nc, L, 1.0, b)
                npar = 1
            else:
                best = (-np.inf, None)
                for b in beta_grid[::2]:
                    for al in alpha_grid:
                        if use_g:
                            best_g = (-np.inf, None)
                            for w in np.linspace(-1.0, 1.0, 9):
                                gmod = 1.0 + genes3 @ np.full(genes3.shape[1], w)
                                p = fkpp_map(x_nc, L, 1.0, b, al, gmod)
                                s = pearson(x_ad[tr], p[tr])
                                if np.isfinite(s) and s > best_g[0]:
                                    best_g = (s, (b, al, w))
                            s, par = best_g
                        else:
                            p = fkpp_map(x_nc, L, 1.0, b, al)
                            s, par = pearson(x_ad[tr], p[tr]), (b, al, None)
                        if np.isfinite(s) and s > best[0]:
                            best = (s, par)
                b, al, w = best[1]
                gmod = (1.0 + genes3 @ np.full(genes3.shape[1], w)) if use_g else None
                pred = fkpp_map(x_nc, L, 1.0, b, al, gmod)
                npar = 3 if use_g else 2
            rows.append(dict(fold=f, model=name, family=fam, seed_run=0, n_params=npar,
                             **_metrics(x_ad[test_idx], pred[test_idx], x_nc[test_idx])))

        for name, fam, gf in [("TauODE (no genes)", "gnn", None),
                              ("TauODE + genes (MAPT/APOE/TREM2)", "gnn+genes", g3_t)]:
            for s in range(n_seeds):
                _, pred = train_human_tauode(x0_t, y_t, A_t, A_t, gf, 1.0, tr, seed=s,
                                             warm={"rate": 1.0, "growth": 1.0})
                pred = pred * sc_nc
                rows.append(dict(fold=f, model=name, family=fam, seed_run=s,
                                 **_metrics(x_ad[test_idx], pred[test_idx], x_nc[test_idx])))
        if verbose:
            print(f"  [human H-1] fold {f + 1}/{n_folds} done")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RES, out_csv), index=False)
    return df


# --------------------------------------------------------------------------------------
def run_cohort_transfer(hd, n_seeds: int = 5, out_csv: str = "human_cohort_transfer.csv",
                        verbose: bool = True) -> pd.DataFrame:
    """H-2: fit every model on ADNI (NC -> AD), then apply unchanged to the Lyoo/Cho cohort.

    The replication cohort uses a different tracer preparation and a different scanner site,
    so this is a genuine out-of-cohort test rather than a resampling of the same data.
    """
    x_nc = hd.tau["nc_tau"].values.astype(float)
    x_ad = hd.tau["ad_tau"].values.astype(float)
    r_nl = hd.repl.loc["NL"].values.astype(float) - REPL_SUVR_OFFSET
    r_ad = hd.repl.loc["AD"].values.astype(float) - REPL_SUVR_OFFSET
    genes3 = gene_features(hd.genes, CORE_GENES_HUMAN)
    models = human_models(hd.conn, genes3)
    beta_grid = np.exp(np.linspace(-3, 3, 25))
    alpha_grid = np.exp(np.linspace(-3, 3, 13))
    all_idx = np.ones(len(hd.regions), bool)
    rows = [dict(model="Persistence (no change)", family="null", seed_run=0,
                 **_metrics(r_ad, r_nl, r_nl))]

    for name, (fam, L, use_g) in models.items():
        if name.startswith("NDM"):
            b, _ = fit_scalar(lambda z, L=L: pearson(x_ad, ndm_map(x_nc, L, 1.0, z)), beta_grid)
            pred = ndm_map(r_nl, L, 1.0, b)
        else:
            best = (-np.inf, None)
            for b in beta_grid[::2]:
                for al in alpha_grid:
                    ws = np.linspace(-1.0, 1.0, 9) if use_g else [None]
                    for w in ws:
                        gmod = (1.0 + genes3 @ np.full(genes3.shape[1], w)) if use_g else None
                        s = pearson(x_ad, fkpp_map(x_nc, L, 1.0, b, al, gmod))
                        if np.isfinite(s) and s > best[0]:
                            best = (s, (b, al, w))
            b, al, w = best[1]
            gmod = (1.0 + genes3 @ np.full(genes3.shape[1], w)) if use_g else None
            pred = fkpp_map(r_nl, L, 1.0, b, al, gmod)
        rows.append(dict(model=name, family=fam, seed_run=0, **_metrics(r_ad, pred, r_nl)))

    A_acs = normalize_adj(hd.conn["acs"])
    A_t = torch.tensor(A_acs, dtype=torch.float32)
    g3_t = torch.tensor(genes3, dtype=torch.float32)
    x_nc_s, sc_nc = scaled(x_nc)
    r_nl_s, sc_r = scaled(r_nl)
    for name, fam, gf in [("TauODE (no genes)", "gnn", None),
                          ("TauODE + genes (MAPT/APOE/TREM2)", "gnn+genes", g3_t)]:
        for s in range(n_seeds):
            model, _ = train_human_tauode(torch.tensor(x_nc_s, dtype=torch.float32),
                                          torch.tensor(x_ad / sc_nc, dtype=torch.float32),
                                          A_t, A_t, gf, 1.0, all_idx, seed=s,
                                          warm={"rate": 1.0, "growth": 1.0})
            with torch.no_grad():   # applied unchanged to the held-out cohort
                pred = model(torch.tensor(r_nl_s, dtype=torch.float32), A_t, A_t,
                             [1.0], gf)[0].numpy() * sc_r
            rows.append(dict(model=name, family=fam, seed_run=s, **_metrics(r_ad, pred, r_nl)))

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RES, out_csv), index=False)
    if verbose:
        print(df.groupby("model")[["R", "dR", "dR2"]].mean().round(3).to_string())
    return df


if __name__ == "__main__":
    hd = load_human()
    print("== H-1: stage transition, region-held-out CV ==")
    d1 = run_stage_transition(hd)
    print(d1.groupby(["family", "model"])[["R", "dR", "dR2"]].mean().round(3).to_string())
    print("\n== H-2: ADNI -> independent replication cohort ==")
    run_cohort_transfer(hd)
