"""Data loading for the tau-propagation benchmark.

Two experimental arms:
  * MOUSE  -- 14 seeded-tauopathy experiments, truly longitudinal, known injection site.
              Primary test of H1 (`load_mouse`).
  * HUMAN  -- ADNI DK-86 stage-stratified tau + an independent replication cohort.
              Secondary / relevance arm (`load_human`).

All matrices are returned in a single verified canonical region order. For the human arm this
order comes from `region_order_canonical_dk86.csv`; using the raw upstream connectome files
instead silently scrambles the anatomy (see datasets/README.md).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DMOUSE = os.path.join(ROOT, "datasets", "mouse_tau_longitudinal")
DHUMAN = os.path.join(ROOT, "datasets", "human_tau_adni")
DGENE = os.path.join(ROOT, "datasets", "ahba_gene_expression")

# Late-onset AD risk loci (GWAS consensus panel, human symbols) used as the "AD-risk panel"
# ablation condition. Mouse orthologs are the same symbols in title case.
AD_RISK_PANEL = [
    "APP", "PSEN1", "PSEN2", "MAPT", "APOE", "TREM2", "BIN1", "CLU", "CR1", "PICALM",
    "CD33", "MS4A6A", "ABCA7", "SORL1", "PLCG2", "ABI3", "CD2AP", "EPHA1", "INPP5D",
]
CORE_GENES_HUMAN = ["MAPT", "APOE", "TREM2"]
CORE_GENES_MOUSE = ["Mapt", "Apoe", "Trem2"]


# --------------------------------------------------------------------------------------
# containers
# --------------------------------------------------------------------------------------
@dataclass
class Experiment:
    """One seeded-tauopathy experiment (one mouse strain / injection site / study)."""
    name: str
    source: str                      # 'kaufman' | 'endm'
    regions: list[str]               # measured region names, canonical order subset
    idx: np.ndarray                  # indices into the full 426-region space
    timepoints: list[float]          # months post injection
    path: np.ndarray                 # (T, n_measured) pathology, NaN-free
    seed: np.ndarray                 # (n_measured,) injection seed vector
    seed_full: np.ndarray            # (426,) seed on the full graph -- models are simulated on
                                     # the whole connectome and compared only where measured,
                                     # so pathology can route through unmeasured regions


@dataclass
class MouseData:
    regions_all: list[str]                       # 426 region names
    conn: dict[str, np.ndarray] = field(default_factory=dict)   # 426x426 matrices
    genes: pd.DataFrame = None                   # 426 x 3855
    experiments: list[Experiment] = field(default_factory=list)


@dataclass
class HumanData:
    regions: list[str]
    conn: dict[str, np.ndarray]
    tau: pd.DataFrame          # adni stage tau, index=region
    genes: pd.DataFrame        # 86 x 15633 AHBA
    repl: pd.DataFrame         # replication cohort, index=group
    gene_ok: np.ndarray        # boolean mask of regions with expression data


# --------------------------------------------------------------------------------------
# mouse arm
# --------------------------------------------------------------------------------------
def load_mouse(verbose: bool = False) -> MouseData:
    """Load the 426-region mouse longitudinal tauopathy benchmark.

    Returns a MouseData with 14 Experiment objects (8 Kaufman + 6 eNDM). Each experiment
    keeps only the regions it actually measured (missingness is structural per-experiment,
    not random), so `Experiment.idx` maps back to the shared 426-region graph.
    """
    regions_all = pd.read_csv(os.path.join(DMOUSE, "region_names_426.csv"))["region"].tolist()
    pos = {r: i for i, r in enumerate(regions_all)}

    conn = {}
    for key, fn in [("retro", "connectome_ret_426.csv"), ("antero", "connectome_ant_426.csv"),
                    ("nd", "connectome_nd_426.csv"), ("spatial", "connectome_spat_426.csv")]:
        A = pd.read_csv(os.path.join(DMOUSE, fn), index_col=0)
        assert A.index.tolist() == regions_all, f"{fn}: region order mismatch"
        conn[key] = A.values.astype(float)

    genes = pd.read_csv(os.path.join(DMOUSE, "gene_expression_426x3855.csv"), index_col=0)
    assert genes.index.tolist() == regions_all, "gene matrix region order mismatch"

    experiments = []
    for src, pfile, sfile in [("kaufman", "kaufman_pathology_long.csv", "kaufman_seeds.csv"),
                              ("endm", "endm_pathology_long.csv", "endm_seeds.csv")]:
        P = pd.read_csv(os.path.join(DMOUSE, pfile))
        S = pd.read_csv(os.path.join(DMOUSE, sfile))
        for ds, sub in P.groupby("dataset"):
            wide = sub.pivot(index="timepoint", columns="region", values="pathology")
            # keep only regions measured at *every* timepoint of this experiment
            keep = wide.columns[wide.notna().all(axis=0)].tolist()
            keep = [r for r in regions_all if r in set(keep)]      # enforce canonical order
            wide = wide[keep]
            tps = sorted(wide.index.tolist())
            wide = wide.loc[tps]
            sfull = (S[S.dataset == ds].set_index("region")["seed"]
                     .reindex(regions_all).fillna(0.0).values.astype(float))
            idx = np.array([pos[r] for r in keep])
            experiments.append(Experiment(
                name=ds, source=src, regions=keep, idx=idx,
                timepoints=[float(t) for t in tps],
                path=wide.values.astype(float), seed=sfull[idx],
                seed_full=sfull,
            ))
    experiments.sort(key=lambda e: (e.source, e.name))

    if verbose:
        for e in experiments:
            print(f"  {e.source:8s} {e.name:14s} T={e.timepoints} n={len(e.regions):3d} "
                  f"seed_nnz={int((e.seed > 0).sum())} max_path={e.path.max():.4f}")
    return MouseData(regions_all=regions_all, conn=conn, genes=genes, experiments=experiments)


# --------------------------------------------------------------------------------------
# human arm
# --------------------------------------------------------------------------------------
def load_human() -> HumanData:
    """Load the human DK-86 arm in verified canonical region order."""
    regions = pd.read_csv(os.path.join(DHUMAN, "region_order_canonical_dk86.csv"))["region"].tolist()

    conn = {}
    for key, fn in [("acs", "connectome_ACS_dk86_canonical.csv"),
                    ("fibercount", "connectome_HCP_fibercount_dk86_canonical.csv"),
                    ("fiberlength", "connectome_HCP_fiberlength_dk86_canonical.csv")]:
        A = pd.read_csv(os.path.join(DHUMAN, fn), index_col=0)
        assert A.index.tolist() == regions, f"{fn}: region order mismatch"
        conn[key] = A.values.astype(float)
    # distance null: invert fibre length into a "closeness" adjacency (Schafer's distance model)
    L = conn["fiberlength"].copy()
    with np.errstate(divide="ignore"):
        D = np.where(L > 0, 1.0 / np.maximum(L, 1e-9), 0.0)
    np.fill_diagonal(D, 0.0)
    conn["distance"] = D

    tau = pd.read_csv(os.path.join(DHUMAN,
                                   "adni_group_regional_tau_amyloid_atrophy_dk86.csv"))
    assert tau["region"].tolist() == regions
    tau = tau.set_index("region")

    genes = pd.read_csv(os.path.join(DGENE, "ahba_expression_dk86_canonical.csv"), index_col=0)
    genes.index = regions          # already aligned by align_gene_expression.py
    gene_ok = ~genes.isna().all(axis=1).values

    repl = pd.read_csv(os.path.join(DHUMAN, "replication_cohort_T807_tau_by_group.csv"))
    repl = repl.rename(columns={repl.columns[0]: "group"}).set_index("group")
    repl = repl[regions]           # enforce canonical order

    return HumanData(regions=regions, conn=conn, tau=tau, genes=genes,
                     repl=repl, gene_ok=gene_ok)


# --------------------------------------------------------------------------------------
# feature construction
# --------------------------------------------------------------------------------------
def gene_features(genes: pd.DataFrame, names: list[str], idx: np.ndarray | None = None,
                  ) -> np.ndarray:
    """Z-scored regional expression for `names`, restricted to rows `idx`.

    Missing regions (AHBA has 6 with no tissue) are imputed to the column mean *after*
    z-scoring, i.e. to 0 -- a neutral value that cannot leak information.
    """
    if not names:
        n = len(genes) if idx is None else len(idx)
        return np.zeros((n, 0))
    have = [g for g in names if g in genes.columns]
    if not have:
        n = len(genes) if idx is None else len(idx)
        return np.zeros((n, 0))
    X = genes[have].values.astype(float)
    if idx is not None:
        X = X[idx]
    mu = np.nanmean(X, axis=0)
    sd = np.nanstd(X, axis=0)
    sd = np.where(sd > 0, sd, 1.0)
    Z = (X - mu) / sd
    return np.nan_to_num(Z, nan=0.0)


def random_gene_sets(genes: pd.DataFrame, size: int, n_sets: int, seed: int = 0,
                     exclude: list[str] | None = None) -> list[list[str]]:
    """Size-matched random gene-set null. Only genes with non-zero regional variance are drawn."""
    rng = np.random.default_rng(seed)
    V = genes.values.astype(float)
    var_ok = np.nanstd(V, axis=0) > 0
    pool = [g for g, ok in zip(genes.columns, var_ok) if ok]
    if exclude:
        ex = set(exclude)
        pool = [g for g in pool if g not in ex]
    return [list(rng.choice(pool, size=size, replace=False)) for _ in range(n_sets)]
