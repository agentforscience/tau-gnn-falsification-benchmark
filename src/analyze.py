"""Statistical analysis and figures for the tau-propagation benchmark.

Reads everything in results/ and produces:
  * results/analysis_summary.json  -- every number quoted in REPORT.md
  * results/table_*.csv            -- publication-format tables
  * figures/*.png                  -- the figures referenced by REPORT.md

The inferential design (pre-registered in planning.md):
  * Primary comparisons are PAIRED across held-out folds (Wilcoxon signed-rank, two-sided,
    alpha = 0.05). Folds are whole held-out experiments (mouse) or held-out region sets
    (human), so pairs are exchangeable under the null.
  * Effect sizes are reported as Cohen's dz and as the mean paired difference with a
    percentile bootstrap CI.
  * The gene-set tests use empirical (permutation) p-values against size-matched random
    gene sets, which is the only valid null given a 3,855-gene search space.
"""
from __future__ import annotations

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                       # noqa: E402
import pandas as pd                      # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metrics import empirical_p, paired_test   # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "figures")
os.makedirs(FIG, exist_ok=True)

# consistent colours by model family across every figure
FAMCOL = {"null": "#9aa0a6", "null-network": "#c58ad0", "diffusion": "#4c78a8",
          "diffusion+genes": "#72b7b2", "gnn": "#e45756", "gnn+genes": "#f58518",
          "gnn-ablation": "#bab0ac", "hypothesis": "#f58518", "ablation": "#4c78a8",
          "reference": "#9aa0a6", "random": "#d3d3d3"}


def _load(name):
    p = os.path.join(RES, name)
    return pd.read_csv(p) if os.path.exists(p) else None


def fold_means(df, metric="R", by=("model", "family")):
    """Mean metric per (model, fold), averaging over seeds and timepoints first.

    Averaging over seeds *within* a fold before the paired test is deliberate: seeds are not
    independent observations, folds are.
    """
    g = df.groupby(list(by) + ["fold"])[metric].mean().reset_index()
    return g


def compare_table(df, metric, reference_models, out_name, title):
    """Per-model summary plus paired tests against each reference model."""
    fm = fold_means(df, metric)
    piv = fm.pivot_table(index="fold", columns="model", values=metric)
    fam = df.groupby("model")["family"].first()
    rows = []
    for m in piv.columns:
        row = {"model": m, "family": fam.get(m, ""),
               f"mean_{metric}": piv[m].mean(), f"sd_{metric}": piv[m].std(ddof=1),
               "n_folds": int(piv[m].notna().sum())}
        for ref in reference_models:
            if ref in piv.columns and ref != m:
                t = paired_test(piv[m].values, piv[ref].values)
                row[f"vs_{ref}__diff"] = t["mean_diff"]
                row[f"vs_{ref}__p"] = t.get("wilcoxon_p", np.nan)
                row[f"vs_{ref}__dz"] = t.get("cohens_dz", np.nan)
                row[f"vs_{ref}__wins"] = t.get("n_wins", np.nan)
        rows.append(row)
    out = pd.DataFrame(rows).sort_values(f"mean_{metric}", ascending=False)
    out.to_csv(os.path.join(RES, out_name), index=False)
    print(f"\n=== {title} (metric = {metric}) ===")
    show = ["model", "family", f"mean_{metric}", f"sd_{metric}"] + \
           [c for c in out.columns if c.startswith("vs_")]
    print(out[show].round(4).to_string(index=False))
    return out, piv


# --------------------------------------------------------------------------------------
def fig_task_a(df, piv, fname="fig1_mouse_taskA.png"):
    """Per-model held-out R, with per-fold points and the persistence null marked."""
    order = piv.mean().sort_values().index.tolist()
    fam = df.groupby("model")["family"].first()
    fig, ax = plt.subplots(figsize=(9, 6.5))
    for i, m in enumerate(order):
        v = piv[m].dropna().values
        c = FAMCOL.get(fam.get(m, ""), "#666")
        ax.barh(i, v.mean(), color=c, alpha=0.85, height=0.68)
        ax.plot(v, np.full(len(v), i) + np.linspace(-0.22, 0.22, len(v)), "o",
                ms=4, color="k", alpha=0.55, mew=0)
    ref = piv["Persistence (seed only)"].mean() if "Persistence (seed only)" in piv else None
    if ref is not None:
        ax.axvline(ref, color="k", ls="--", lw=1.4,
                   label=f"persistence null (R = {ref:.3f})")
    ax.axvline(0, color="k", lw=0.8)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order, fontsize=9)
    ax.set_xlabel("Held-out spatial Pearson R (seed $\\rightarrow$ trajectory, LOEO)")
    ax.set_title("Mouse Task A: predicting the tau trajectory from the injection seed\n"
                 "leave-one-experiment-out over 8 seeded tauopathy experiments", fontsize=11)
    ax.legend(loc="lower right", fontsize=9)
    handles = [plt.Rectangle((0, 0), 1, 1, color=FAMCOL[k]) for k in
               ["null", "diffusion", "diffusion+genes", "null-network", "gnn", "gnn+genes"]]
    ax.legend(handles + [plt.Line2D([], [], color="k", ls="--")],
              ["persistence null", "network diffusion", "diffusion + genes",
               "null networks", "GNN (TauODE)", "GNN + genes", "persistence level"],
              fontsize=8, loc="lower right", framealpha=0.95)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, fname), dpi=160)
    plt.close(fig)


def fig_per_experiment(df, fname="fig2_mouse_per_experiment.png"):
    """Per held-out experiment: how each model family does, showing where signal exists."""
    keep = ["Persistence (seed only)", "FKPP / NexIS (no genes)",
            "NexIS + genes (MAPT/APOE/TREM2)", "FKPP on distance graph",
            "TauODE (no genes)", "TauODE + genes (MAPT/APOE/TREM2)"]
    sub = df[df.model.isin(keep)]
    g = sub.groupby(["held_out", "model"])["R"].mean().unstack()[
        [k for k in keep if k in sub.model.unique()]]
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(g.index))
    w = 0.8 / len(g.columns)
    for j, m in enumerate(g.columns):
        fam = sub[sub.model == m]["family"].iloc[0]
        ax.bar(x + j * w - 0.4 + w / 2, g[m].values, w, label=m,
               color=FAMCOL.get(fam, "#666"), alpha=0.9,
               edgecolor="k" if "genes" in m else "none", lw=0.6)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(g.index, rotation=25, ha="right", fontsize=9)
    ax.set_ylabel("Held-out spatial Pearson R")
    ax.set_title("Per-experiment breakdown: two of eight experiments (DS1, DS10) carry no\n"
                 "predictable signal even for an in-sample fit", fontsize=11)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, fname), dpi=160)
    plt.close(fig)


def fig_task_b(dfb, fname="fig3_mouse_taskB.png"):
    """One-step forecasting: absolute R hides the persistence trap; Delta R exposes it."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), sharey=True)
    for ax, metric, lab in zip(axes, ["R", "dR"],
                               ["absolute-scale R\n(what the literature usually reports)",
                                "$\\Delta$-tau R\n(correlation of predicted with observed change)"]):
        fm = dfb.groupby(["model", "family", "fold"])[metric].mean().reset_index()
        piv = fm.pivot_table(index="fold", columns="model", values=metric)
        order = piv.mean().sort_values().index.tolist()
        fam = dfb.groupby("model")["family"].first()
        for i, m in enumerate(order):
            v = piv[m].dropna().values
            ax.barh(i, v.mean(), color=FAMCOL.get(fam.get(m, ""), "#666"), alpha=0.85)
            ax.plot(v, np.full(len(v), i) + np.linspace(-0.2, 0.2, len(v)), "o", ms=3,
                    color="k", alpha=0.5, mew=0)
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels(order, fontsize=8)
        ax.axvline(0, color="k", lw=0.8)
        ax.set_xlabel(lab, fontsize=10)
    axes[0].set_title("Mouse Task B: one-step forecast x($t_k$) $\\rightarrow$ x($t_{k+1}$)",
                      fontsize=11, loc="left")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, fname), dpi=160)
    plt.close(fig)


def fig_genes(e5a, e5b, e5c, fname="fig4_gene_ablation.png"):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    # panel A: ablation ladder
    ax = axes[0]
    if e5a is not None:
        fm = e5a.groupby(["condition", "fold"])["test_R"].mean().reset_index()
        piv = fm.pivot(index="fold", columns="condition", values="test_R")
        order = piv.mean().sort_values().index.tolist()
        for i, c in enumerate(order):
            ax.barh(i, piv[c].mean(), color="#f58518" if "core3" in c else "#4c78a8", alpha=0.85)
            ax.plot(piv[c].values, np.full(len(piv), i) + np.linspace(-0.2, 0.2, len(piv)),
                    "o", ms=4, color="k", alpha=0.5, mew=0)
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels(order, fontsize=9)
        ax.set_xlabel("Held-out R (TauODE, LOEO)")
        ax.set_title("A. Node-feature ablation ladder", fontsize=11)
    # panel B: random gene-set null for the GNN
    ax = axes[1]
    if e5b is not None:
        m = e5b.groupby("condition")["test_R"].mean()
        rnd = m[[c for c in m.index if c.startswith("random_")]].values
        ax.hist(rnd, bins=15, color="#d3d3d3", edgecolor="k", lw=0.5,
                label=f"random gene triplets (n={len(rnd)})")
        if "core3 (Mapt/Apoe/Trem2)" in m.index:
            ax.axvline(m["core3 (Mapt/Apoe/Trem2)"], color="#f58518", lw=2.4,
                       label="Mapt / Apoe / Trem2")
        if "none" in m.index:
            ax.axvline(m["none"], color="k", ls="--", lw=1.6, label="no gene features")
        ax.set_xlabel("Mean held-out R across folds")
        ax.set_ylabel("count")
        ax.set_title("B. Size-matched random gene-set null (GNN)", fontsize=11)
        ax.legend(fontsize=8)
    # panel C: residual-explanation test
    ax = axes[2]
    if e5c is not None:
        d = e5c.set_index("condition")
        labs = list(d.index)
        y = np.arange(len(labs))
        ax.barh(y, d["null_mean"], color="#d3d3d3", label="random-gene null (mean)")
        ax.errorbar(d["null_mean"], y, xerr=d["null_sd"], fmt="none", ecolor="k", lw=1)
        ax.plot(d["observed_R2"], y, "D", color="#e45756", ms=8, label="observed")
        for i, (v, p) in enumerate(zip(d["observed_R2"], d["empirical_p"])):
            ax.text(v + 0.008, i, f"p={p:.3f}", va="center", fontsize=8)
        ax.set_yticks(y)
        ax.set_yticklabels(labs, fontsize=9)
        ax.set_xlabel("$R^2$ of held-out residual explained")
        ax.set_title("C. Residual-explanation test (1000 nulls)", fontsize=11)
        ax.legend(fontsize=8, loc="lower right")
        ax.set_xlim(0, max(0.2, d["observed_R2"].max() * 1.35))
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, fname), dpi=160)
    plt.close(fig)


def fig_human(h1, h2, fname="fig5_human.png"):
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.4))
    for ax, df, title, metric in [
            (axes[0], h1, "A. ADNI NC$\\rightarrow$AD, region-held-out CV (10 folds)", "dR"),
            (axes[1], h2, "B. Transfer: fit on ADNI, test on independent cohort", "dR")]:
        if df is None:
            continue
        if "fold" in df.columns:
            piv = df.groupby(["model", "fold"])[metric].mean().unstack().T
            means = piv.mean()
        else:
            means = df.groupby("model")[metric].mean()
            piv = None
        order = means.sort_values().index.tolist()
        fam = df.groupby("model")["family"].first()
        for i, m in enumerate(order):
            ax.barh(i, means[m], color=FAMCOL.get(fam.get(m, ""), "#666"), alpha=0.87)
            if piv is not None:
                ax.plot(piv[m].values, np.full(len(piv), i) + np.linspace(-0.2, 0.2, len(piv)),
                        "o", ms=3, color="k", alpha=0.45, mew=0)
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels(order, fontsize=9)
        ax.axvline(0, color="k", lw=0.8)
        ax.set_xlabel("$\\Delta$-tau spatial Pearson R")
        ax.set_title(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, fname), dpi=160)
    plt.close(fig)


def fig_regional(e6, fname="fig6_regional_vulnerability.png"):
    if e6 is None:
        return
    d = e6.dropna(subset=["mean_residual"])
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))
    for ax, g in zip(axes, ["Mapt", "Apoe", "Trem2"]):
        col = f"expr_{g}"
        r = np.corrcoef(d[col], d["mean_residual"])[0, 1]
        ax.scatter(d[col], d["mean_residual"], s=13, alpha=0.6, color="#4c78a8", lw=0)
        z = np.polyfit(d[col], d["mean_residual"], 1)
        xs = np.linspace(d[col].min(), d[col].max(), 20)
        ax.plot(xs, np.polyval(z, xs), color="#e45756", lw=2)
        ax.set_xlabel(f"{g} expression (z)")
        ax.set_ylabel("mean held-out residual\n(observed - predicted, z)")
        ax.set_title(f"{g}:  R = {r:+.3f}", fontsize=11)
        ax.axhline(0, color="k", lw=0.7, ls=":")
    fig.suptitle("Regional vulnerability not explained by connectivity, vs. gene expression",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, fname), dpi=160)
    plt.close(fig)


# --------------------------------------------------------------------------------------
def main():
    S: dict = {}
    a = _load("mouse_taskA_loeo.csv")
    b = _load("mouse_taskB_loeo.csv")
    e5a, e5b, e5c = _load("genes_e5a_ablation.csv"), _load("genes_e5b_random_null.csv"), \
        _load("genes_e5c_residual.csv")
    e6 = _load("genes_e6_regional.csv")
    h1, h2 = _load("human_stage_transition.csv"), _load("human_cohort_transfer.csv")

    refs = ["Persistence (seed only)", "FKPP / NexIS (no genes)", "FKPP on distance graph"]
    if a is not None:
        ta, piva = compare_table(a, "R", refs, "table_mouse_taskA.csv",
                                 "Mouse Task A: seed -> trajectory, LOEO")
        S["taskA"] = ta.to_dict("records")
        fig_task_a(a, piva)
        fig_per_experiment(a)
        # pre-specified sensitivity: drop the two experiments with no in-sample signal
        low = ["DS1", "DS10"]
        a6 = a[~a.held_out.isin(low)]
        ta6, _ = compare_table(a6, "R", refs, "table_mouse_taskA_6exp.csv",
                               "Mouse Task A, 6 informative experiments (sensitivity)")
        S["taskA_6exp"] = ta6.to_dict("records")

    if b is not None:
        refb = ["Persistence (no change)", "FKPP / NexIS (no genes)"]
        for metric in ["R", "dR"]:
            tb, _ = compare_table(b, metric, refb, f"table_mouse_taskB_{metric}.csv",
                                  "Mouse Task B: one-step forecast")
            S[f"taskB_{metric}"] = tb.to_dict("records")
        fig_task_b(b)

    if e5a is not None:
        fm = e5a.groupby(["condition", "fold"])["test_R"].mean().unstack().T
        base = fm["none"]
        S["e5a"] = []
        for c in fm.columns:
            t = paired_test(fm[c].values, base.values)
            S["e5a"].append({"condition": c, "mean_R": float(fm[c].mean()),
                             "sd_R": float(fm[c].std(ddof=1)),
                             "vs_none_diff": t["mean_diff"],
                             "vs_none_p": t.get("wilcoxon_p"),
                             "vs_none_dz": t.get("cohens_dz"),
                             "wins": t.get("n_wins")})
        pd.DataFrame(S["e5a"]).to_csv(os.path.join(RES, "table_genes_e5a.csv"), index=False)
        print("\n=== E5a: gene-feature ablation ladder (TauODE, LOEO) ===")
        print(pd.DataFrame(S["e5a"]).round(4).to_string(index=False))

    if e5b is not None:
        m = e5b.groupby("condition")["test_R"].mean()
        rnd = m[[c for c in m.index if c.startswith("random_")]].values
        obs = float(m.get("core3 (Mapt/Apoe/Trem2)", np.nan))
        S["e5b"] = {"observed_core3_R": obs, "no_gene_R": float(m.get("none", np.nan)),
                    "n_random": int(len(rnd)), "null_mean": float(np.mean(rnd)),
                    "null_sd": float(np.std(rnd)),
                    "null_p95": float(np.percentile(rnd, 95)) if len(rnd) else np.nan,
                    "empirical_p": empirical_p(obs, rnd)}
        print("\n=== E5b: random gene-set null for the GNN ===")
        print(json.dumps(S["e5b"], indent=2))

    if e5c is not None:
        S["e5c"] = e5c.to_dict("records")
    fig_genes(e5a, e5b, e5c)

    if e6 is not None:
        d = e6.dropna(subset=["mean_residual"])
        S["e6"] = {g: float(np.corrcoef(d[f"expr_{g}"], d["mean_residual"])[0, 1])
                   for g in ["Mapt", "Apoe", "Trem2"]}
        S["e6_top_underpredicted"] = d.nlargest(10, "mean_residual")["region"].tolist()
        S["e6_top_overpredicted"] = d.nsmallest(10, "mean_residual")["region"].tolist()
        fig_regional(e6)

    if h1 is not None:
        th, _ = compare_table(h1, "dR", ["Persistence (no change)", "FKPP (ACS connectome)",
                                         "FKPP on distance graph"],
                              "table_human_stage.csv", "Human: NC->AD, region-held-out CV")
        S["human_stage"] = th.to_dict("records")
    if h2 is not None:
        S["human_transfer"] = (h2.groupby(["model", "family"])[["R", "dR", "dR2"]]
                               .mean().reset_index().to_dict("records"))
        h2.groupby(["model", "family"])[["R", "dR", "dR2"]].mean().reset_index() \
            .to_csv(os.path.join(RES, "table_human_transfer.csv"), index=False)
    fig_human(h1, h2)

    with open(os.path.join(RES, "analysis_summary.json"), "w") as f:
        json.dump(S, f, indent=2, default=float)
    print(f"\nWrote results/analysis_summary.json and figures to {FIG}/")


if __name__ == "__main__":
    main()
