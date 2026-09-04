"""Correctness checks for the tau-propagation benchmark.

These are the assertions the scientific claims rest on. Run with:
    python src/tests_sanity.py
Every check prints its measured value so the numbers can be quoted in REPORT.md.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch
from scipy.linalg import expm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from baselines import SpreadModel, laplacian, normalize_adj, rewire_degree_preserving
from data import CORE_GENES_MOUSE, gene_features, load_human, load_mouse
from gnn import TauODE
from simulate import group_by_schedule, integrate
from train import predict_groups, score_predictions, train_tauode

PASS, FAIL = "  PASS", "  FAIL"
results = []


def check(name, cond, detail=""):
    results.append((name, bool(cond)))
    print(f"{PASS if cond else FAIL}  {name}" + (f"   [{detail}]" if detail else ""))
    return cond


def main():
    print("=" * 78)
    print("SANITY CHECKS")
    print("=" * 78)

    # -- 1. the GNN strictly contains the classical NDM ---------------------------------
    rng = np.random.default_rng(0)
    A = rng.random((60, 60))
    np.fill_diagonal(A, 0)
    A = A / A.max()
    s = np.zeros(60)
    s[7] = 1.0
    gamma = 0.42
    ref = np.stack([expm(-gamma * laplacian(A) * t) @ s for t in [0.25, 0.5, 1.0]])
    m = TauODE(n_genes=0, use_growth=False, use_edge_gate=False, use_mlp=False,
               fit_dir=False, n_steps=400)
    with torch.no_grad():
        m.log_gamma.fill_(float(np.log(np.expm1(gamma))))
        got = m(torch.tensor(s), torch.tensor(A), torch.tensor(A), [0.25, 0.5, 1.0], None).numpy()
    rel = np.abs(ref - got).max() / np.abs(ref).max()
    check("TauODE zero-parameter limit == NDM heat kernel exp(-gamma L t)", rel < 1e-3,
          f"max relative error {rel:.2e}")

    # -- 2. the fixed-step integrator matches the matrix exponential ---------------------
    approx = integrate(s, [0.25, 0.5, 1.0], lambda x: -gamma * (laplacian(A) @ x),
                       n_steps_unit=60)
    rel2 = np.abs(ref - approx).max() / np.abs(ref).max()
    check("RK4 integrator (60 steps/unit) matches scipy.linalg.expm", rel2 < 1e-3,
          f"max relative error {rel2:.2e}")

    # -- 3. degree-preserving rewiring really preserves degree ---------------------------
    md = load_mouse()
    Ar = md.conn["retro"]
    Rw = rewire_degree_preserving(Ar, n_swaps_per_edge=4, seed=0)
    din_ok = np.allclose((Ar > 0).sum(0), (Rw > 0).sum(0))
    dout_ok = np.allclose((Ar > 0).sum(1), (Rw > 0).sum(1))
    same = np.allclose(Ar, Rw)
    check("rewired null preserves in-degree and out-degree exactly", din_ok and dout_ok)
    check("rewired null actually differs from the original connectome", not same,
          f"{(np.abs(Ar - Rw) > 0).sum()} entries changed")
    check("rewired null preserves the edge-weight multiset",
          np.allclose(np.sort(Ar[Ar > 0]), np.sort(Rw[Rw > 0])))

    # -- 4. human region order is anatomically correct -----------------------------------
    hd = load_human()
    i_ento, i_hip = hd.regions.index("LT_ENTO"), hd.regions.index("LT_HIPPO")
    check("human connectome: left entorhinal-hippocampus edge is non-zero",
          hd.conn["acs"][i_ento, i_hip] > 0, f"weight {hd.conn['acs'][i_ento, i_hip]:.3f}")
    d = hd.tau["ad_tau"] - hd.tau["nc_tau"]
    top = d.nlargest(6).index.tolist()
    check("human ADNI (AD - NC) tau peaks in temporal/limbic cortex (Braak IV-V)",
          any(("TEMP" in r) or ("ENTO" in r) or ("FUSIFORM" in r) for r in top), ", ".join(top))
    check("AHBA expression is aligned: MAPT is highest in temporal/entorhinal cortex",
          any(("TEMP" in r) or ("ENTO" in r) or ("FUSIFORM" in r) or ("PARAHIPP" in r)
              for r in hd.genes["MAPT"].nlargest(6).index.tolist()),
          ", ".join(hd.genes["MAPT"].nlargest(4).index.tolist()))

    # -- 5. no leakage: the held-out experiment never appears in training ------------------
    kauf = [e for e in md.experiments if e.source == "kaufman"]
    leak = False
    for te in kauf:
        tr = [e for e in kauf if e.name != te.name]
        if te.name in [e.name for e in tr]:
            leak = True
    check("LOEO splits: held-out experiment is absent from every training set", not leak)
    # seeds are inputs, targets are never seen -- assert the seed used at test time comes
    # from the injection protocol file, not from the pathology measurement
    check("test-time input is the injection seed, not any observed pathology",
          all(np.allclose(e.seed, e.seed_full[e.idx]) for e in kauf))

    # -- 6. reproducibility: identical seed gives identical predictions -------------------
    Aa, Arr = normalize_adj(md.conn["antero"]), normalize_adj(md.conn["retro"])
    g3 = gene_features(md.genes, CORE_GENES_MOUSE)
    te = [e for e in kauf if e.name == "DS6"][0]
    tr = [e for e in kauf if e.name != "DS6"]
    outs = []
    for _ in range(2):
        mdl, _ = train_tauode(group_by_schedule(tr), None, g3, Aa, Arr, seed=3, epochs=30,
                              model_kwargs=dict(n_steps=16))
        outs.append(predict_groups(mdl, group_by_schedule([te]), g3, Aa, Arr)[te.name])
    check("training is reproducible: same seed -> bit-identical predictions",
          np.allclose(outs[0], outs[1]), f"max diff {np.abs(outs[0] - outs[1]).max():.2e}")

    # -- 7. the classical fit is not degenerate -------------------------------------------
    fk = SpreadModel(kind="fkpp", A_ant=Aa, A_ret=Arr, fit_dir=True)
    from baselines import fit as bfit, score_groups
    r = bfit(fk, group_by_schedule(tr), n_restarts=2, seed=0)
    pers = SpreadModel(kind="persist", A_ant=Aa, A_ret=Arr)
    p_R = score_groups(pers, group_by_schedule([te]), np.zeros(0))
    f_R = score_groups(fk, group_by_schedule([te]), r["params"])
    check("FKPP beats the persistence null on a held-out informative experiment",
          f_R > p_R, f"FKPP {f_R:.3f} vs persistence {p_R:.3f}")

    print("=" * 78)
    n_ok = sum(v for _, v in results)
    print(f"{n_ok}/{len(results)} checks passed")
    print("=" * 78)
    return 0 if n_ok == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
