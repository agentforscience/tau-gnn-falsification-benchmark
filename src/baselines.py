"""Classical network-spreading baselines, re-implemented in Python.

Upstream reference implementations are MATLAB (Raj Lab: NDM / eNDM / NexIS / NTM) or R
(Cornblath `tau-spread`); neither runtime is available in this workspace, so the models are
re-derived from the equations in the papers held in `papers/`:

  Persistence  x(t) = x0                                     (the "no diffusion" null; Schafer 2020)
  NDM          x(t) = exp(-beta L t) x0                       (Raj 2012; Schafer 2020 Eq. 1)
  FKPP         dc/dt = -kappa L c + alpha c (1 - c)           (Schafer 2020 Eq. 2; Fornari 2019)
  NexIS/eNDM   dx/dt = -gamma L(s) x + alpha (1 + B g) x (1-x)  (Anand 2022; Raj 2015)

`L(s)` is the Laplacian of the directionality blend `s A_antero + (1-s) A_retro`, so s = 1 is
pure anterograde, s = 0 pure retrograde, s = 0.5 effectively non-directional. The gene term
`B g` is the *classical* H2 comparator: it already lets regional expression modulate the local
aggregation rate, which is what eNDM/NexIS were built to do.

All models are fitted by maximising the SAME objective the GNN is trained on -- mean spatial
Pearson R on log10 pathology across training experiments and timepoints -- so no model is
advantaged by an objective mismatch.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from metrics import pearson, transform
from simulate import integrate, Group


# --------------------------------------------------------------------------------------
# graph operators
# --------------------------------------------------------------------------------------
def normalize_adj(A: np.ndarray, mode: str = "max") -> np.ndarray:
    """Scale an adjacency so different connectomes are on a comparable footing.

    'max' divides by the largest edge (preserves relative weights; the Nexis convention).
    """
    A = np.asarray(A, float).copy()
    np.fill_diagonal(A, 0.0)
    if mode == "max":
        m = A.max()
        return A / m if m > 0 else A
    if mode == "row":
        r = A.sum(axis=1, keepdims=True)
        return np.divide(A, np.where(r > 0, r, 1.0))
    if mode == "none":
        return A
    raise ValueError(mode)


def laplacian(A: np.ndarray) -> np.ndarray:
    """Out-degree Laplacian for the flow dx/dt = -L x.

    L = diag(rowsum A) - A^T gives
        dx_i/dt = -x_i sum_j A_ij + sum_j A_ji x_j
    i.e. mass leaves i along its outgoing edges and arrives from nodes projecting to it.
    Reduces to the usual D - A for symmetric A.
    """
    A = np.asarray(A, float)
    return np.diag(A.sum(axis=1)) - A.T


def rewire_degree_preserving(A: np.ndarray, n_swaps_per_edge: int = 8,
                             seed: int = 0) -> np.ndarray:
    """Maslov-Sneppen double-edge swap: preserves in- and out-degree sequences exactly.

    Weights travel with their edges, so the weight distribution is preserved and only the
    alignment between topology and anatomy is destroyed. Standard null network for
    connectome-spreading models (Henderson 2019; Cornblath 2021).
    """
    rng = np.random.default_rng(seed)
    B = np.asarray(A, float).copy()
    src, dst = np.nonzero(B)
    edges = list(zip(src.tolist(), dst.tolist()))
    for _ in range(n_swaps_per_edge * len(edges)):
        i, j = rng.integers(len(edges)), rng.integers(len(edges))
        (a, b), (c, d) = edges[i], edges[j]
        if len({a, b, c, d}) < 4 or B[a, d] != 0 or B[c, b] != 0:
            continue
        B[a, d], B[c, b] = B[a, b], B[c, d]
        B[a, b] = B[c, d] = 0.0
        edges[i], edges[j] = (a, d), (c, b)
    return B


# --------------------------------------------------------------------------------------
# model specification
# --------------------------------------------------------------------------------------
class SpreadModel:
    """A parameterised network-spreading model over a fixed connectome pair.

    Args:
        kind: 'persist' | 'ndm' | 'fkpp' | 'nexis'
        A_ant, A_ret: max-normalised anterograde / retrograde adjacency (N x N). For an
            undirected connectome pass the same matrix twice; the direction parameter then
            has no effect and is dropped.
        genes: (N, F) z-scored regional expression, or None. Present only for 'nexis'.
        fit_dir: whether the anterograde/retrograde blend is a free parameter.
    """

    def __init__(self, kind: str, A_ant: np.ndarray, A_ret: np.ndarray | None = None,
                 genes: np.ndarray | None = None, fit_dir: bool = False,
                 dir_fixed: float = 0.5):
        self.kind = kind
        self.A_ant = A_ant
        self.A_ret = A_ret if A_ret is not None else A_ant
        self.genes = genes if (genes is not None and genes.shape[1] > 0) else None
        self.fit_dir = fit_dir and not np.array_equal(self.A_ant, self.A_ret)
        self.dir_fixed = dir_fixed
        self._Lcache: dict[float, np.ndarray] = {}

    # ---- parameter book-keeping --------------------------------------------------------
    @property
    def n_params(self) -> int:
        n = {"persist": 0, "ndm": 1, "fkpp": 2, "nexis": 2}[self.kind]
        if self.fit_dir and self.kind != "persist":
            n += 1
        if self.kind == "nexis" and self.genes is not None:
            n += self.genes.shape[1]
        return n

    def bounds(self):
        b = []
        if self.kind in ("ndm", "fkpp", "nexis"):
            b.append((-4.0, 5.0))                    # log rate
        if self.kind in ("fkpp", "nexis"):
            b.append((-4.0, 4.0))                    # log growth
        if self.fit_dir and self.kind != "persist":
            b.append((-4.0, 4.0))                    # direction logit
        if self.kind == "nexis" and self.genes is not None:
            b += [(-3.0, 3.0)] * self.genes.shape[1]
        return b

    def x0(self):
        return np.array([{"ndm": -1.0, "fkpp": -1.0, "nexis": -1.0}.get(self.kind, 0.0)
                         if i == 0 else 0.0 for i in range(self.n_params)])

    # ---- graph ------------------------------------------------------------------------
    def _L(self, s: float) -> np.ndarray:
        key = round(float(s), 4)
        if key not in self._Lcache:
            if len(self._Lcache) > 64:
                self._Lcache.clear()
            A = self.A_ant if key == 1.0 else (self.A_ret if key == 0.0
                                               else key * self.A_ant + (1 - key) * self.A_ret)
            self._Lcache[key] = laplacian(A)
        return self._Lcache[key]

    # ---- forward ----------------------------------------------------------------------
    def predict(self, group: Group, params: np.ndarray) -> np.ndarray:
        """Return (T, N, B) predicted pathology for a batched Group of experiments."""
        p = list(np.asarray(params, float))
        X0 = group.seeds
        if self.kind == "persist":
            return np.stack([X0.copy() for _ in group.times])
        rate = np.exp(np.clip(p.pop(0), -4, 5))
        growth = np.exp(np.clip(p.pop(0), -4, 4)) if self.kind in ("fkpp", "nexis") else 0.0
        s = 1.0 / (1.0 + np.exp(-p.pop(0))) if self.fit_dir else self.dir_fixed
        L = self._L(s)
        if self.kind == "nexis" and self.genes is not None:
            b = np.asarray(p[: self.genes.shape[1]], float)
            m = np.clip(1.0 + self.genes @ b, 0.05, 20.0)[:, None]
        else:
            m = 1.0
        if self.kind == "ndm":
            return integrate(X0, group.times, lambda x: -rate * (L @ x))
        return integrate(X0, group.times,
                         lambda x: -rate * (L @ x) + growth * m * x * (1.0 - x))


# --------------------------------------------------------------------------------------
# scoring and fitting
# --------------------------------------------------------------------------------------
def score_groups(model: SpreadModel, groups: list[Group], params: np.ndarray,
                 scale: str = "raw") -> float:
    """Mean spatial Pearson R across all experiments and timepoints, on the given scale."""
    rs = []
    for gr in groups:
        P = model.predict(gr, params)                     # (T, N, B)
        for b, e in enumerate(gr.exps):
            for k in range(P.shape[0]):
                r = pearson(transform(e.path[k], scale), transform(P[k][e.idx, b], scale))
                if np.isfinite(r):
                    rs.append(r)
    return float(np.mean(rs)) if rs else -1.0


def fit(model: SpreadModel, groups: list[Group], n_restarts: int = 3,
        seed: int = 0, maxiter: int = 500, scale: str = "raw") -> dict:
    """Fit scalar parameters by Nelder-Mead on -mean training Pearson R (multi-restart)."""
    if model.n_params == 0:
        return {"params": np.zeros(0),
                "train_R": score_groups(model, groups, np.zeros(0), scale)}
    bnds = model.bounds()
    lo = np.array([b[0] for b in bnds])
    hi = np.array([b[1] for b in bnds])
    rng = np.random.default_rng(seed)

    def neg(par):
        par = np.clip(par, lo, hi)
        try:
            return -score_groups(model, groups, par, scale)
        except Exception:
            return 10.0

    starts = [model.x0()] + [lo + rng.random(len(bnds)) * (hi - lo) * 0.4 + 0.3 * (hi - lo)
                             for _ in range(n_restarts - 1)]
    best = None
    for s0 in starts:
        res = minimize(neg, np.clip(s0, lo, hi), method="Nelder-Mead",
                       options={"maxiter": maxiter, "xatol": 1e-3, "fatol": 1e-4})
        par = np.clip(res.x, lo, hi)
        sc = -neg(par)
        if best is None or sc > best["train_R"]:
            best = {"params": par, "train_R": sc}
    return best
