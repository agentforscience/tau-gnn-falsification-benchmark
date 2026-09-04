"""Batched forward simulators shared by the classical baselines and the GNN.

Two implementation choices matter for tractability and for fairness:

1. **Normalised time.** Physical timepoints (months post injection) are divided by a global
   constant TSCALE. Because every model's rate parameter is free, this is a pure
   reparameterisation, but it lets a fixed-step integrator resolve the whole trajectory in
   ~60 steps instead of ~800. Verified against `scipy.linalg.expm` (see tests in exp_mouse.py).

2. **Batching over experiments.** Experiments that share a timepoint schedule are integrated
   simultaneously as an (N, B) state matrix on the *same* 426-region graph. This is what makes
   leave-one-experiment-out x 5 seeds x the full model ladder finish in minutes rather than
   hours.

Models are always simulated on the FULL connectome and scored only on the regions a given
experiment actually measured, so pathology may route through unmeasured territory
(the convention used by Nexis and by Cornblath's `tau-spread`).
"""
from __future__ import annotations

import numpy as np

TSCALE = 12.0            # months; the Kaufman schedule ends at 12 MPI
N_STEPS_UNIT = 60        # integrator steps per unit of normalised time


def norm_times(times) -> list[float]:
    return [float(t) / TSCALE for t in times]


def integrate(x0: np.ndarray, times: list[float], dxdt, n_steps_unit: int = N_STEPS_UNIT,
              clip: float = 5.0) -> np.ndarray:
    """RK4 integration of dx/dt = dxdt(x) sampled at `times` (already normalised).

    Args:
        x0: (N,) or (N, B) initial state.
    Returns: (T, N) or (T, N, B).
    """
    times = [float(t) for t in times]
    tmax = max(times)
    n = max(4, int(np.ceil(tmax * n_steps_unit)))
    dt = tmax / n
    x = np.asarray(x0, float).copy()
    res: list = [None] * len(times)
    order = sorted(range(len(times)), key=lambda k: times[k])
    ti, t = 0, 0.0
    for step in range(n + 1):
        while ti < len(times) and t >= times[order[ti]] - 1e-9:
            res[order[ti]] = x.copy()
            ti += 1
        if step == n:
            break
        k1 = dxdt(x)
        k2 = dxdt(np.clip(x + 0.5 * dt * k1, 0, clip))
        k3 = dxdt(np.clip(x + 0.5 * dt * k2, 0, clip))
        k4 = dxdt(np.clip(x + dt * k3, 0, clip))
        x = np.clip(x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4), 0, clip)
        t += dt
    for k in range(len(times)):
        if res[k] is None:
            res[k] = x.copy()
    return np.stack(res)


def unit_seed(s: np.ndarray) -> np.ndarray:
    """Normalise a seed to unit maximum (a property of the injection, not of the outcome)."""
    s = np.asarray(s, float)
    m = s.max()
    return s / m if m > 0 else s


class Group:
    """A set of experiments sharing one timepoint schedule, batched into one simulation."""

    def __init__(self, exps):
        assert exps, "empty group"
        self.exps = list(exps)
        self.times_raw = list(exps[0].timepoints)
        self.times = norm_times(self.times_raw)
        self.seeds = np.stack([unit_seed(e.seed_full) for e in exps], axis=1)   # (N, B)

    @property
    def B(self):
        return len(self.exps)


def group_by_schedule(exps) -> list[Group]:
    """Group experiments by identical timepoint tuples."""
    buckets: dict[tuple, list] = {}
    for e in exps:
        buckets.setdefault(tuple(e.timepoints), []).append(e)
    return [Group(v) for _, v in sorted(buckets.items())]
