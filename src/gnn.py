"""Graph neural network models for tau propagation.

Two families, deliberately contrasted:

1. `TauODE` -- a *diffusion-constrained graph neural ODE*:

       dx/dt = -gamma * L_W x  +  x * [ alpha * m(g) * (1 - x) + eta * tanh(MLP([x, g])) ]

   with gene-gated edge weights on a learned anterograde/retrograde blend

       Abar   = sigmoid(d) * A_antero + (1 - sigmoid(d)) * A_retro
       W_ij   = Abar_ij * softplus(a + u . g_i) * softplus(b + v . g_j)
       L_W    = diag(rowsum W) - W^T        (so -L_W x = W^T x - deg * x)

   Setting u = v = 0, alpha = 0, eta = 0 recovers **exactly** the NDM heat kernel
   x(t) = exp(-gamma L t) x0 (verified numerically to 2e-5 relative error in
   `src/tests_sanity.py`). The GNN and its classical baseline are therefore *nested* models
   differing only in the learned terms, which makes "does the GNN beat network diffusion?"
   a well-posed nested-model question rather than an architecture beauty contest.

   The multiplicative `x *` on the local term encodes a biological constraint: a region with
   no pathology and no afferent input cannot spontaneously develop pathology.

2. `PlainGNN` -- an unconstrained GCN/GAT-style one-step predictor
   x_{t+1} = f(x_t, genes ; A). The "obvious" GNN a practitioner would reach for, included to
   quantify what the diffusion constraint is worth on data this small.

Everything runs on CPU: the graphs have 86-426 nodes and TauODE has ~60 parameters, so kernel
launch overhead dominates any GPU speedup (measured; see REPORT.md).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from metrics import LOG_EPS


def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def t_transform(x: torch.Tensor, scale: str) -> torch.Tensor:
    if scale == "raw":
        return x
    return torch.log10(torch.clamp(x, min=0.0) + LOG_EPS)


def corr_loss(pred: torch.Tensor, target: torch.Tensor, scale: str = "raw") -> torch.Tensor:
    """1 - spatial Pearson R, the exact evaluation metric.

    The classical baselines are fitted by maximising the same quantity, so no model is
    advantaged by an objective mismatch.
    """
    p = t_transform(pred, scale)
    t = t_transform(target, scale)
    p = p - p.mean()
    t = t - t.mean()
    return 1.0 - (p @ t) / (p.norm() * t.norm() + 1e-8)


# ======================================================================================
class TauODE(nn.Module):
    """Diffusion-constrained graph neural ODE for connectome tau spreading.

    Args:
        n_genes: number of node features (0 disables all gene terms).
        hidden: width of the residual local MLP.
        use_growth / use_edge_gate / use_mlp: ablation switches. With all three off and
            `fit_dir=False` the model *is* the NDM.
        fit_dir: learn the anterograde/retrograde blend.
        n_steps: fixed integration steps for one unit of normalised time.
    """

    def __init__(self, n_genes: int = 0, hidden: int = 8, use_growth: bool = True,
                 use_edge_gate: bool = True, use_mlp: bool = True, fit_dir: bool = True,
                 n_steps: int = 40):
        super().__init__()
        self.n_genes = n_genes
        self.use_growth = use_growth
        self.use_edge_gate = bool(use_edge_gate and n_genes > 0)
        self.use_mlp = use_mlp
        self.fit_dir = fit_dir
        self.n_steps = n_steps

        self.log_gamma = nn.Parameter(torch.tensor(0.5))     # softplus -> diffusion rate
        self.log_alpha = nn.Parameter(torch.tensor(0.5))     # softplus -> logistic growth rate
        self.dir_logit = nn.Parameter(torch.tensor(0.0))     # 0 -> equal antero/retro blend
        self.gene_growth = nn.Parameter(torch.zeros(max(n_genes, 1)))
        self.u = nn.Parameter(torch.zeros(max(n_genes, 1)))
        self.v = nn.Parameter(torch.zeros(max(n_genes, 1)))
        self.a = nn.Parameter(torch.tensor(0.5413))          # softplus(0.5413) ~ 1
        self.b = nn.Parameter(torch.tensor(0.5413))
        if use_mlp:
            self.mlp = nn.Sequential(nn.Linear(1 + n_genes, hidden), nn.Tanh(),
                                     nn.Linear(hidden, 1))
            for p in self.mlp[-1].parameters():
                nn.init.zeros_(p)                            # start exactly at the FKPP model
            self.log_eta = nn.Parameter(torch.tensor(-1.0))

    # ---- graph -------------------------------------------------------------------------
    def edge_weights(self, A_ant: torch.Tensor, A_ret: torch.Tensor,
                     g: torch.Tensor | None) -> torch.Tensor:
        if self.fit_dir:
            s = torch.sigmoid(self.dir_logit)
            A = s * A_ant + (1 - s) * A_ret
        else:
            A = 0.5 * (A_ant + A_ret)
        if not self.use_edge_gate or g is None or g.shape[1] == 0:
            return A
        su = F.softplus(self.a + g @ self.u[: g.shape[1]]).unsqueeze(1)    # source gate (N,1)
        sv = F.softplus(self.b + g @ self.v[: g.shape[1]]).unsqueeze(0)    # target gate (1,N)
        return A * su * sv

    def dxdt(self, x, W, deg, g, gmod):
        """RHS. `x` is (N,) or (N,B); everything broadcasts over the batch of experiments."""
        gamma = F.softplus(self.log_gamma)
        diff = (W.t() @ x) - (deg.unsqueeze(-1) * x if x.dim() == 2 else deg * x)
        out = gamma * diff
        if self.use_growth:
            alpha = F.softplus(self.log_alpha)
            m = gmod.unsqueeze(-1) if (gmod is not None and x.dim() == 2) else gmod
            out = out + alpha * (m if m is not None else 1.0) * x * (1.0 - x)
        if self.use_mlp:
            eta = torch.exp(self.log_eta)
            if x.dim() == 1:
                feats = x.unsqueeze(1) if (g is None or g.shape[1] == 0) \
                    else torch.cat([x.unsqueeze(1), g], 1)
                loc = self.mlp(feats).squeeze(1)
            else:
                N, B = x.shape
                xb = x.t().reshape(B, N, 1)
                feats = xb if (g is None or g.shape[1] == 0) \
                    else torch.cat([xb, g.unsqueeze(0).expand(B, N, g.shape[1])], 2)
                loc = self.mlp(feats).squeeze(-1).t()
            out = out + eta * x * torch.tanh(loc)
        return out

    def forward(self, seed: torch.Tensor, A_ant: torch.Tensor, A_ret: torch.Tensor,
                times: list[float], g: torch.Tensor | None = None) -> torch.Tensor:
        """Integrate from `seed` over *normalised* times; returns (T, N) or (T, N, B)."""
        W = self.edge_weights(A_ant, A_ret, g)
        deg = W.sum(1)
        gmod = None
        if self.use_growth and g is not None and g.shape[1] > 0:
            gmod = F.softplus(1.0 + g @ self.gene_growth[: g.shape[1]])
        tmax = max(times)
        n = max(4, int(np.ceil(tmax * self.n_steps)))
        dt = tmax / n
        x = seed
        res: list = [None] * len(times)
        order = sorted(range(len(times)), key=lambda k: times[k])
        ti, t = 0, 0.0
        for step in range(n + 1):
            while ti < len(times) and t >= times[order[ti]] - 1e-9:
                res[order[ti]] = x
                ti += 1
            if step == n:
                break
            k1 = self.dxdt(x, W, deg, g, gmod)                       # midpoint (RK2)
            k2 = self.dxdt(torch.clamp(x + 0.5 * dt * k1, 0, 5), W, deg, g, gmod)
            x = torch.clamp(x + dt * k2, 0, 5)
            t += dt
        for k in range(len(times)):
            if res[k] is None:
                res[k] = x
        return torch.stack(res)


# ======================================================================================
class PlainGNN(nn.Module):
    """Unconstrained message-passing GNN one-step predictor: x_{t+1} = f(x_t, genes ; A).

    Dense normalised-adjacency propagation (equivalent to GCNConv on a dense weighted graph),
    optionally with edge attention (`conv='gat'`). Predicts a non-negative multiplicative
    change on the input scale, which is the closest unconstrained analogue of a growth model.
    """

    def __init__(self, n_genes: int = 0, hidden: int = 16, layers: int = 2,
                 conv: str = "gcn", dropout: float = 0.1):
        super().__init__()
        self.conv = conv
        d_in = 1 + n_genes
        dims = [d_in] + [hidden] * layers
        self.lins = nn.ModuleList(nn.Linear(dims[i], dims[i + 1]) for i in range(layers))
        self.self_lins = nn.ModuleList(nn.Linear(dims[i], dims[i + 1]) for i in range(layers))
        self.att = nn.Linear(2 * hidden, 1) if conv == "gat" else None
        self.head = nn.Linear(hidden + d_in, 1)
        self.dropout = dropout

    def forward(self, x0, A_norm, g=None):
        feats = x0.unsqueeze(1) if (g is None or g.shape[1] == 0) \
            else torch.cat([x0.unsqueeze(1), g], 1)
        h = feats
        for lin, slin in zip(self.lins, self.self_lins):
            h = torch.relu(A_norm @ lin(h) + slin(h))
            h = F.dropout(h, p=self.dropout, training=self.training)
        out = self.head(torch.cat([h, feats], 1)).squeeze(1)
        return torch.clamp(x0 * torch.exp(torch.clamp(out, -6, 6)), 0.0, 5.0)


def sym_norm(A: np.ndarray) -> np.ndarray:
    """D^-1/2 (A + I) D^-1/2 -- the standard GCN propagation operator."""
    A = np.asarray(A, float).copy()
    np.fill_diagonal(A, 0.0)
    A = A + np.eye(len(A))
    d = np.maximum(A.sum(1), 1e-12)
    dinv = 1.0 / np.sqrt(d)
    return A * dinv[:, None] * dinv[None, :]


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)
