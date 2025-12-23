import torch
from torch import nn
import torch.nn.functional as F
import warnings
from typing import Optional
from ._model_utils import pairwise_euclidean_distance


def _sinkhorn_inline(
    M: torch.Tensor,
    r: Optional[torch.Tensor] = None,
    c: Optional[torch.Tensor] = None,
    gamma: float = 1.0,
    eps: float = 1e-6,
    maxiters: int = 1000,
) -> torch.Tensor:
    """Compute an entropy-regularized OT plan with Sinkhorn iterations.

    Intended to run under `torch.no_grad()` as part of an implicit-diff OT layer.

    Args:
        M: Cost matrix of shape (B, H, W).
        r: Row marginals (B, H) or (1, H). If None, uniform.
        c: Col marginals (B, W) or (1, W). If None, uniform.
        gamma: Entropic regularization inverse (larger => sharper).
        eps: Convergence tolerance on marginal error.
        maxiters: Max Sinkhorn iterations.
    """
    if M.ndim != 3:
        raise ValueError("M must have shape (B, H, W)")
    B, H, W = M.shape

    if r is None:
        r_ = torch.full((B, H, 1), 1.0 / float(H), device=M.device, dtype=M.dtype)
    else:
        if r.shape not in {(B, H), (1, H)}:
            raise ValueError(f"r must have shape (B,H) or (1,H), got {tuple(r.shape)}")
        r_ = r.unsqueeze(2).expand(B, H, 1)

    if c is None:
        c_ = torch.full((B, 1, W), 1.0 / float(W), device=M.device, dtype=M.dtype)
    else:
        if c.shape not in {(B, W), (1, W)}:
            raise ValueError(f"c must have shape (B,W) or (1,W), got {tuple(c.shape)}")
        c_ = c.unsqueeze(1).expand(B, 1, W)

    # Stable kernel: subtract row-wise minima (absorbed by scaling factors).
    M_min = torch.amin(M, dim=2, keepdim=True)
    P = torch.exp(-gamma * (M - M_min))

    tiny = torch.as_tensor(1e-30, device=M.device, dtype=M.dtype)
    for _ in range(int(maxiters)):
        alpha = torch.sum(P, dim=2, keepdim=True)
        P = P / torch.clamp(alpha, min=tiny) * r_

        beta = torch.sum(P, dim=1, keepdim=True)
        # Use L1 marginal error (bounded by 2 for probability vectors) to match
        # the semantics of the legacy `check_convergence` implementation.
        err = torch.sum(torch.abs(beta - c_), dim=2).max()
        if float(err.item()) <= float(eps):
            break
        P = P / torch.clamp(beta, min=tiny) * c_

    return P


class _OptimalTransportPlan(torch.autograd.Function):
    """Sinkhorn plan with implicit gradients (block solve).

    Forward runs Sinkhorn under `no_grad()` (no unrolled autograd graph).
    Backward uses an implicit differentiation formula; complexity scales with
    the *row* dimension (H) for method='block'.
    """

    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        M: torch.Tensor,
        r: Optional[torch.Tensor] = None,
        c: Optional[torch.Tensor] = None,
        gamma: float = 1.0,
        eps: float = 1e-6,
        maxiters: int = 1000,
        method: str = "block",
    ) -> torch.Tensor:
        if method not in {"block", "approx"}:
            raise ValueError("method must be 'block' or 'approx'")

        with torch.no_grad():
            inv_r_sum = None
            inv_c_sum = None

            if r is not None:
                inv_r_sum = 1.0 / torch.sum(r, dim=1, keepdim=True)
                r = r * inv_r_sum
            if c is not None:
                inv_c_sum = 1.0 / torch.sum(c, dim=1, keepdim=True)
                c = c * inv_c_sum

            P = _sinkhorn_inline(
                M,
                r=r,
                c=c,
                gamma=float(gamma),
                eps=float(eps),
                maxiters=int(maxiters),
            )

        ctx.save_for_backward(M, r, c, P)
        ctx.gamma = float(gamma)
        ctx.method = method
        ctx.inv_r_sum = inv_r_sum
        ctx.inv_c_sum = inv_c_sum
        return P

    @staticmethod
    def backward(ctx, dJdP: torch.Tensor):  # type: ignore[override]
        M, r, c, P = ctx.saved_tensors
        B, H, W = M.shape
        gamma = float(ctx.gamma)

        dJdM = -gamma * P * dJdP
        dJdr = None if (r is None or not ctx.needs_input_grad[1]) else torch.zeros_like(r)
        dJdc = None if (c is None or not ctx.needs_input_grad[2]) else torch.zeros_like(c)

        if ctx.method == "approx":
            return dJdM, dJdr, dJdc, None, None, None, None

        alpha = torch.sum(P, dim=2)
        beta = torch.sum(P, dim=1)
        beta = torch.clamp(beta, min=1e-30)

        vHAt1 = torch.sum(dJdM[:, 1:H, 0:W], dim=2).view(B, H - 1, 1)
        vHAt2 = torch.sum(dJdM, dim=1).view(B, W, 1)

        P_sub = P[:, 1:H, 0:W]
        PdivC = P_sub / beta.view(B, 1, W)
        RminusPPdivC = torch.diag_embed(alpha[:, 1:H]) - torch.bmm(P_sub, PdivC.transpose(1, 2))
        jitter = 1e-6
        RminusPPdivC = RminusPPdivC + jitter * torch.eye(H - 1, device=M.device, dtype=M.dtype).view(1, H - 1, H - 1)

        try:
            chol = torch.linalg.cholesky(RminusPPdivC)
        except Exception:
            chol = torch.eye(H - 1, device=M.device, dtype=M.dtype).view(1, H - 1, H - 1).repeat(B, 1, 1)
            for b in range(B):
                try:
                    chol[b, :, :] = torch.linalg.cholesky(RminusPPdivC[b, :, :])
                except Exception:
                    warnings.warn("Sinkhorn implicit backward encountered a singular matrix; using identity fallback.")

        block_12 = torch.cholesky_solve(PdivC, chol)

        v1 = torch.cholesky_solve(vHAt1, chol) - torch.bmm(block_12, vHAt2)

        # Avoid materializing the (W x W) Schur complement explicitly:
        # block_22 = block_12^T @ PdivC
        tmp = torch.bmm(PdivC, vHAt2)  # (B, H-1, 1)
        block22_vHAt2 = torch.bmm(block_12.transpose(1, 2), tmp)  # (B, W, 1)
        block12T_vHAt1 = torch.bmm(block_12.transpose(1, 2), vHAt1)  # (B, W, 1)
        v2 = vHAt2 / beta.view(B, W, 1) + block22_vHAt2 - block12T_vHAt1

        dJdM[:, 1:H, 0:W] -= v1 * P_sub
        dJdM -= v2.view(B, 1, W) * P

        if dJdr is not None:
            inv_r_sum = ctx.inv_r_sum
            if inv_r_sum is None:
                inv_r_sum = torch.ones((B, 1), device=M.device, dtype=M.dtype)
            dJdr = (
                inv_r_sum / gamma
                * (
                    torch.sum(r[:, 1:H] * v1.view(B, H - 1), dim=1, keepdim=True)
                    - torch.cat(
                        (
                            torch.zeros((B, 1), device=r.device, dtype=r.dtype),
                            v1.view(B, H - 1),
                        ),
                        dim=1,
                    )
                )
            )

        if dJdc is not None:
            inv_c_sum = ctx.inv_c_sum
            if inv_c_sum is None:
                inv_c_sum = torch.ones((B, 1), device=M.device, dtype=M.dtype)
            dJdc = inv_c_sum / gamma * (torch.sum(c * v2.view(B, W), dim=1, keepdim=True) - v2.view(B, W))

        return dJdM, dJdr, dJdc, None, None, None, None


class ETP(nn.Module):
    def __init__(
        self,
        sinkhorn_alpha,
        init_a_dist=None,
        init_b_dist=None,
        OT_max_iter=1000,
        stop_thr=0.5e-2,
        implicit: bool = True,
        implicit_method: str = "block",
    ):
        super().__init__()
        self.sinkhorn_alpha = sinkhorn_alpha
        self.OT_max_iter = OT_max_iter
        self.stop_thr = stop_thr
        self.implicit = bool(implicit)
        self.implicit_method = implicit_method
        self.init_a_dist = init_a_dist
        self.init_b_dist = init_b_dist

        if init_a_dist is not None:
            self.a_dist = init_a_dist

        if init_b_dist is not None:
            self.b_dist = init_b_dist

    def forward(self, x, y, return_cost_mean: bool = False):
        M = pairwise_euclidean_distance(x, y)
        device = M.device

        if self.init_a_dist is None:
            a = (torch.ones(M.shape[0]) / M.shape[0]).unsqueeze(1).to(device)
        else:
            a = F.softmax(self.a_dist, dim=0).to(device)

        if self.init_b_dist is None:
            b = (torch.ones(M.shape[1]) / M.shape[1]).unsqueeze(1).to(device)
        else:
            b = F.softmax(self.b_dist, dim=0).to(device)

        if self.implicit:
            n, m = M.shape
            a_vec = a.squeeze(1)
            b_vec = b.squeeze(1)

            transpose = n > m
            if transpose:
                M_in = M.t().unsqueeze(0)
                r_in = b_vec.unsqueeze(0)
                c_in = a_vec.unsqueeze(0)
            else:
                M_in = M.unsqueeze(0)
                r_in = a_vec.unsqueeze(0)
                c_in = b_vec.unsqueeze(0)

            P_in = _OptimalTransportPlan.apply(
                M_in,
                r_in,
                c_in,
                float(self.sinkhorn_alpha),
                float(self.stop_thr),
                int(self.OT_max_iter),
                str(self.implicit_method),
            ).squeeze(0)
            transp = P_in.t() if transpose else P_in
        else:
            # Legacy: Sinkhorn's algorithm unrolled in autograd (can OOM).
            log_a = torch.log(a + 1e-30)  # Shape: (n, 1)
            log_b = torch.log(b + 1e-30)  # Shape: (m, 1)

            log_u = torch.zeros_like(log_a)  # Shape: (n, 1)
            log_v = torch.zeros_like(log_b)  # Shape: (m, 1)

            log_K = -M * self.sinkhorn_alpha  # Shape: (n, m)

            err = 1
            cpt = 0
            while err > self.stop_thr and cpt < self.OT_max_iter:
                log_Ku = log_K.T + log_u.T  # Shape: (m, n)
                log_v = log_b - torch.logsumexp(log_Ku, dim=1).unsqueeze(1)  # Shape: (m, 1)

                log_Kv = log_K + log_v.T  # Shape: (n, m)
                log_u = log_a - torch.logsumexp(log_Kv, dim=1).unsqueeze(1)  # Shape: (n, 1)

                cpt += 1
                if cpt % 50 == 1:
                    log_K = log_K + log_u + log_v.T
                    log_u = torch.zeros_like(log_a)
                    log_v = torch.zeros_like(log_b)

                    err = self.check_convergence(log_K, log_u, log_v, a, b)

            u = torch.exp(log_u)  # Shape: (n, 1)
            v = torch.exp(log_v)  # Shape: (m, 1)
            K = torch.exp(log_K)  # Shape: (n, m)

            transp = u * (K * v.T)  # Shape: (n, m)

        # Compute transport cost: <P, M>
        loss_ETP = torch.sum(transp * M)

        if return_cost_mean:
            return loss_ETP, transp, M.mean()

        return loss_ETP, transp

    def check_convergence(self, log_K, log_u, log_v, a, b):
        """
        Check convergence by verifying the marginal constraints using absolute error.
        The transport plan is P = diag(u) @ K @ diag(v)
        We need: P @ 1 = a and P^T @ 1 = b
        """
        with torch.no_grad():
            # Row constraint: (u ⊙ (K @ v)) should equal a
            # In log domain: log(u) + log(K @ v) should equal log(a)
            log_Kv = log_K + log_v.T  # Shape: (n, m)
            log_Kv_sum = torch.logsumexp(
                log_Kv, dim=1, keepdim=True
            )  # log(K @ v), Shape: (n, 1)
            log_row_sums = log_u + log_Kv_sum  # log(u ⊙ (K @ v)), Shape: (n, 1)
            row_sums = torch.exp(log_row_sums)  # Shape: (n, 1)

            # Column constraint: (v ⊙ (K^T @ u)) should equal b
            # In log domain: log(v) + log(K^T @ u) should equal log(b)
            log_Ku = log_K.T + log_u.T  # Shape: (m, n)
            log_Ku_sum = torch.logsumexp(
                log_Ku, dim=1, keepdim=True
            )  # log(K^T @ u), Shape: (m, 1)
            log_col_sums = log_v + log_Ku_sum  # log(v ⊙ (K^T @ u)), Shape: (m, 1)
            col_sums = torch.exp(log_col_sums)  # Shape: (m, 1)

            # Compute absolute errors
            row_err = torch.abs(row_sums - a).sum()

            col_err = torch.abs(col_sums - b).sum()

            return max(row_err.item(), col_err.item())
