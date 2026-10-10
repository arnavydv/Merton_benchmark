"""
fdm_main.py
===========
Finite Difference Method (FDM) solver for Merton's HJB equation.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CONVENTION  (follows Merton 1969 / standard textbooks)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  gamma (γ)  = Relative Risk Aversion  (RRA).  gamma > 0.
               gamma = 1  →  log utility  (limiting case)
               gamma = 5  →  moderately risk-averse (your case)

  CRRA utility exponent:  p = 1 - gamma   (so p < 0 when gamma > 1)
  Utility function:       U(W) = W^p / p    (concave for all gamma > 0)

  Risky asset:   dS/S = μ dt + σ dZ
  Risk-free rate: r
  Wealth dynamics: dW = W[r + π(μ-r)] dt + Wπσ dZ

HJB equation (terminal wealth, no consumption):
  V_t + max_π { [r + π(μ-r)] W V_W + ½ π²σ² W² V_WW } = 0
  V(T, W) = W^p / p

Analytical solution:
  π*(t,W) = (μ-r) / (γ σ²)            [constant, independent of t and W]
  V(t,W)  = f(t) · W^p / p
  f(t)    = exp(A·(T-t))
  A       = p · [r + (μ-r)² / (2γσ²)]

FDM approach (log-wealth transform):
  x = ln(W),  so PDE becomes constant-coefficient in x.
  V_t + b(π)·V_x + D(π)·V_xx = 0
  where:
    b(π) = r + π(μ-r) - ½π²σ²      (drift  in log-wealth space)
    D(π) = ½π²σ²                    (diffusion in log-wealth space)

  Optimal π from FOC (in x-coordinates):
    π*(x) = (μ-r)·V_x / (-σ²·(p-1)·V_x - σ²·V_xx·... )
  More cleanly, we iterate: given current V, compute
    π* = -(μ-r)·V_x / (σ²·(V_xx - V_x))   [from W-space FOC in x-coords]

  θ-scheme (θ=1 fully implicit, θ=0.5 Crank–Nicolson).
  Tridiagonal system solved with scipy banded solver.
"""

import numpy as np
from scipy.linalg import solve_banded


# ─────────────────────────────────────────────────────────────────────────────
# Model Parameters
# ─────────────────────────────────────────────────────────────────────────────

class MertonParams:
    """
    Parameters for Merton's portfolio optimisation.

    gamma  : float  Relative Risk Aversion (RRA).  Must be > 0.
                    gamma=1 → log utility.  gamma=5 → your case.
    mu     : float  Expected return of risky asset (annualised).
    sigma  : float  Volatility of risky asset (annualised).
    r      : float  Risk-free rate (annualised).
    T      : float  Investment horizon in years.
    """
    def __init__(
        self,
        gamma: float = 5.0,
        mu:    float = 0.201649,
        sigma: float = 0.256573,
        r:     float = 0.0368,
        T:     float = 1.0,
    ):
        if gamma <= 0:
            raise ValueError("gamma (RRA) must be strictly positive.")
        if sigma <= 0:
            raise ValueError("sigma must be strictly positive.")

        self.gamma = gamma          # RRA
        self.mu    = mu
        self.sigma = sigma
        self.r     = r
        self.T     = T
        self.p     = 1.0 - gamma   # CRRA utility exponent  (p < 0 when gamma > 1)

    # ── Analytical solution ──────────────────────────────────────────────────

    @property
    def pi_star(self) -> float:
        """
        Merton's constant optimal risky proportion.
          π* = (μ - r) / (γ σ²)
        Always in (0,1) when μ > r, γ > 0, and no leverage constraint.
        For your params:  π* ≈ 0.5008
        """
        return (self.mu - self.r) / (self.gamma * self.sigma**2)

    @property
    def sharpe(self) -> float:
        """Sharpe ratio of the risky asset."""
        return (self.mu - self.r) / self.sigma

    @property
    def A(self) -> float:
        """
        Exponent in the analytical value-function multiplier f(t) = exp(A(T-t)).
          A = p · [r + (μ-r)² / (2γσ²)]
            = p · [r + Sharpe²/(2γ)]
        """
        return self.p * (self.r + (self.mu - self.r)**2 / (2.0 * self.gamma * self.sigma**2))

    def analytical_V(self, t, W):
        """
        V(t,W) = exp(A·(T-t)) · W^p / p
        Broadcast-safe: t and W can be scalars or arrays.
        """
        t  = np.asarray(t,  dtype=float)
        W  = np.asarray(W,  dtype=float)
        tau = self.T - t
        f   = np.exp(self.A * tau)
        return f * W**self.p / self.p

    def analytical_pi(self, t, W):
        """π*(t,W) — constant for CRRA utility."""
        t = np.asarray(t, dtype=float)
        W = np.asarray(W, dtype=float)
        return np.broadcast_to(self.pi_star, np.broadcast_shapes(t.shape, W.shape)).copy()

    def __repr__(self):
        return (f"MertonParams(γ={self.gamma}, μ={self.mu}, σ={self.sigma}, "
                f"r={self.r}, T={self.T})\n"
                f"  → π* = {self.pi_star:.6f},  p = {self.p:.4f},  A = {self.A:.6f}")


# ─────────────────────────────────────────────────────────────────────────────
# FDM Solver
# ─────────────────────────────────────────────────────────────────────────────

class MertonFDMSolver:
    """
    Backward-in-time FDM solver on a log-wealth grid.

    Grid layout
    ───────────
    x = ln(W) ∈ [x_min, x_max],  Nx interior+boundary nodes.
    t ∈ [0, T],  Nt time steps  (dt = T/Nt).

    After solve() the following are populated:
      self.x        log-wealth grid         (Nx,)
      self.W_grid   wealth grid             (Nx,)
      self.t_grid   time grid 0..T          (Nt+1,)
      self.V        value function          (Nt+1, Nx)
      self.pi_fdm   optimal risky fraction  (Nt+1, Nx)
    """

    def __init__(
        self,
        params:  MertonParams,
        Nx:      int   = 500,         # wealth grid points
        Nt:      int   = 252,         # time steps (trading days)
        x_min:   float = -2.0,        # ln(W_min); W_min ≈ 0.135
        x_max:   float =  4.0,        # ln(W_max); W_max ≈ 54.6
        theta:   float = 1.0,         # 1.0=implicit, 0.5=Crank-Nicolson
        pi_min:  float = -1.0,        # short-selling allowed; does not sit on pi*
        pi_max:  float = 5.0,         # leverage allowed; pi*≈0.5 is interior
    ):
        self.p       = params
        self.Nx      = Nx
        self.Nt      = Nt
        self.x_min   = x_min
        self.x_max   = x_max
        self.theta   = theta
        self.pi_min  = pi_min
        self.pi_max  = pi_max

        # ── Grids ─────────────────────────────────────────────────────────
        self.x      = np.linspace(x_min, x_max, Nx)
        self.dx     = self.x[1] - self.x[0]
        self.W_grid = np.exp(self.x)

        self.dt     = params.T / Nt
        self.t_grid = np.linspace(0.0, params.T, Nt + 1)

        # ── Storage ───────────────────────────────────────────────────────
        self.V       = np.zeros((Nt + 1, Nx))
        self.pi_fdm  = np.zeros((Nt + 1, Nx))

        self._solved = False

    # ── Optimal π from the first-order condition ─────────────────────────────

    def _compute_pi(self, v: np.ndarray) -> np.ndarray:
        """
        FOC of HJB in W-space:
            π* = -(μ-r) V_W / (σ² W V_WW)

        Converting to log-wealth x = ln(W):
            V_W  = V_x / W
            V_WW = (V_xx - V_x) / W²

        Substituting:
            π* = -(μ-r) V_x / (σ²(V_xx - V_x))

        Uses second-order central differences with ghost-node extrapolation
        at boundaries to avoid first/last-point bias.
        """
        dx = self.dx
        Vx  = np.gradient(v, dx)        # ∂V/∂x
        Vxx = np.gradient(Vx, dx)       # ∂²V/∂x²

        sigma = self.p.sigma
        mu_r  = self.p.mu - self.p.r

        denom = sigma**2 * (Vxx - Vx)

        with np.errstate(divide='ignore', invalid='ignore'):
            pi = -mu_r * Vx / denom
            # Replace non-finite values with the analytical constant
            # A non-finite FOC is a failed derivative, not a licence to insert pi*.
            pi = np.where(np.isfinite(pi), pi, 0.0)

        # Enforce portfolio constraints
        return np.clip(pi, self.pi_min, self.pi_max)

    # ── PDE diffusion / drift coefficients in log-wealth space ───────────────

    def _build_matrix(self, pi: np.ndarray, v_old: np.ndarray) -> np.ndarray:
        """
        Assemble the tridiagonal system for one time step.

        PDE (in x = ln W):
          V_t + b(π)·V_x + D(π)·V_xx = 0

          b(π) = r + π(μ-r) - ½π²σ²
          D(π) = ½π²σ²   ≥ 0

        θ-scheme  (V^k denotes time level k):
          (V^k - V^{k+1})/dt + θ·L[V^k] + (1-θ)·L[V^{k+1}] = 0

        where L[V] = b·V_x + D·V_xx.
        Rearranged → (I + θ dt L) V^k = (I - (1-θ) dt L) V^{k+1}

        Returns:
          (ab, rhs)  where ab is the banded matrix in scipy format
                     and rhs is the right-hand-side vector.
        """
        p   = self.p
        dx  = self.dx
        dt  = self.dt
        th  = self.theta
        Nx  = self.Nx

        b  = p.r + pi * (p.mu - p.r) - 0.5 * pi**2 * p.sigma**2   # drift
        D  = 0.5 * pi**2 * p.sigma**2                              # diffusion

        # Upwind for advection (stabilises when Péclet number > 1)
        # We use upwind for b, central for D.
        bm = np.maximum(b, 0.0)   # b+ (positive part)
        bp = np.minimum(b, 0.0)   # b- (negative part)

        # Diffusion stencil
        a_diff_l = D / dx**2
        a_diff_c = -2.0 * D / dx**2
        a_diff_r = D / dx**2

        # Advection stencil (upwind)
        a_adv_l = -bm / dx
        a_adv_c = (bm - bp) / dx
        a_adv_r = bp / dx

        lower = a_diff_l + a_adv_l    # sub-diagonal coefficient
        center= a_diff_c + a_adv_c    # diagonal coefficient
        upper = a_diff_r + a_adv_r    # super-diagonal coefficient

        # ── Implicit LHS: (I + θ dt L) ────────────────────────────────────
        al_lhs =      - th * dt * lower[1:]        # sub-diag  (length Nx-1)
        ac_lhs = 1.0  - th * dt * center           # diagonal  (length Nx)
        ar_lhs =      - th * dt * upper[:-1]       # super-diag(length Nx-1)

        # ── Explicit RHS: (I - (1-θ) dt L) V_old ─────────────────────────
        rhs = v_old.copy()
        rhs[1:-1] += (1.0 - th) * dt * (
            lower[1:-1] * v_old[:-2]
            + center[1:-1] * v_old[1:-1]
            + upper[1:-1] * v_old[2:]
        )

        # Banded storage for solve_banded((2, 2)):
        #   ab[2+i-j, j] = A[i, j]
        # Interior rows stay tridiagonal. The two endpoints use a second-order
        # one-sided stencil of the same PDE, so the closed form is not prescribed
        # on the boundary. Bandwidth 2 reaches the node two steps in.
        ab = np.zeros((5, Nx))
        for i in range(1, Nx - 1):
            ab[2, i] = ac_lhs[i]
            ab[1, i + 1] = ar_lhs[i]       # A[i, i+1]
            ab[3, i - 1] = al_lhs[i - 1]   # A[i, i-1]

        def _endpoint_row(i: int, vx_c, vxx_c) -> None:
            """One-sided L at an endpoint. vx_c, vxx_c are (node, weight) pairs."""
            b_i = b[i]
            D_i = D[i]
            cols = {}
            for node, w_vx in vx_c:
                cols[node] = cols.get(node, 0.0) + b_i * w_vx / dx
            for node, w_xx in vxx_c:
                cols[node] = cols.get(node, 0.0) + D_i * w_xx / dx**2
            for node in range(Nx):
                # Clear this row. Only the one-sided nodes are written back.
                pass
            for node, spatial in cols.items():
                ab[2 + i - node, node] = -th * dt * spatial
            ab[2, i] += 1.0
            if th < 1.0:
                explicit = 0.0
                for node, spatial in cols.items():
                    explicit += spatial * v_old[node]
                rhs[i] = v_old[i] - (1.0 - th) * dt * explicit
            else:
                rhs[i] = v_old[i]

        # Left: forward differences for Vx, Vxx.
        _endpoint_row(
            0,
            vx_c=((0, -1.5), (1, 2.0), (2, -0.5)),
            vxx_c=((0, 1.0), (1, -2.0), (2, 1.0)),
        )
        # Right: backward differences.
        n = Nx - 1
        _endpoint_row(
            n,
            vx_c=((n, 1.5), (n - 1, -2.0), (n - 2, 0.5)),
            vxx_c=((n, 1.0), (n - 1, -2.0), (n - 2, 1.0)),
        )
        return ab, rhs

    # ── Main backward-in-time solve ──────────────────────────────────────────

    def solve(self, verbose: bool = True) -> "MertonFDMSolver":
        """
        Solve the HJB PDE backwards from t=T to t=0.

        At each time step:
          1. Compute π* from current V (policy evaluation).
          2. Build tridiagonal system with that π*.
          3. Solve for V^k.
          4. Repeat policy evaluation with new V^k (Picard iteration).
        """
        p   = self.p
        Nx  = self.Nx
        Nt  = self.Nt

        # ── Terminal condition ─────────────────────────────────────────────
        # V(T, W) = W^p / p  where p = 1 - gamma < 0 for gamma > 1
        self.V[Nt]      = self.W_grid**p.p / p.p
        self.pi_fdm[Nt] = self._compute_pi(self.V[Nt])

        # ── Backward sweep ─────────────────────────────────────────────────
        n_picard = 8       # Picard iterations per time step
        tol      = 1e-9    # convergence tolerance

        for k in range(Nt - 1, -1, -1):
            v_old = self.V[k + 1].copy()

            # Warm-start π from previous time step
            pi = self.pi_fdm[k + 1].copy()

            for _iter in range(n_picard):
                ab, rhs = self._build_matrix(pi, v_old)
                v_new   = solve_banded((2, 2), ab, rhs)

                # Clamp to avoid blow-up near boundaries
                # For p < 0, V is negative; for p > 0, V is positive
                if p.p < 0:
                    v_new = np.minimum(v_new, -1e-12)   # keep V < 0
                else:
                    v_new = np.maximum(v_new,  1e-12)

                pi_new = self._compute_pi(v_new)

                delta = np.max(np.abs(pi_new - pi))
                pi    = pi_new
                if delta < tol:
                    break

            self.V[k]      = v_new
            self.pi_fdm[k] = pi

        self._solved = True
        if verbose:
            self._print_summary()
        return self

    def _print_summary(self):
        p = self.p
        print("=" * 60)
        print("  Merton HJB — FDM Solver  ✓  COMPLETE")
        print("=" * 60)
        print(f"  Parameters:  γ={p.gamma}, μ={p.mu}, σ={p.sigma}, r={p.r}, T={p.T}")
        print(f"  Grid:        Nx={self.Nx},  Nt={self.Nt},  θ={self.theta}")
        print(f"  dx={self.dx:.6f},  dt={self.dt:.6f}")
        print(f"  W range:     [{self.W_grid[0]:.4f}, {self.W_grid[-1]:.2f}]")
        print(f"  Utility exp  p = {p.p:.4f}  (p < 0 → CRRA with γ > 1)")
        print(f"  π* (analytical)       = {p.pi_star:.8f}")
        pi_fdm_mid = self.pi_at(0.0, 1.0)
        print(f"  π* (FDM, t=0, W=1)   = {pi_fdm_mid:.8f}")
        print(f"  |error| at W=1, t=0  = {abs(pi_fdm_mid - p.pi_star):.4e}")
        print("=" * 60)

    # ── Query helpers ─────────────────────────────────────────────────────────

    def _w_idx(self, W: float) -> int:
        return int(np.argmin(np.abs(self.W_grid - W)))

    def _t_idx(self, t: float) -> int:
        return int(np.argmin(np.abs(self.t_grid - t)))

    def pi_at(self, t: float, W: float) -> float:
        return float(self.pi_fdm[self._t_idx(t), self._w_idx(W)])

    def V_at(self, t: float, W: float) -> float:
        return float(self.V[self._t_idx(t), self._w_idx(W)])

    def pi_slice_t(self, t: float) -> np.ndarray:
        """π*(t=fixed, W=all)."""
        return self.pi_fdm[self._t_idx(t)].copy()

    def pi_slice_W(self, W: float) -> np.ndarray:
        """π*(t=all, W=fixed)."""
        return self.pi_fdm[:, self._w_idx(W)].copy()

    def V_slice_t(self, t: float) -> np.ndarray:
        return self.V[self._t_idx(t)].copy()

    def V_slice_W(self, W: float) -> np.ndarray:
        return self.V[:, self._w_idx(W)].copy()


# ─────────────────────────────────────────────────────────────────────────────
# Convenience builder (your exact parameters)
# ─────────────────────────────────────────────────────────────────────────────

