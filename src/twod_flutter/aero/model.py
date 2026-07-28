"""The 14-state modified Leishman-Beddoes aerodynamic model.

State layout (paper numbering ``x1..x14``, zero-based here):

===== ====== =========================================================
index paper  meaning
===== ====== =========================================================
0-7   x1-x8  attached-flow indicial states (Eq. 8)
8     x9     ``C_N'``  leading-edge pressure lag (Eq. 9)
9     x10    ``f''``   delayed separation point (Eq. 12)
10    x11    ``tau``   vortex convection clock (Eq. 14)
11    x12    ``C_N^v`` vortex-induced normal force (Eq. 15)
12    x13    separation point during reattachment (Eq. 16)
13    x14    ``C_N''`` low-Mach onset lag -- the stall trigger (Eq. 17)
===== ====== =========================================================

All ``derivatives`` results are rates with respect to non-dimensional time
``s = V t / b``. The caller multiplies by ``V/b`` for real time.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import Airfoil
from . import airloads as al
from . import attached, dynamic_stall as ds, separation as sep

N_AERO = 14

# Named indices, so the physics reads like the paper.
I_X9, I_X10, I_X11, I_X12, I_X13, I_X14 = 8, 9, 10, 11, 12, 13


@dataclass
class Airloads:
    """Airload coefficients plus the diagnostics worth plotting."""

    c_n: float
    c_m: float
    c_c: float
    c_l: float
    c_d: float
    alpha_e: float
    f_static: float
    f_delayed: float
    c_n_potential: float
    stalled: bool


class LBModel:
    """Modified Leishman-Beddoes model for one airfoil at one Mach number."""

    def __init__(self, airfoil: Airfoil, mach: float):
        self.af = airfoil
        self.mach = mach

    # -- initial condition ------------------------------------------------
    def initial_state(self, alpha0: float = 0.0) -> np.ndarray:
        """Steady state consistent with a constant ``alpha0``.

        Starting from the converged steady state rather than zeros removes the
        startup transient, which otherwise contaminates the first cycle of a
        forced-pitch run.
        """
        x = np.zeros(N_AERO)
        x[:8] = attached.steady_state(alpha0, self.af, self.mach)
        # Pressure lag settles at the steady potential load.
        c_n_steady = self.af.c_n_alpha_rad * alpha0
        x[I_X9] = c_n_steady
        f0 = sep.kirchhoff_f(alpha0, self.af.alpha1_deg, self.af)
        x[I_X10] = f0
        x[I_X11] = 0.0
        x[I_X12] = 0.0
        x[I_X13] = f0
        x[I_X14] = c_n_steady
        return x

    # -- dynamics ---------------------------------------------------------
    def derivatives(
        self, x: np.ndarray, alpha: float, q: float, dalpha_ds: float
    ) -> np.ndarray:
        """d(x)/ds for the 14 aerodynamic states.

        ``dalpha_ds`` sets the loading direction used by the sigma1, sigma2 and
        alpha_1n switch tables (theory.md §9.2).
        """
        af = self.af
        dx = np.zeros(N_AERO)

        # -- attached flow, x1..x8 (single pass; see attached.evaluate) ----
        dxa, alpha_e, c_n_circ, c_n_i, _ = attached.evaluate(
            x[:8], alpha, q, af, self.mach
        )
        dx[:8] = dxa
        c_n_pot = c_n_circ + c_n_i

        # -- x9: leading-edge pressure lag --------------------------------
        dx[I_X9] = sep.pressure_lag_derivative(x[I_X9], c_n_pot, af)

        # Loading direction, alpha * d(alpha)/ds: >= 0 loading up, < 0 unloading.
        # This is the switch variable for the T_f, T_v and alpha_1 tables
        # (Chantharasenawong 2007, Tables 2.3-2.4 and Eq. 2.54).
        loading = alpha * dalpha_ds

        # "Shedding" selects the vortex-shedding branch of those tables. The
        # thesis keys it on |x9| >= C_N1; we use the paper's modified low-Mach
        # criterion on x14 instead, since that is what Eq. (17) replaces it with.
        shedding = ds.is_stalled(x[I_X14], af)

        # -- x10: delayed separation point --------------------------------
        s1 = sep.sigma1(loading, x[I_X10], x[I_X11], shedding, af)
        dx[I_X10] = sep.separation_point_derivative(
            x[I_X9], x[I_X10], s1, loading, af
        )

        # -- x11: vortex clock. Reset is a discrete event, see step_events.
        dx[I_X11] = 1.0

        # -- x12: vortex-induced normal force -----------------------------
        # d(alpha_E)/ds follows from alpha_E = beta^2*(A1*b1*x1 + A2*b2*x2).
        bsq = attached.beta(self.mach) ** 2
        dalpha_e_ds = bsq * (af.A1 * af.b1 * dx[0] + af.A2 * af.b2 * dx[1])
        dc_n_circ_ds = af.c_n_alpha_rad * dalpha_e_ds
        cv_dot = ds.vortex_feed_rate(
            c_n_circ, dc_n_circ_ds, x[I_X10], dx[I_X10]
        )
        if af.vortex_overshoot_mode == "feed":
            # Eq. (18) read as an addition to the vortex FEED rather than to
            # the finished load: dC_N^v joins c_v and is integrated by Eq. (15).
            f_stat = sep.kirchhoff_f(alpha, af.alpha1_deg, af)
            df_stat_ds = (
                sep.kirchhoff_f_gradient(alpha, af.alpha1_deg, af) * dalpha_ds
            )
            cv_dot += ds.overshoot_rate(
                x[I_X10], dx[I_X10], f_stat, df_stat_ds, x[I_X11], af
            )
        s2 = ds.sigma2(loading, x[I_X11], shedding, af)
        dx[I_X12] = ds.vortex_normal_force_derivative(
            x[I_X11], x[I_X12], alpha, cv_dot, s2, af
        )

        # -- x13: reattachment separation point ---------------------------
        dx[I_X13] = sep.reattachment_derivative(x[I_X13], alpha, loading, af)

        # -- x14: low-Mach onset lag --------------------------------------
        dx[I_X14] = ds.onset_lag_derivative(x[I_X9], x[I_X14], af)

        return dx

    def step_events(
        self, x: np.ndarray, was_stalled: bool
    ) -> tuple[np.ndarray, bool]:
        """Apply discrete events between integration steps.

        Eq. (14) integrates the vortex clock at ``V/b`` but the
        paper never states when it resets. Chantharasenawong (2007) §2.4 supplies
        the rule. We reset ``x11`` to zero on each *rising* crossing of the
        stall criterion, i.e. at each new shedding event.
        """
        stalled = ds.is_stalled(x[I_X14], self.af)
        if stalled and not was_stalled:
            x = x.copy()
            x[I_X11] = 0.0
        return x, stalled

    # -- outputs ----------------------------------------------------------
    def airloads(self, x: np.ndarray, alpha: float, q: float) -> Airloads:
        """Assemble the airload coefficients, Eqs. (2)-(7)."""
        af = self.af

        _, alpha_e, c_n_circ, c_n_i, c_m_i = attached.evaluate(
            x[:8], alpha, q, af, self.mach
        )
        c_n_pot = c_n_circ + c_n_i

        f_delayed = float(np.clip(x[I_X10], 0.0, 1.0))
        f_static = sep.kirchhoff_f(alpha, af.alpha1_deg, af)  # static ref, Eq. (18)

        # Separated circulatory loads (Eqs. 5-7 superscript f).
        c_n_f = al.separated_normal_force(alpha_e, f_delayed, af)
        c_m_f = al.separated_moment(c_n_f, f_delayed, af)
        c_c_f = al.chord_force(alpha_e, f_delayed, af)

        # Vortex loads: state x12 plus the paper's low-Mach overshoot.
        # theory.md §7.2 flags that the paper never states whether Eq. (18) is
        # additive to x12 or replaces it; additive is the reading consistent
        # with the word "overshoot", and is what we adopt.
        if af.vortex_overshoot_mode == "feed":
            # Already integrated into x12 by the RHS -- adding it here too
            # would double-count.
            d_c_n_v = 0.0
            c_n_v = x[I_X12]
        else:
            d_c_n_v = ds.normal_force_overshoot(f_delayed, alpha, x[I_X11], af)
            c_n_v = x[I_X12] + d_c_n_v
        c_m_v = al.vortex_moment(x[I_X12], x[I_X11], af)
        c_m_v += ds.moment_overshoot(d_c_n_v, x[I_X11], af)

        c_n = c_n_i + c_n_f + c_n_v
        c_m = c_m_i + c_m_f + c_m_v + af.C_m0
        c_c = c_c_f
        c_l, c_d = al.lift_and_drag(c_n, c_c, alpha_e, alpha)

        return Airloads(
            c_n=c_n,
            c_m=c_m,
            c_c=c_c,
            c_l=c_l,
            c_d=c_d,
            alpha_e=alpha_e,
            f_static=f_static,
            f_delayed=f_delayed,
            c_n_potential=c_n_pot,
            stalled=ds.is_stalled(x[I_X14], af),
        )
