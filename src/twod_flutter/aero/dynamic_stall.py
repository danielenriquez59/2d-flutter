"""Dynamic stall and the low-Mach modifications.

States x11, x12, x14 (paper Eqs. 13-15 and 17-20).

Section 3 of the paper -- the low-Mach modification -- is its actual
contribution, and lives here:

* Eq. (17) adds a *second* lag ``x14`` on top of ``x9``, so stall onset is
  triggered by ``|C_N''| > C_N1`` rather than ``|C_N'| > C_N1``. This is what
  fixes "stall onset predicted too early".
* Eqs. (18)-(20) add the post-onset overshoot in normal force and the matching
  pitching-moment increment.
"""

from __future__ import annotations

import numpy as np

from ..config import Airfoil
from .separation import kirchhoff_f, alpha1n


def kirchhoff_kn(f: float) -> float:
    """``K_N = ((1 + sqrt(f))/2)^2`` -- the Kirchhoff lift-reduction factor."""
    return (0.5 * (1.0 + np.sqrt(max(f, 0.0)))) ** 2


def vortex_feed(c_n_circ: float, x10: float) -> float:
    """``c_v`` -- the strength of the vortex-induced normal force (Eq. 15).

    **GAP-3.** Not defined in the paper. Canonical Leishman-Beddoes takes it as
    the lift *deficit* caused by separation -- the circulatory normal force
    that Kirchhoff theory says the section is no longer carrying, which is what
    physically rolls up into the leading-edge vortex:

        c_v = C_N^C * (1 - K_N)
    """
    return c_n_circ * (1.0 - kirchhoff_kn(x10))


def vortex_feed_rate(
    c_n_circ: float, dc_n_circ_ds: float, x10: float, dx10_ds: float
) -> float:
    """``d(c_v)/ds``, differentiated analytically.

    Eq. (15) needs ``c_v_dot`` both as a source term and inside its switch
    condition. Computing it analytically rather than by finite difference keeps
    the RHS a pure function of the state.

        d(c_v)/ds = (1 - K_N) * d(C_N^C)/ds
                    - C_N^C * (1 + sqrt(f)) / (4*sqrt(f)) * d(f)/ds
    """
    f = max(x10, 0.0)
    sqrt_f = np.sqrt(f)
    d_kn_df = (1.0 + sqrt_f) / (4.0 * sqrt_f) if sqrt_f > 1e-9 else 0.0
    return (1.0 - kirchhoff_kn(f)) * dc_n_circ_ds - c_n_circ * d_kn_df * dx10_ds


def onset_lag_derivative(x9: float, x14: float, af: Airfoil) -> float:
    """d(x14)/ds -- the low-Mach onset lag, Eq. (17).

    A first-order lag of ``x9`` (``C_N'``) with time constant ``T_b``, giving
    the paper's ``C_N''``.
    """
    return (x9 - x14) / af.T_b


def is_stalled(x14: float, af: Airfoil) -> bool:
    """Modified low-Mach dynamic-stall criterion: ``|C_N''| > C_N1`` (Eq. 17).

    Supersedes the baseline criterion ``|C_N'| > C_N1`` of Eq. (13).
    """
    return abs(x14) > af.C_N1


def vortex_normal_force_derivative(
    x11: float,
    x12: float,
    alpha: float,
    cv_dot: float,
    sigma_2: float,
    af: Airfoil,
) -> float:
    """d(x12)/ds -- vortex-induced normal force, Eq. (15).

    Accumulates while the vortex is being fed *and* is still over the section;
    otherwise decays. ``alpha * cv_dot >= 0`` tests that loading and vortex
    strength move together.
    """
    decay = -x12 / (sigma_2 * af.T_v)
    feeding = (alpha * cv_dot >= 0.0) and (0.0 < x11 < 2.0 * af.T_vl)
    return cv_dot + decay if feeding else decay


def vortex_shape(tau: float, af: Airfoil) -> float:
    """``V_x`` -- vortex shape function, Eq. (19).

    Grows as ``sin^{3/2}`` while the vortex traverses the chord, then decays as
    ``cos^2`` after it passes the trailing edge.
    """
    if tau <= 0.0:
        return 0.0
    if tau <= af.T_v:
        return np.sin(np.pi * tau / (2.0 * af.T_v)) ** 1.5
    arg = np.pi * (tau - af.T_v) / af.T_vl
    # Past this the cos^2 has completed its lobe and the vortex is gone;
    # without the clamp it would spuriously regrow.
    if arg >= 0.5 * np.pi:
        return 0.0
    return np.cos(arg) ** 2


def normal_force_overshoot(
    x10: float, alpha: float, x11: float, af: Airfoil
) -> float:
    """``dC_N^v`` -- low-Mach normal-force overshoot, Eq. (18).

    Proportional to the gap between the delayed separation point ``f'' = x10``
    and its static counterpart ``f``, shaped by ``V_x``.
    """
    f_static = kirchhoff_f(alpha, alpha1n(x10, af), af)
    return af.B1 * (x10 - f_static) * vortex_shape(x11, af)


def moment_overshoot(dc_n_v: float, x11: float, af: Airfoil) -> float:
    """``dC_m^v`` -- matching pitching-moment increment, Eq. (20).

    .. note::
       Eq. (19) writes ``tau`` and Eq. (20) writes ``tau_v``; they are taken to
       be the same quantity, ``x11`` (theory.md §14, defect 3).
    """
    return af.B2 * (1.0 - np.cos(np.pi * x11 / af.T_v)) * dc_n_v
