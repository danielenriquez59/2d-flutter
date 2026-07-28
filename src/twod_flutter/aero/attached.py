"""Unsteady attached flow -- states x1..x8 (paper Eq. 8).

    xdot_tilde = A x_tilde + B u,    u = [alpha, q]^T

**GAP-1.** The paper does not print ``A`` and ``B``; it defers to Ref. [16]
(DYMORE manual). What follows is the canonical Leishman-Beddoes / Leishman-Crouse
construction, reproduced here so the model runs end to end. It must be checked
against Ref. [16] before any result is trusted. See ``docs/theory.md`` §9.

Layout
------
====== ==================================================================
x1, x2 circulatory lag states driven by ``alpha`` (Wagner, 2 exponentials)
x3, x4 circulatory lag states driven by ``q/2``
x5, x6 impulsive (added-mass) deficiency states for normal force
x7, x8 impulsive deficiency states for pitching moment
====== ==================================================================

Circulatory formulation
-----------------------
The indicial (Wagner) lift response is approximated by

    phi(s) = 1 - A1*exp(-b1*beta^2*s) - A2*exp(-b2*beta^2*s),   A1 + A2 = 1

so that ``phi(0) = 0`` and ``phi(inf) = 1``. Writing the Duhamel integral with
deficiency states driven by ``alpha_dot`` and then substituting
``z = x - A*alpha`` removes the derivative from the input, which is what makes
``A`` and ``B`` *constant* with ``u = [alpha, q]`` exactly as Eq. (8) states:

    dz1/ds = -b1*beta^2 * (z1 + A1*alpha)
    alpha_E = -(z1 + z2 + z3 + z4)

Check: for constant ``alpha`` the states settle at ``z1 -> -A1*alpha``,
``z2 -> -A2*alpha``, giving ``alpha_E -> alpha``; for a step from zero the
states start at zero, giving ``alpha_E(0) = 0``. Both are the required
behaviour of ``phi``.

Impulsive formulation
---------------------
Standard compressible deficiency functions with time constant
``T_I = c/a``. In non-dimensional time ``s = V t / b`` that is ``T_I_s = 2*Ma``.
The ``1/Ma`` factors in the load expressions cancel the ``Ma`` in the time
constant, so the incompressible limit is finite:

    (4/Ma)*(alpha - x5) -> 8*K_alpha * d(alpha)/ds   as Ma -> 0

This is why the low-Mach case is handled by keeping ``Ma`` finite rather than
by dropping the impulsive terms.
"""

from __future__ import annotations

import numpy as np

from ..config import Airfoil

N_STATES = 8

#: Floor on the impulsive time constant, guarding the Ma -> 0 singularity.
_MIN_TI_S = 1e-6


def _beta_sq(mach: float) -> float:
    return max(1.0 - mach * mach, 1e-6)


def _ti_s(mach: float) -> float:
    """Impulsive time constant in semi-chords of travel: T_I * V/b = 2*Ma."""
    return max(2.0 * mach, _MIN_TI_S)


def derivatives(x: np.ndarray, alpha: float, q: float, af: Airfoil, mach: float) -> np.ndarray:
    """d(x1..x8)/ds -- rates with respect to NON-DIMENSIONAL time ``s``.

    The caller multiplies by ``V/b`` to get rates in real time.
    """
    bsq = _beta_sq(mach)
    ti = _ti_s(mach)
    dx = np.empty(N_STATES)

    # -- circulatory: alpha-driven (x1, x2) and q-driven (x3, x4) ----------
    dx[0] = -af.b1 * bsq * (x[0] + af.A1 * alpha)
    dx[1] = -af.b2 * bsq * (x[1] + af.A2 * alpha)
    # A pitch rate q about the elastic axis induces q/2 of incidence at the
    # 3/4-chord control point, which is what the circulatory response sees.
    dx[2] = -af.b1 * bsq * (x[2] + af.A1 * 0.5 * q)
    dx[3] = -af.b2 * bsq * (x[3] + af.A2 * 0.5 * q)

    # -- impulsive first-order lags ---------------------------------------
    dx[4] = (alpha - x[4]) / (af.K_alpha * ti)
    dx[5] = (q - x[5]) / (af.K_q * ti)
    dx[6] = (alpha - x[6]) / (af.K_M * ti)
    dx[7] = (q - x[7]) / (af.K_Mq * ti)

    return dx


def alpha_effective(x: np.ndarray) -> float:
    """Effective angle of attack, radians (paper's ``alpha_E``)."""
    return -(x[0] + x[1] + x[2] + x[3])


def circulatory_normal_force(x: np.ndarray, af: Airfoil) -> float:
    """``C_N^C`` -- attached circulatory normal force (Eq. 10)."""
    return af.c_n_alpha_rad * alpha_effective(x)


def impulsive_loads(
    x: np.ndarray, alpha: float, q: float, mach: float
) -> tuple[float, float]:
    """``(C_N^I, C_m^I)`` -- the added-mass contributions.

    These are the superscript-``I`` terms of Eqs. (5) and (6).
    """
    m = max(mach, 1e-6)
    c_n_i = (4.0 / m) * (alpha - x[4]) + (1.0 / m) * (q - x[5])
    c_m_i = -(1.0 / m) * (alpha - x[6]) - (7.0 / (12.0 * m)) * (q - x[7])
    return c_n_i, c_m_i


def potential_normal_force(
    x: np.ndarray, alpha: float, q: float, af: Airfoil, mach: float
) -> float:
    """``C_N^p = C_N^C + C_N^I`` (Eq. 10) -- drives the pressure lag x9."""
    c_n_i, _ = impulsive_loads(x, alpha, q, mach)
    return circulatory_normal_force(x, af) + c_n_i
