"""Trailing-edge separation -- states x9, x10, x13 (paper Eqs. 9-12, 16)."""

from __future__ import annotations

import numpy as np

from ..config import Airfoil
from ..units import RAD2DEG


def kirchhoff_f(alpha_rad: float, alpha1_deg: float, af: Airfoil) -> float:
    """Static trailing-edge separation point, Eq. (11).

    Returns the separation point as a fraction of chord: 1 = fully attached,
    ~0.04 = fully separated.

    This is the one function in the package that works in **degrees** -- the
    correlation constants ``alpha1 = 15.25``, ``S1 = 3.0``, ``S2 = 2.3`` are
    degrees (theory.md §2). The input is radians and is converted here.

    The correlation is written in ``|alpha|``, so it is even. That evenness is
    precisely why the aerodynamics alone cannot break symmetry, and why the
    paper needs ``C_m0`` to do it (theory.md §10.3).
    """
    a = abs(alpha_rad) * RAD2DEG
    if a <= alpha1_deg:
        return 1.0 - 0.3 * np.exp((a - alpha1_deg) / af.S1_deg)
    # Clipped: for very large |alpha| the exponent underflows to 0.04 anyway,
    # but the raw expression can overflow on transients.
    return 0.04 + 0.66 * np.exp(max((alpha1_deg - a) / af.S2_deg, -50.0))


def alpha1n(x10: float, loading: float, af: Airfoil) -> float:
    """Effective break angle for the separation correlation, in degrees.

    **GAP-2: CLOSED** by Chantharasenawong (2007) Eq. (2.54):

        alpha_1 = alpha_10                                  if alpha*alpha' >= 0
        alpha_1 = alpha_10 - (1 - x10)^0.25 * delta_alpha1   if alpha*alpha' < 0

    The break angle droops during unloading because reattachment occurs at a
    lower incidence than separation did -- the model's hysteresis in ``f``.

    ``loading`` is ``alpha * d(alpha)/ds``: positive when the section is loading
    up, negative when unloading.
    """
    if loading >= 0.0:
        return af.alpha1_deg
    f = min(max(x10, 0.0), 1.0)
    return af.alpha1_deg - (1.0 - f) ** 0.25 * af.delta_alpha1_deg


def sigma1(
    loading: float, x10: float, tau_v: float, shedding: bool, af: Airfoil
) -> float:
    """Multiplier on the separation time constant ``T_f`` (paper Eq. 12).

    **GAP-2: CLOSED** by Chantharasenawong (2007) Table 2.3 and Eq. (2.55).
    ``T_f`` is a switch on flow state, not a constant -- the separation point
    moves at different speeds depending on where the shed vortex is and whether
    the section is loading or unloading.

    Vortex shedding phase (Table 2.3), by vortex time ``tau_v``:

    ================  =================  ====================  ==============
    condition         0 <= tau_v <= T_vl  T_vl < tau_v <= 2T_vl  2T_vl < tau_v
    ================  =================  ====================  ==============
    ``a*a' >= 0``     ``T_f0``           ``T_f0 / 3``          ``4 T_f0``
    ``a*a' <  0``     ``T_f0 / 2``       ``T_f0 / 2``          ``4 T_f0``
    ================  =================  ====================  ==============

    Reattachment phase (Eq. 2.55): ``T_f0`` if ``x10 >= 0.7``, else ``T_f0/2``.
    """
    if shedding:
        if tau_v > 2.0 * af.T_vl:
            return 4.0
        if tau_v > af.T_vl:
            return 1.0 / 3.0 if loading >= 0.0 else 0.5
        return 1.0 if loading >= 0.0 else 0.5
    return 1.0 if x10 >= 0.7 else 0.5


def pressure_lag_derivative(x9: float, c_n_potential: float, af: Airfoil) -> float:
    """d(x9)/ds -- leading-edge pressure lag, Eq. (9).

    ``x9`` is the paper's ``C_N'``: a first-order lag of the potential-flow
    normal force with time constant ``T_p``.
    """
    return (c_n_potential - x9) / af.T_p


def separation_point_derivative(
    x9: float, x10: float, sigma_1: float, loading: float, af: Airfoil
) -> float:
    """d(x10)/ds -- delayed separation point, Eq. (12).

    ``x9 / C_Nalpha`` converts the lagged normal force back into the equivalent
    angle of incidence before it enters the Kirchhoff correlation. Note this is
    *not* the effective angle ``alpha_E``; it is the angle that would produce
    the lagged normal force under static conditions.
    """
    alpha_lagged = x9 / af.c_n_alpha_rad
    f_target = kirchhoff_f(alpha_lagged, alpha1n(x10, loading, af), af)
    return (f_target - x10) / (sigma_1 * af.T_f)


def reattachment_derivative(
    x13: float, alpha: float, loading: float, af: Airfoil
) -> float:
    """d(x13)/ds -- separation point used for the pitching moment, Eq. (16).

    Chantharasenawong (2007) Eq. (2.53) is the same equation, and explains why
    it exists: the dynamic separation point ``x10`` fails to predict the
    pitching moment during reattachment from deep stall, so the moment uses its
    own, faster-responding separation point. Note the time constant is the
    *base* ``T_f0``, not the switched ``T_f``.

    .. note::
       Eq. (16) as printed lacks the ``V/b`` factor carried by every other rate
       equation in the paper, making it dimensionally inconsistent
       (theory.md §14, defect 2). We implement it *with* the factor -- i.e. this
       function returns a d/ds rate like its siblings -- and flag the deviation.
    """
    f_target = kirchhoff_f(alpha, alpha1n(x13, loading, af), af)
    return (f_target - x13) / (0.5 * af.T_f)
