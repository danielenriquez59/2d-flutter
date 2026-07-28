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


def alpha1n(x10: float, af: Airfoil) -> float:
    """Effective break angle for the separation correlation, in degrees.

    **GAP-2.** The paper says only that ``alpha_1n`` is "the function of x10"
    and never defines it. In Beddoes' model the break angle droops once the
    flow separates, delaying reattachment. Default ``alpha1n_droop = 0``
    reduces this to the constant ``alpha_1``, which is the assumption-free
    choice; set it non-zero to explore the effect.
    """
    if af.alpha1n_droop == 0.0:
        return af.alpha1_deg
    f = min(max(x10, 0.0), 1.0)
    return af.alpha1_deg * (1.0 - af.alpha1n_droop * (1.0 - f) ** 0.25)


def sigma1(alpha: float, alpha_dot: float, stalled: bool, af: Airfoil) -> float:
    """Multiplier on the separation time constant ``T_f`` (Eq. 12).

    **GAP-2.** Undefined in the paper. In Beddoes' formulation this is a
    *switch*, not a constant: separation and reattachment proceed at different
    rates, and the rate changes again once stalled. Defaults are all 1.0, which
    disables the switching.
    """
    if stalled:
        return af.sigma1_stalled
    # alpha * alpha_dot < 0 means the section is unloading -> reattaching.
    if alpha * alpha_dot < 0.0:
        return af.sigma1_reattach
    return af.sigma1_base


def pressure_lag_derivative(x9: float, c_n_potential: float, af: Airfoil) -> float:
    """d(x9)/ds -- leading-edge pressure lag, Eq. (9).

    ``x9`` is the paper's ``C_N'``: a first-order lag of the potential-flow
    normal force with time constant ``T_p``.
    """
    return (c_n_potential - x9) / af.T_p


def separation_point_derivative(
    x9: float, x10: float, sigma_1: float, af: Airfoil
) -> float:
    """d(x10)/ds -- delayed separation point, Eq. (12).

    ``x9 / C_Nalpha`` converts the lagged normal force back into an equivalent
    angle before it enters the Kirchhoff correlation.
    """
    alpha_lagged = x9 / af.c_n_alpha_rad
    f_target = kirchhoff_f(alpha_lagged, alpha1n(x10, af), af)
    return (f_target - x10) / (sigma_1 * af.T_f)


def reattachment_derivative(x13: float, alpha: float, af: Airfoil) -> float:
    """d(x13)/ds -- separation point during reattachment, Eq. (16).

    .. note::
       Eq. (16) as printed lacks the ``V/b`` factor carried by every other rate
       equation in the paper, making it dimensionally inconsistent
       (theory.md §14, defect 2). We implement it *with* the factor -- i.e. this
       function returns a d/ds rate like its siblings -- and flag the deviation.
    """
    f_target = kirchhoff_f(alpha, alpha1n(x13, af), af)
    return (f_target - x13) / (0.5 * af.T_f)
