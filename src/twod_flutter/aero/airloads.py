"""Airload assembly -- paper Eqs. 2-7.

    C_N = C_N^I + C_N^f + C_N^v          (5)
    C_m = C_m^I + C_m^f + C_m^v + C_m0   (6)
    C_C = C_C^f                          (7)

Note how the circulatory normal force does *not* appear as its own term: it
enters through ``C_N^f``, which is the circulatory load with the Kirchhoff
separation factor applied. Superscript ``I`` is therefore impulsive only. This
is consistent with Eq. (10), where ``C_N^p = C_N^C + C_N^I`` is the fully
attached potential load used to drive the pressure lag.

The paper gives the separated-flow loads only as "the linear function of
unsteady trailing edge separation point x10". The explicit forms below are from
Chantharasenawong (2007) Eqs. (2.7)-(2.9). See ``docs/theory.md`` §9.2.
"""

from __future__ import annotations

import numpy as np

from ..config import Airfoil
from .dynamic_stall import kirchhoff_kn


def separated_normal_force(alpha_e: float, f: float, af: Airfoil) -> float:
    """``C_N^f`` -- circulatory normal force reduced by separation.

        C_N^f = C_Nalpha * ((1 + sqrt(f))/2)^2 * alpha_E
    """
    return af.c_n_alpha_rad * kirchhoff_kn(f) * alpha_e


def separated_moment(c_n_f: float, f: float, af: Airfoil) -> float:
    """``C_m^f`` -- centre-of-pressure travel as the flow separates (Beddoes).

        C_m^f = (K0 + K1*(1 - f) + K2*sin(pi * f^m)) * C_N^f
    """
    shape = af.K0 + af.K1 * (1.0 - f) + af.K2 * np.sin(np.pi * f**af.m_exp)
    return shape * c_n_f


def chord_force(alpha_e: float, f: float, af: Airfoil) -> float:
    """``C_C^f`` -- chord force, Eq. (7).

        C_C = eta * C_Nalpha * alpha_E^2 * sqrt(f)
    """
    return af.eta * af.c_n_alpha_rad * alpha_e**2 * np.sqrt(max(f, 0.0))


def vortex_moment(c_n_v: float, tau: float, af: Airfoil) -> float:
    """``C_m^v`` -- moment from the vortex traversing the chord.

    Canonical Leishman-Beddoes centre-of-pressure travel; the paper's Eq. (20)
    increment is applied separately on top of this.
    """
    return -0.25 * (1.0 - np.cos(np.pi * min(tau / af.T_vl, 1.0))) * c_n_v


def lift_and_drag(
    c_n: float, c_c: float, alpha_e: float, alpha: float
) -> tuple[float, float]:
    """``(C_L, C_D)`` from normal and chord force, Eqs. (3) and (4).

    .. note::
       Eq. (3) uses ``alpha_E`` but Eq. (4) uses ``alpha``. Transcribed
       verbatim (theory.md §14, defect 1); almost certainly a typo, but it
       touches only ``C_D``, which the aeroelastic coupling does not use.
    """
    c_l = c_n * np.cos(alpha_e) + c_c * np.sin(alpha_e)
    c_d = c_n * np.sin(alpha) - c_c * np.cos(alpha)
    return c_l, c_d
