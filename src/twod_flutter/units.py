"""Angle-unit discipline.

The single largest correctness risk in this model (see ``docs/theory.md`` §2).
The source paper mixes conventions: ``C_Nalpha = 0.108`` is **per degree**
(2*pi/rad ~= 0.1097/deg) and ``alpha_1 = 15.25``, ``S_1 = 3.0``, ``S_2 = 2.3``
in Eq. (11) are **degrees**, while the structural equations are naturally in
radians.

Convention adopted by this package:

    Everything internal is in RADIANS.

The only exception is :func:`~twod_flutter.aero.separation.kirchhoff_f`, whose
empirical correlation is inherently in degrees; it converts on entry and is the
only place degrees appear in a formula.
"""

from __future__ import annotations

import numpy as np

DEG2RAD = np.pi / 180.0
RAD2DEG = 180.0 / np.pi


def deg(x_rad: float | np.ndarray) -> float | np.ndarray:
    """Radians -> degrees."""
    return x_rad * RAD2DEG


def rad(x_deg: float | np.ndarray) -> float | np.ndarray:
    """Degrees -> radians."""
    return x_deg * DEG2RAD


def cn_alpha_per_rad(cn_alpha_per_deg: float) -> float:
    """Convert a lift-curve slope from per-degree to per-radian.

    The paper's ``C_Nalpha = 0.108`` /deg becomes ~6.188 /rad, close to the
    thin-airfoil value 2*pi ~= 6.283. A converted value far from 2*pi is a
    strong signal that the input was already in per-radian.
    """
    return cn_alpha_per_deg * RAD2DEG
