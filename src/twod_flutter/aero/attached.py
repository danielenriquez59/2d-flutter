"""Unsteady attached flow -- states x1..x8 (paper Eq. 8).

    xdot_tilde = A x_tilde + B u,    u = [alpha, q]^T

The source paper defers ``A`` and ``B`` to Ref. [16]. They
are given in full by

    Leishman J.G., Nguyen K.Q., "State-Space Representation of Unsteady Airfoil
    Behavior," *AIAA Journal* **28**(5), 1990, 836-844

whose Eqs. (17)-(21) and Appendices A-C are implemented here. That paper's
state numbering ``x1..x8`` matches the source paper's exactly, and its ``A`` is
diagonal, which is why Eq. (8) can be written with constant matrices.

Layout (Leishman & Nguyen numbering)
------------------------------------
====== ===========================================================
x1, x2 circulatory, driven by ``alpha_3/4 = alpha + q/2``
x3     non-circulatory normal force due to ``alpha``
x4     non-circulatory normal force due to ``q``
x5, x6 non-circulatory moment due to ``alpha``
x7     circulatory moment due to ``q``
x8     non-circulatory moment due to ``q``
====== ===========================================================

State scaling
-------------
Leishman & Nguyen write Eq. (17) in real time with ``B = [1 1]^T`` carrying no
``2V/c`` factor, which leaves the raw states carrying units of time x angle. We
integrate the **scaled** states ``xhat = x * (2V/c)``, which makes every
equation below a pure function of non-dimensional time ``s = 2Vt/c = Vt/b``
with no residual velocity dependence. All rates returned are ``d/ds``.

Verification of the scaling, for constant ``alpha``:

    xhat1 -> alpha/(b1*beta^2)  =>  alpha_E = beta^2*(A1*b1*xhat1 + A2*b2*xhat2)
                                            -> (A1 + A2)*alpha = alpha

with ``A1 + A2 = 1``, and ``alpha_E(0) = 0`` from zero initial states -- the two
required limits of the indicial function ``phi``.
"""

from __future__ import annotations

import numpy as np

from ..config import Airfoil

N_STATES = 8

#: Leishman & Nguyen reduce the theoretical non-circulatory time constants by
#: 25% to correlate with test data (their closing remark, p. 844).
K_REDUCTION = 0.75

#: Non-circulatory moment indicial constants, their Appendix B.
A3, A4, B3, B4 = 1.5, -0.5, 0.25, 0.1
#: Circulatory pitch-rate moment pole, their Appendix C (A10).
B5 = 0.5


def beta(mach: float) -> float:
    """Prandtl-Glauert compressibility factor ``sqrt(1 - M^2)``."""
    return np.sqrt(max(1.0 - mach * mach, 1e-9))


def ti_s(mach: float) -> float:
    """Non-circulatory time constant in semi-chords: ``T_I * 2V/c = 2*Ma``.

    ``T_I = c/a``, so ``T_I * 2V/c = 2V/a = 2*Ma``.
    """
    return max(2.0 * mach, 1e-9)


def _k_constants(mach: float) -> tuple[float, float, float, float]:
    """``(K_alpha, K_q, K_alphaM, K_qM)`` -- Leishman & Nguyen Eqs. (A2), (A6), (A12).

    All four carry the empirical 25% reduction.
    """
    m, bta = mach, beta(mach)
    ab = A3 * B4 + A4 * B3  # reused in K_alphaM
    sum_ab = 0.3 * 0.14 + 0.7 * 0.53  # A1*b1 + A2*b2, circulatory constants

    k_alpha = 1.0 / ((1.0 - m) + np.pi * bta * m * m * sum_ab)
    k_q = 1.0 / ((1.0 - m) + 2.0 * np.pi * bta * m * m * sum_ab)
    k_am = ab / (B3 * B4 * (1.0 - m))
    k_qm = 7.0 / (15.0 * (1.0 - m) + 3.0 * np.pi * bta * m * m * B5)

    return (
        K_REDUCTION * k_alpha,
        K_REDUCTION * k_q,
        K_REDUCTION * k_am,
        K_REDUCTION * k_qm,
    )


def derivatives(
    x: np.ndarray, alpha: float, q: float, af: Airfoil, mach: float
) -> np.ndarray:
    """d(x1..x8)/ds for the scaled states."""
    bsq = beta(mach) ** 2
    ti = ti_s(mach)
    k_a, k_q, k_am, k_qm = _k_constants(mach)

    alpha_34 = alpha + 0.5 * q  # Leishman & Nguyen, below Eq. (18)

    dx = np.empty(N_STATES)
    # Circulatory (Eq. 17), driven by the 3/4-chord incidence.
    dx[0] = -af.b1 * bsq * x[0] + alpha_34
    dx[1] = -af.b2 * bsq * x[1] + alpha_34
    # Non-circulatory normal force (Eqs. 19, A3).
    dx[2] = alpha - x[2] / (k_a * ti)
    dx[3] = q - x[3] / (k_q * ti)
    # Non-circulatory moment due to alpha (Eqs. A7, A9).
    dx[4] = alpha - x[4] / (B3 * k_am * ti)
    dx[5] = alpha - x[5] / (B4 * k_am * ti)
    # Moment due to pitch rate: circulatory (A13) and non-circulatory (A14).
    dx[6] = q - B5 * bsq * x[6]
    dx[7] = q - x[7] / (k_qm * ti)
    return dx


def alpha_effective(x: np.ndarray, af: Airfoil, mach: float) -> float:
    """Effective angle of attack (the paper's ``alpha_E``), radians.

    From Leishman & Nguyen Eq. (18) with the lift-curve slope divided out:
    ``C_N^C = (2*pi/beta) * alpha_E``.
    """
    bsq = beta(mach) ** 2
    return bsq * (af.A1 * af.b1 * x[0] + af.A2 * af.b2 * x[1])


def circulatory_normal_force(x: np.ndarray, af: Airfoil, mach: float) -> float:
    """``C_N^C`` -- attached circulatory normal force.

    Uses the airfoil's measured ``C_Nalpha`` rather than the theoretical
    ``2*pi/beta``, since the source paper's separated-flow equations (its
    Eq. 12) are written against the measured value.
    """
    return af.c_n_alpha_rad * alpha_effective(x, af, mach)


def impulsive_loads(
    x: np.ndarray, alpha: float, q: float, af: Airfoil, mach: float
) -> tuple[float, float]:
    """``(C_N^I, C_m^I)`` -- the attached-flow loads carried by superscript I.

    Normal force from Eqs. (20) and (A4); moment from (A8), (A15), (A16) plus
    the circulatory moment arm ``(1/4 - x_ac) * C_N^C``.
    """
    m = max(mach, 1e-9)
    bta = beta(mach)
    ti = ti_s(mach)
    k_a, k_q, k_am, k_qm = _k_constants(mach)
    dx = derivatives(x, alpha, q, af, mach)

    # -- normal force: C_N^I = (4/M)*xdot3 + (1/M)*xdot4 ------------------
    c_n_i = (4.0 / m) * dx[2] + (1.0 / m) * dx[3]

    # -- moment due to alpha, Eq. (A8) ------------------------------------
    a55 = -1.0 / (B3 * k_am * ti)
    a66 = -1.0 / (B4 * k_am * ti)
    c_m_alpha = (-1.0 / m) * (A3 * a55 * x[4] + A4 * a66 * x[5]) - (1.0 / m) * alpha

    # -- moment due to q: circulatory (A15) + non-circulatory (A16) -------
    c_m_q_circ = -(np.pi / 8.0 / bta) * B5 * bta**2 * x[6]
    c_m_q_imp = -(7.0 / (12.0 * m)) * dx[7]

    # -- circulatory moment arm about the quarter chord --------------------
    c_m_circ = (0.25 - af.x_ac) * circulatory_normal_force(x, af, mach)

    return c_n_i, c_m_alpha + c_m_q_circ + c_m_q_imp + c_m_circ


def potential_normal_force(
    x: np.ndarray, alpha: float, q: float, af: Airfoil, mach: float
) -> float:
    """``C_N^p = C_N^C + C_N^I`` (paper Eq. 10) -- drives the pressure lag x9."""
    c_n_i, _ = impulsive_loads(x, alpha, q, af, mach)
    return circulatory_normal_force(x, af, mach) + c_n_i


def time_constants(af: Airfoil, mach: float) -> dict[str, float]:
    """Decay time constant of each attached state, in semi-chords.

    The non-circulatory moment states are two orders of magnitude faster than
    the circulatory ones (stiffness ratio ~350 at Ma = 0.12), which is the
    concrete reason the source paper reports explicit integration as unstable
    and adopts an implicit Newmark scheme (its Ref. [15]).
    """
    bsq = beta(mach) ** 2
    ti = ti_s(mach)
    k_a, k_q, k_am, k_qm = _k_constants(mach)
    return {
        "x1": 1.0 / (af.b1 * bsq),
        "x2": 1.0 / (af.b2 * bsq),
        "x3": k_a * ti,
        "x4": k_q * ti,
        "x5": B3 * k_am * ti,
        "x6": B4 * k_am * ti,
        "x7": 1.0 / (B5 * bsq),
        "x8": k_qm * ti,
    }


def min_time_constant(af: Airfoil, mach: float) -> float:
    """Fastest attached-flow time constant -- what sets the explicit step limit."""
    return min(time_constants(af, mach).values())


def steady_state(alpha0: float, af: Airfoil, mach: float) -> np.ndarray:
    """States consistent with a constant ``alpha0`` and zero pitch rate."""
    bsq = beta(mach) ** 2
    ti = ti_s(mach)
    k_a, k_q, k_am, k_qm = _k_constants(mach)
    x = np.zeros(N_STATES)
    x[0] = alpha0 / (af.b1 * bsq)
    x[1] = alpha0 / (af.b2 * bsq)
    x[2] = alpha0 * k_a * ti
    x[3] = 0.0
    x[4] = alpha0 * B3 * k_am * ti
    x[5] = alpha0 * B4 * k_am * ti
    x[6] = 0.0
    x[7] = 0.0
    return x
