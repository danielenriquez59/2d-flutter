"""2-DOF pitch/plunge structural model -- states x15..x18 (paper Eqs. 21-25).

    m*h_ddot + S*theta_ddot + K_h*h     = Q_h        (21)
    S*h_ddot + I_t*theta_ddot + K_t*theta = Q_theta  (22)

State ordering follows the paper exactly -- displacements first, then rates:

    [x15 x16 x17 x18]^T = [h  theta  h_dot  theta_dot]^T   (23)

so ``alpha = x16 + x17/V`` and ``q = x18*c/V`` (Eq. 25).

Sign conventions, read off Eqs. (21)-(25) and Fig. 2:

* ``h`` is positive **downward** -- Eq. (25) has ``alpha = theta + h_dot/V``, so
  descending increases incidence, and Eq. (24) has ``Q_h = -0.5*rho*V^2*c*C_L``,
  so positive lift drives ``h`` negative (upward). Both only make sense with
  ``h`` down, which is Fung's convention.
* ``Q_theta = +0.5*rho*V^2*c^2*C_m``, nose-up positive.

The paper reports **no structural damping**; ``C_h`` and ``C_theta`` default to
zero and every dissipative mechanism in the model is aerodynamic (theory.md
§15.7).
"""

from __future__ import annotations

import numpy as np

from .config import Structure

N_STRUCT = 4


class StructuralModel:
    """Linear 2-DOF section on bending and torsional springs."""

    def __init__(self, struct: Structure):
        self.st = struct
        m, s, i_t = struct.mass, struct.static_moment, struct.I_theta

        self.M = np.array([[m, s], [s, i_t]], dtype=float)
        det = m * i_t - s * s
        if det <= 0.0:
            raise ValueError(
                f"structure {struct.name!r}: mass matrix is not positive definite "
                f"(m*I_theta - S^2 = {det:.4g}). Check mass, I_theta and x_theta."
            )
        self.M_inv = np.linalg.inv(self.M)
        self.K = np.diag([struct.K_h, struct.K_theta])
        self.C = np.diag([struct.C_h, struct.C_theta])

        # xdot = A_s x + B_s u, with u = [0, 0, Q_h, Q_theta] (paper Eq. 24).
        self.A_s = np.zeros((4, 4))
        self.A_s[0, 2] = 1.0
        self.A_s[1, 3] = 1.0
        self.A_s[2:, :2] = -self.M_inv @ self.K
        self.A_s[2:, 2:] = -self.M_inv @ self.C

    # -- modal properties, for sanity checks ------------------------------
    @property
    def uncoupled_frequencies(self) -> tuple[float, float]:
        """``(omega_h, omega_theta)`` in rad/s, ignoring inertial coupling."""
        return (
            np.sqrt(self.st.K_h / self.st.mass),
            np.sqrt(self.st.K_theta / self.st.I_theta),
        )

    def coupled_frequencies(self) -> np.ndarray:
        """In-vacuo natural frequencies of the coupled system, rad/s."""
        w2 = np.linalg.eigvals(self.M_inv @ self.K)
        return np.sqrt(np.sort(np.abs(w2)))

    # -- dynamics ---------------------------------------------------------
    @property
    def ac_to_ea(self) -> float:
        """Distance from the aerodynamic centre to the elastic axis, metres.

        Positive when the elastic axis lies **aft** of the quarter chord. With
        ``x_ea = b(1 + a_h)`` and the aerodynamic centre at ``c/4 = b/2``:

            d = b*(a_h + 0.5)

        so ``a_h = -0.5`` puts the two coincident and ``d = 0``.
        """
        return self.st.b * (self.st.a_h + 0.5)

    def divergence_velocity(self, c_n_alpha_rad: float, rho: float) -> float:
        """Static divergence speed from linear attached-flow theory, m/s.

        Balance of aerodynamic and elastic pitching moment:
        ``K_theta = 0.5*rho*V^2*c*C_Nalpha*d``. Infinite when ``d <= 0``, i.e.
        when the elastic axis is at or ahead of the aerodynamic centre.
        """
        d = self.ac_to_ea
        if d <= 0.0:
            return float("inf")
        return float(
            np.sqrt(self.st.K_theta / (0.5 * rho * self.st.chord * c_n_alpha_rad * d))
        )

    def generalised_forces(
        self, c_l: float, c_m: float, rho: float, V: float, chord: float
    ) -> tuple[float, float]:
        """``(Q_h, Q_theta)`` from the airload coefficients, paper Eq. (24).

        Eq. (24) gives ``Q_theta = 0.5*rho*V^2*c^2*C_m`` alone, with ``C_m``
        taken about the quarter chord. That is only complete if the elastic axis
        sits **at** the quarter chord -- and if it does, the lift has no moment
        arm about the elastic axis and the section can never diverge.

        The paper nonetheless quotes a static divergence velocity of 17.7 m/s
        from its Ref. [3], so the elastic axis must be aft of the quarter chord
        and Eq. (24) is missing the transfer term. We add it:

            Q_theta = 0.5*rho*V^2*c^2*C_m + 0.5*rho*V^2*c*C_L*d

        With ``a_h = -0.5`` the term vanishes and Eq. (24) is recovered exactly,
        so this is a strict generalisation. See theory.md §10.3.
        """
        qbar = 0.5 * rho * V * V
        lift = qbar * chord * c_l
        q_theta = qbar * chord * chord * c_m + lift * self.ac_to_ea
        return -lift, q_theta

    def derivatives(
        self, x: np.ndarray, q_h: float, q_theta: float
    ) -> np.ndarray:
        """d(x15..x18)/dt -- rates in REAL time, unlike the aerodynamic states."""
        dx = np.empty(N_STRUCT)
        dx[0] = x[2]
        dx[1] = x[3]
        dx[2:] = self.A_s[2:, :] @ x + self.M_inv @ np.array([q_h, q_theta])
        return dx

    def alpha_and_q(self, x: np.ndarray, V: float, chord: float) -> tuple[float, float]:
        """``(alpha, q)`` from the structural state, paper Eq. (25)."""
        return x[1] + x[2] / V, x[3] * chord / V
