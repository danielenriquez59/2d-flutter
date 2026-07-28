"""The full 18-state aeroelastic system (theory.md §8).

Couples the 14-state modified Leishman-Beddoes model to the 4-state structural
model. The loop closes through paper Eq. (25):

    structure -> (alpha, q) -> L-B model -> (C_L, C_m) -> structure

State vector (zero-based; paper numbering x1..x18):

    [0:14]  aerodynamic states x1..x14
    [14:18] structural states  x15..x18 = [h, theta, h_dot, theta_dot]

**Rates are in real time.** The aerodynamic sub-model returns ``d/ds``, which is
converted here by the ``V/b`` factor the paper carries explicitly in Eqs. (9),
(12), (15) and (17).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .aero import LBModel, N_AERO
from .aero.model import Airloads
from .config import Case
from .structure import N_STRUCT, StructuralModel

N_TOTAL = N_AERO + N_STRUCT  # 18
I_H, I_THETA, I_HDOT, I_THETADOT = 14, 15, 16, 17


@dataclass
class AeroelasticState:
    """Decoded snapshot of the coupled system, for diagnostics."""

    h: float
    theta: float
    h_dot: float
    theta_dot: float
    alpha: float
    q: float
    loads: Airloads


class AeroelasticSystem:
    """18-state coupled model at one freestream velocity."""

    def __init__(self, case: Case):
        if case.structure is None:
            raise ValueError(f"case {case.name!r} has no structure block")
        self.case = case
        self.aero = LBModel(case.airfoil, case.flow.mach)
        self.struct = StructuralModel(case.structure)
        self.V = case.flow.V
        self.rho = case.flow.rho
        self.chord = case.structure.chord
        self.b = 0.5 * self.chord

    # -- initial condition ------------------------------------------------
    def initial_state(
        self, theta0: float = 0.0, h0: float = 0.0,
        theta_dot0: float = 0.0, h_dot0: float = 0.0,
    ) -> np.ndarray:
        """Aerodynamics converged to the initial incidence; structure as given."""
        x = np.zeros(N_TOTAL)
        alpha0 = theta0 + h_dot0 / self.V
        x[:N_AERO] = self.aero.initial_state(alpha0)
        x[I_H], x[I_THETA] = h0, theta0
        x[I_HDOT], x[I_THETADOT] = h_dot0, theta_dot0
        return x

    # -- dynamics ---------------------------------------------------------
    def decode(self, x: np.ndarray) -> AeroelasticState:
        """Airloads and kinematics at the current state."""
        alpha, q = self.struct.alpha_and_q(x[N_AERO:], self.V, self.chord)
        loads = self.aero.airloads(x[:N_AERO], alpha, q)
        return AeroelasticState(
            h=x[I_H], theta=x[I_THETA], h_dot=x[I_HDOT], theta_dot=x[I_THETADOT],
            alpha=alpha, q=q, loads=loads,
        )

    def derivatives(self, t: float, x: np.ndarray) -> np.ndarray:
        """d(x)/dt for all 18 states."""
        xs = x[N_AERO:]
        alpha, q = self.struct.alpha_and_q(xs, self.V, self.chord)
        loads = self.aero.airloads(x[:N_AERO], alpha, q)

        # -- structure ----------------------------------------------------
        q_h, q_t = self.struct.generalised_forces(
            loads.c_l, loads.c_m, self.rho, self.V, self.chord
        )
        dxs = self.struct.derivatives(xs, q_h, q_t)

        # -- aerodynamics -------------------------------------------------
        # The sigma switch tables key on alpha*d(alpha)/ds. From Eq. (25),
        # alpha = theta + h_dot/V, so d(alpha)/dt = theta_dot + h_ddot/V --
        # which needs the structural acceleration just computed. No circularity:
        # the airloads depend on the state, not on d(alpha)/ds.
        alpha_dot = dxs[1] + dxs[2] / self.V
        dalpha_ds = alpha_dot * self.b / self.V

        dx = np.empty(N_TOTAL)
        dx[:N_AERO] = self.aero.derivatives(
            x[:N_AERO], alpha, q, dalpha_ds
        ) * (self.V / self.b)
        dx[N_AERO:] = dxs
        return dx

    def step_events(self, x: np.ndarray, was_stalled: bool):
        """Forward the aerodynamic vortex-clock reset (theory.md §9.2)."""
        xa, stalled = self.aero.step_events(x[:N_AERO], was_stalled)
        if xa is not x[:N_AERO]:
            x = x.copy()
            x[:N_AERO] = xa
        return x, stalled
