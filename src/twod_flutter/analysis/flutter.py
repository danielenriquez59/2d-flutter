"""Aeroelastic time marching and LCO extraction (theory.md §10.3).

Integrates the 18-state coupled system at a fixed freestream velocity, discards
the startup transient, and characterises whatever the response settles onto:
decay to equilibrium, a symmetric limit cycle, or an asymmetric one.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..aero import attached
from ..config import Case
from ..integrators import get_integrator
from ..system import AeroelasticSystem, I_THETA, N_AERO
from ..units import RAD2DEG

#: Kirchhoff reconstruction is only valid to about 60 deg (theory.md §15.5).
#: A growing LCO can leave that range silently, so runs are flagged.
ALPHA_VALID_DEG = 60.0


@dataclass
class FlutterResult:
    """Outcome of one fixed-velocity run."""

    V: float
    t: np.ndarray
    theta: np.ndarray
    h: np.ndarray
    alpha: np.ndarray
    diverged: bool = False
    out_of_range: bool = False
    peaks: np.ndarray = field(default_factory=lambda: np.zeros(0))
    troughs: np.ndarray = field(default_factory=lambda: np.zeros(0))

    # -- classification ---------------------------------------------------
    @property
    def amplitude(self) -> float:
        """Peak-to-peak pitch amplitude over the settled window, radians."""
        if self.peaks.size == 0 or self.troughs.size == 0:
            return 0.0
        return float(np.mean(self.peaks) - np.mean(self.troughs))

    @property
    def offset(self) -> float:
        """Mean of the LCO extrema -- nonzero means an asymmetric cycle."""
        if self.peaks.size == 0 or self.troughs.size == 0:
            return float(np.mean(self.theta)) if self.theta.size else 0.0
        return 0.5 * float(np.mean(self.peaks) + np.mean(self.troughs))

    def classify(self, decay_threshold_deg: float = 0.5) -> str:
        """``"decayed" | "symmetric_lco" | "asymmetric_lco" | "diverged"``.

        Asymmetry is judged by comparing the offset of the cycle against its
        own amplitude, so the test is scale-free: a cycle whose centre sits more
        than 10% of its half-amplitude away from zero is asymmetric.
        """
        if self.diverged:
            return "diverged"
        amp_deg = self.amplitude * RAD2DEG
        if amp_deg < decay_threshold_deg:
            return "decayed"
        if abs(self.offset) > 0.10 * (0.5 * self.amplitude):
            return "asymmetric_lco"
        return "symmetric_lco"


def _extrema(t: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Interior local maxima and minima of ``y``."""
    if y.size < 3:
        return np.zeros(0), np.zeros(0)
    up = (y[1:-1] > y[:-2]) & (y[1:-1] >= y[2:])
    dn = (y[1:-1] < y[:-2]) & (y[1:-1] <= y[2:])
    return y[1:-1][up], y[1:-1][dn]


def run(
    case: Case,
    V: float,
    *,
    theta0_deg: float = 1.0,
    h0: float = 0.0,
    t_end: float | None = None,
    settle_fraction: float = 0.6,
    steps_per_period: int = 4000,
    integrator: str = "rk4",
    store_every: int = 10,
) -> FlutterResult:
    """March the coupled system at one velocity.

    ``t_end`` defaults to 40 torsional periods, which is ample for the transient
    to decay at these damping levels. ``settle_fraction`` is the portion of the
    record discarded before extrema are collected.
    """
    case.flow.V = V
    case.flow.mach = V / case.flow.sound_speed
    sys_ = AeroelasticSystem(case)

    _, omega_t = sys_.struct.uncoupled_frequencies
    period = 2.0 * np.pi / omega_t
    if t_end is None:
        t_end = 40.0 * period

    dt = period / steps_per_period
    # Explicit RK4 is bounded by the fastest aerodynamic state (theory.md §9.1).
    if integrator == "rk4":
        limit = 2.78 * attached.min_time_constant(case.airfoil, case.flow.mach)
        ds = dt * V / sys_.b
        if ds > limit:
            dt = 0.95 * limit * sys_.b / V
    n_steps = int(np.ceil(t_end / dt))

    step = get_integrator(integrator)
    x = sys_.initial_state(theta0=np.radians(theta0_deg), h0=h0)
    was_stalled = False

    n_store = n_steps // store_every + 1
    t_h = np.zeros(n_store); th_h = np.zeros(n_store)
    h_h = np.zeros(n_store); al_h = np.zeros(n_store)

    t = 0.0
    j = 0
    diverged = False
    out_of_range = False
    for i in range(n_steps):
        if i % store_every == 0 and j < n_store:
            st = sys_.decode(x)
            t_h[j] = t; th_h[j] = st.theta; h_h[j] = st.h; al_h[j] = st.alpha
            if abs(st.alpha) * RAD2DEG > ALPHA_VALID_DEG:
                out_of_range = True
            j += 1
        x = step(sys_.derivatives, t, x, dt)
        x, was_stalled = sys_.step_events(x, was_stalled)
        t += dt
        if not np.all(np.isfinite(x)) or abs(x[I_THETA]) > 10.0:
            diverged = True
            break

    t_h, th_h, h_h, al_h = t_h[:j], th_h[:j], h_h[:j], al_h[:j]

    res = FlutterResult(
        V=V, t=t_h, theta=th_h, h=h_h, alpha=al_h,
        diverged=diverged, out_of_range=out_of_range,
    )
    if not diverged and t_h.size > 10:
        mask = t_h >= settle_fraction * t_h[-1]
        pk, tr = _extrema(t_h[mask], th_h[mask])
        res.peaks, res.troughs = pk, tr
    return res
