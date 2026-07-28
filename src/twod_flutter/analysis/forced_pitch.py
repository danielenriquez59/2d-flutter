"""Prescribed-motion airload validation (theory.md §10.1, §10.2).

Drives the aerodynamic model with a prescribed sinusoidal pitch

    alpha(t) = alpha_mean + alpha_amp * sin(omega * t),   omega = k * V / b

and records the airload hysteresis loops. This is the first validation gate:
the aeroelastic result in §10.3 is meaningless if these loops are wrong.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..aero import LBModel
from ..aero import attached
from ..config import Case
from ..integrators import get_integrator
from ..units import RAD2DEG, rad


@dataclass
class ForcedPitchResult:
    """Time history of a forced-pitch run. Angles in radians."""

    t: np.ndarray
    alpha: np.ndarray
    q: np.ndarray
    c_n: np.ndarray
    c_m: np.ndarray
    c_c: np.ndarray
    c_l: np.ndarray
    alpha_e: np.ndarray
    f_delayed: np.ndarray
    x9: np.ndarray
    x14: np.ndarray
    stalled: np.ndarray
    period: float

    @property
    def alpha_deg(self) -> np.ndarray:
        return self.alpha * RAD2DEG

    def last_cycle(self) -> "ForcedPitchResult":
        """Slice out the final period, once startup transients have decayed.

        Hysteresis loops are only meaningful after the response is periodic;
        comparing the first cycle to steady-state experimental data is a
        classic way to get a plausible-looking but wrong answer.
        """
        mask = self.t >= (self.t[-1] - self.period)
        return ForcedPitchResult(
            **{
                name: (getattr(self, name)[mask] if name != "period" else self.period)
                for name in self.__dataclass_fields__
            }
        )


def run(case: Case, integrator: str = "rk4") -> ForcedPitchResult:
    """Integrate a forced-pitch case."""
    if case.kind != "forced_pitch":
        raise ValueError(f"case {case.name!r} is kind {case.kind!r}, not forced_pitch")

    af, flow, fp = case.airfoil, case.flow, case.forced_pitch
    V, mach = flow.V, flow.mach

    # The chord for a prescribed-motion case comes from the structure block if
    # present, else from the case's own reference chord.
    chord = case.structure.chord if case.structure else case.reference.get("chord")
    if chord is None:
        raise ValueError(
            f"case {case.name!r}: needs a chord -- give a 'structure', or set "
            "'reference: {chord: ...}'"
        )
    b = 0.5 * chord

    omega = fp.k * V / b
    period = 2.0 * np.pi / omega
    dt = period / fp.steps_per_cycle
    n_steps = fp.n_cycles * fp.steps_per_cycle

    a_mean, a_amp = rad(fp.alpha_mean_deg), rad(fp.alpha_amp_deg)

    def alpha_of(t: float) -> float:
        return a_mean + a_amp * np.sin(omega * t)

    def alpha_dot_of(t: float) -> float:
        return a_amp * omega * np.cos(omega * t)

    model = LBModel(af, mach)
    step = get_integrator(integrator)

    # RK4 is explicit, and the non-circulatory moment states are fast enough to
    # set a hard step limit (see attached.time_constants). Failing here with a
    # number beats returning a silently diverged run.
    if integrator == "rk4":
        tau_min = attached.min_time_constant(af, mach)
        ds = (2.0 * np.pi / fp.k) / fp.steps_per_cycle  # step in semi-chords
        limit = 2.78 * tau_min  # RK4 real-axis stability boundary
        if ds > limit:
            needed = int(np.ceil((2.0 * np.pi / fp.k) / limit))
            raise ValueError(
                f"steps_per_cycle={fp.steps_per_cycle} is too coarse for rk4: "
                f"step {ds:.4f} exceeds the stability limit {limit:.4f} "
                f"semi-chords (fastest state tau = {tau_min:.4f}). "
                f"Use steps_per_cycle >= {needed}, or integrator='newmark'."
            )

    def rhs(t: float, x: np.ndarray) -> np.ndarray:
        alpha = alpha_of(t)
        alpha_dot = alpha_dot_of(t)
        q = alpha_dot * chord / V  # Eq. (25) with theta = alpha (no plunge)
        dalpha_ds = alpha_dot * b / V
        return model.derivatives(x, alpha, q, dalpha_ds) * (V / b)

    x = model.initial_state(alpha_of(0.0))
    was_stalled = False

    n = n_steps + 1
    out = {
        k: np.zeros(n)
        for k in ("t", "alpha", "q", "c_n", "c_m", "c_c", "c_l", "alpha_e",
                  "f_delayed", "x9", "x14")
    }
    stalled_hist = np.zeros(n, dtype=bool)

    t = 0.0
    for i in range(n):
        alpha = alpha_of(t)
        q = alpha_dot_of(t) * chord / V
        loads = model.airloads(x, alpha, q)

        out["t"][i] = t
        out["alpha"][i] = alpha
        out["q"][i] = q
        out["c_n"][i] = loads.c_n
        out["c_m"][i] = loads.c_m
        out["c_c"][i] = loads.c_c
        out["c_l"][i] = loads.c_l
        out["alpha_e"][i] = loads.alpha_e
        out["f_delayed"][i] = loads.f_delayed
        out["x9"][i] = x[8]
        out["x14"][i] = x[13]
        stalled_hist[i] = loads.stalled

        if i == n - 1:
            break
        x = step(rhs, t, x, dt)
        x, was_stalled = model.step_events(x, was_stalled)
        t += dt

    return ForcedPitchResult(stalled=stalled_hist, period=period, **out)
