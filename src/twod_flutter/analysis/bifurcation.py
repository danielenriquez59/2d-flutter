"""Velocity sweep and bifurcation diagram (theory.md §10.3).

Reproduces the paper's Figs. 7-9 and Table 1: sweep freestream velocity,
classify the settled response at each point, and record LCO extrema.

Two things make this more than a loop over :func:`flutter.run`:

* **The bifurcation is subcritical.** A stable equilibrium coexists with a
  finite-amplitude limit cycle, so a small initial disturbance decays at
  velocities where an LCO exists. Sweeping from one initial condition finds only
  one branch. Each velocity is therefore run from several disturbance
  amplitudes.
* **Attractors coexist.** At V = 20 m/s with C_m0 = 0.010 the paper reports
  symmetric *and* asymmetric LCOs at the same velocity. Reporting a single
  outcome per velocity would hide that, so every branch found is kept.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..config import Case
from ..units import RAD2DEG
from . import flutter


@dataclass
class Branch:
    """One settled response found at one velocity."""

    V: float
    theta0_deg: float
    kind: str
    amplitude_deg: float
    offset_deg: float
    peaks_deg: np.ndarray = field(default_factory=lambda: np.zeros(0))
    out_of_range: bool = False


@dataclass
class SweepResult:
    branches: list[Branch]

    def at(self, V: float, tol: float = 1e-9) -> list[Branch]:
        return [b for b in self.branches if abs(b.V - V) < tol]

    @property
    def velocities(self) -> np.ndarray:
        return np.array(sorted({b.V for b in self.branches}))

    def onset(self, kind: str) -> float | None:
        """Lowest velocity at which any branch of ``kind`` appears."""
        vs = [b.V for b in self.branches if b.kind == kind]
        return min(vs) if vs else None

    def summary_rows(self) -> list[tuple[float, str, float, float]]:
        """One row per velocity: the dominant branch, largest amplitude first."""
        rows = []
        for V in self.velocities:
            bs = sorted(self.at(V), key=lambda b: -b.amplitude_deg)
            best = bs[0]
            kinds = {b.kind for b in bs}
            kind = best.kind
            if "asymmetric_lco" in kinds and "symmetric_lco" in kinds:
                kind = "coexist(sym+asym)"
            rows.append((V, kind, best.amplitude_deg, best.offset_deg))
        return rows


def sweep(
    case: Case,
    velocities,
    *,
    disturbances_deg=(1.0, 8.0, 15.0),
    t_end: float = 25.0,
    steps_per_period: int = 4000,
    progress: bool = False,
) -> SweepResult:
    """Run the velocity sweep, several disturbance amplitudes at each point."""
    branches: list[Branch] = []
    for V in velocities:
        for th0 in disturbances_deg:
            r = flutter.run(
                case, float(V), theta0_deg=th0,
                t_end=t_end, steps_per_period=steps_per_period,
            )
            kind = r.classify()
            branches.append(
                Branch(
                    V=float(V), theta0_deg=th0, kind=kind,
                    amplitude_deg=r.amplitude * RAD2DEG,
                    offset_deg=r.offset * RAD2DEG,
                    peaks_deg=r.peaks * RAD2DEG,
                    out_of_range=r.out_of_range,
                )
            )
            if progress:
                print(
                    f"  V={V:5.2f} theta0={th0:5.1f} -> {kind:16s} "
                    f"amp={branches[-1].amplitude_deg:7.3f} "
                    f"off={branches[-1].offset_deg:+7.3f}",
                    flush=True,
                )
    return SweepResult(branches)


#: Paper Table 1 -- the target. Velocities in m/s.
PAPER_TABLE_1 = {
    0.0:   {"decay_below": 12.0, "hopf": 12.0, "sym_lco": 12.0, "asym_lco": 26.0},
    0.005: {"decay_below": 12.0, "hopf": 12.0, "sym_lco": 12.0, "asym_lco": 20.0},
    0.010: {"decay_below": 12.0, "hopf": 12.0, "sym_lco": 12.0, "asym_lco": 18.4},
}
#: Experiment, Ref. [3], as quoted by the paper.
EXPERIMENT = {
    "decay_below": 12.0, "hopf": (12.0, 13.0), "sym_lco": (12.0, 21.4),
    "static_divergence": 17.7,
}


def compare_to_table_1(result: SweepResult, c_m0: float) -> str:
    """Format the sweep against the paper's Table 1 for that ``C_m0``."""
    target = PAPER_TABLE_1.get(round(c_m0, 4))
    lines = [f"Bifurcation summary (C_m0 = {c_m0})", ""]
    lines.append(f"  {'V':>6}  {'behaviour':<20} {'amp/deg':>8} {'offset/deg':>10}")
    for V, kind, amp, off in result.summary_rows():
        lines.append(f"  {V:6.2f}  {kind:<20} {amp:8.3f} {off:+10.3f}")

    sym = result.onset("symmetric_lco")
    asym = result.onset("asymmetric_lco")
    lines += ["", "  onset velocities (m/s):"]
    lines.append(f"    symmetric LCO   model {sym if sym else '--'}"
                 + (f"   paper {target['sym_lco']}" if target else ""))
    lines.append(f"    asymmetric LCO  model {asym if asym else '--'}"
                 + (f"   paper {target['asym_lco']}" if target else ""))
    return "\n".join(lines)
