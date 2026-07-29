"""YAML-backed model definitions.

Three kinds of input file live under ``models/``:

``models/airfoils/*.yaml``
    Aerodynamic identity of a section: the Leishman-Beddoes empirical
    constants. Swap this to change airfoil.

``models/structures/*.yaml``
    Structural / material properties of the elastic mount: mass, inertia,
    stiffnesses. Swap this to change the test article.

``models/cases/*.yaml``
    An analysis to run. References an airfoil (and, for flutter, a structure)
    by relative path, plus the flow condition and solver settings.

Paths inside a case file are resolved relative to the repository root, so a
case can be run from anywhere.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

from .units import cn_alpha_per_rad

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_yaml(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.is_absolute():
        p = (REPO_ROOT / p).resolve()
    if not p.exists():
        raise FileNotFoundError(f"model file not found: {p}")
    with p.open() as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"{p} did not parse to a mapping")
    return data


def _build(cls, data: dict[str, Any], *, context: str):
    """Construct a dataclass from a mapping, rejecting unknown keys.

    Silently ignoring a misspelled key is how a validation run ends up quietly
    using a default instead of the value you set, so this is strict.

    Numeric fields are also coerced. YAML 1.1 requires a *signed* exponent, so
    ``30.5e3`` parses as the string ``"30.5e3"`` rather than a float -- which
    otherwise surfaces much later as an opaque matmul dtype error. Python's
    ``float()`` accepts that spelling, so the value is recovered exactly;
    anything genuinely non-numeric raises with a pointer to the quirk.
    """
    known = {f.name for f in fields(cls)}
    unknown = set(data) - known
    if unknown:
        raise ValueError(
            f"{context}: unknown key(s) {sorted(unknown)}; "
            f"valid keys are {sorted(known)}"
        )

    numeric = {f.name for f in fields(cls) if f.type in ("float", "float | None")}
    clean = dict(data)
    for key in numeric & set(clean):
        val = clean[key]
        if val is None or isinstance(val, (int, float)):
            continue
        try:
            clean[key] = float(val)
        except (TypeError, ValueError):
            raise ValueError(
                f"{context}: {key!r} = {val!r} is not numeric. If this looks "
                "like a float, note YAML needs a signed exponent: write "
                "30.5e+3 or 30500.0, not 30.5e3."
            ) from None
    return cls(**clean)


@dataclass
class Airfoil:
    """Leishman-Beddoes empirical constants for one section.

    Defaults are the NACA0012 low-Mach values from the paper (theory.md §13).
    Constants the paper omits are sourced in theory.md §9.
    """

    name: str = "unnamed"

    # --- from the paper, Table in theory.md §13 -------------------------
    c_n_alpha: float = 0.108  # per DEGREE, converted on use
    alpha1_deg: float = 15.25
    S1_deg: float = 3.0
    S2_deg: float = 2.3
    T_p: float = 1.7
    T_f: float = 3.0
    T_v: float = 6.0
    T_vl: float = 7.0
    T_b: float = 2.2  # low-Mach onset lag (Eq. 17)
    C_N1: float = 1.75  # low-Mach onset threshold (Eq. 17)
    B1: float = 1.0  # C_N overshoot magnitude (Eq. 18)
    B2: float = 0.2  # C_m overshoot magnitude (Eq. 20)
    C_m0: float = 0.0  # symmetry-breaking static moment (Eq. 6)

    # How Eq. (18)'s overshoot enters Eq. (5). The paper never says; see
    # theory.md §7.2. "additive" adds dC_N^v to the x12 state after the fact;
    # "feed" instead injects it into the vortex feed c_v so it is integrated
    # by Eq. (15) and accumulates.
    vortex_overshoot_mode: str = "additive"

    # DIAGNOSTIC ONLY. Multiplies the separated + vortex pitching moment
    # (C_m^f + C_m^v), i.e. the dynamic-stall part that theory.md §9.3 finds
    # 48% low. It is NOT physics -- it exists to test whether that shortfall
    # explains the flutter discrepancies. Leave at 1.0 for any real result.
    c_m_stall_scale: float = 1.0

    # --- from Chantharasenawong (2007) ----------------------------------
    # sigma1 and sigma2 are switch TABLES (his Tables 2.3, 2.4), implemented in
    # aero/separation.py and aero/dynamic_stall.py -- not free parameters.
    # The one genuine constant they need is the break-angle droop:
    delta_alpha1_deg: float = 2.103  # his Table 2.1 at M=0.30: 0.0367 rad

    # --- from Leishman & Nguyen, AIAA J. 28(5) 1990 ---------------------
    # Circulatory indicial constants (their p. 839, from Beddoes).
    # The remaining non-circulatory constants K_alpha, K_q, K_alphaM, K_qM are
    # Mach-dependent functions, not free parameters -- see aero/attached.py.
    A1: float = 0.3
    A2: float = 0.7
    b1: float = 0.14
    b2: float = 0.53
    # --- separated-flow airload shapes, Chantharasenawong Table 2.1 ------
    eta: float = 0.97  # chord-force recovery factor
    K0: float = 0.0025  # aerodynamic-centre offset at f=1
    K1: float = -0.135  # centre-of-pressure travel with separation
    K2: float = 0.04  # break shape near stall
    m_exp: float = 2.0  # exponent in the K2 sin() term
    x_ac: float = 0.25  # aerodynamic centre, fraction of chord

    def __post_init__(self) -> None:
        modes = ("additive", "feed")
        if self.vortex_overshoot_mode not in modes:
            raise ValueError(
                f"vortex_overshoot_mode must be one of {modes}, "
                f"got {self.vortex_overshoot_mode!r}"
            )

    @property
    def c_n_alpha_rad(self) -> float:
        return cn_alpha_per_rad(self.c_n_alpha)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Airfoil":
        return _build(cls, _load_yaml(path), context=str(path))


@dataclass
class Structure:
    """2-DOF pitch/plunge mount (theory.md §8, Eqs. 21-25).

    ``S`` is the static mass moment. If omitted it is derived from
    ``x_theta`` (mass-centre offset from the elastic axis, in semi-chords)
    via ``S = m * b * x_theta``.
    """

    name: str = "unnamed"
    chord: float = 0.3  # m
    mass: float = 16.67  # kg/m
    I_theta: float = 0.31  # kg*m^2/m about the elastic axis
    K_h: float = 30.5e3  # N/m
    K_theta: float = 13.1  # N*m/rad
    S: float | None = None  # kg*m/m; derived from x_theta if None
    x_theta: float = 0.0  # semi-chords, mass centre aft of elastic axis
    a_h: float = -0.5  # semi-chords, elastic axis from mid-chord (-0.5 = c/4)
    C_h: float = 0.0  # plunge damping (paper has none)
    C_theta: float = 0.0  # pitch damping (paper has none)

    @property
    def b(self) -> float:
        return 0.5 * self.chord

    @property
    def static_moment(self) -> float:
        if self.S is not None:
            return self.S
        return self.mass * self.b * self.x_theta

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Structure":
        return _build(cls, _load_yaml(path), context=str(path))


@dataclass
class Flow:
    """Freestream condition."""

    rho: float = 1.225  # kg/m^3
    sound_speed: float = 340.0  # m/s
    V: float | None = None  # m/s; derived from mach if None
    mach: float | None = None  # derived from V if None

    def __post_init__(self) -> None:
        if self.V is None and self.mach is None:
            raise ValueError("flow: specify at least one of 'V' or 'mach'")
        if self.V is None:
            self.V = self.mach * self.sound_speed
        if self.mach is None:
            self.mach = self.V / self.sound_speed
        if self.mach >= 0.3:
            raise ValueError(
                f"flow: mach={self.mach:.3f} is outside the model's validated "
                "range (the low-Mach modification assumes Ma < 0.3)"
            )


@dataclass
class ForcedPitch:
    """Prescribed sinusoidal pitch, for airload validation (theory.md §10.1)."""

    alpha_mean_deg: float = 15.0
    alpha_amp_deg: float = 10.0
    k: float = 0.124  # reduced frequency, k = omega*b/V
    n_cycles: int = 6
    steps_per_cycle: int = 2000


@dataclass
class Case:
    """A runnable analysis."""

    name: str
    kind: str  # "forced_pitch" | "flutter"
    airfoil: Airfoil
    flow: Flow
    structure: Structure | None = None
    forced_pitch: ForcedPitch | None = None
    notes: str = ""
    reference: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Case":
        data = _load_yaml(path)
        ctx = str(path)

        try:
            name = data.pop("name")
            kind = data.pop("kind")
        except KeyError as exc:
            raise ValueError(f"{ctx}: missing required key {exc}") from exc

        if kind not in ("forced_pitch", "flutter"):
            raise ValueError(f"{ctx}: unknown kind {kind!r}")

        airfoil = Airfoil.from_yaml(data.pop("airfoil"))

        structure = None
        if "structure" in data:
            structure = Structure.from_yaml(data.pop("structure"))
        if kind == "flutter" and structure is None:
            raise ValueError(f"{ctx}: kind 'flutter' requires a 'structure'")

        # Per-case overrides of airfoil constants, e.g. C_m0 for the
        # symmetry-breaking sweep in theory.md §10.3.
        for key, val in (data.pop("airfoil_overrides", None) or {}).items():
            if not hasattr(airfoil, key):
                raise ValueError(f"{ctx}: airfoil_overrides has unknown key {key!r}")
            setattr(airfoil, key, val)

        flow = _build(Flow, data.pop("flow"), context=f"{ctx}:flow")

        fp = None
        if "forced_pitch" in data:
            fp = _build(
                ForcedPitch, data.pop("forced_pitch"), context=f"{ctx}:forced_pitch"
            )
        if kind == "forced_pitch" and fp is None:
            raise ValueError(f"{ctx}: kind 'forced_pitch' requires 'forced_pitch'")

        return cls(
            name=name,
            kind=kind,
            airfoil=airfoil,
            flow=flow,
            structure=structure,
            forced_pitch=fp,
            notes=data.pop("notes", ""),
            reference=data.pop("reference", {}) or {},
        )
