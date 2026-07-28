"""Unit tests for the aerodynamic model.

These pin down the parts we can verify independently of the paper's missing
definitions (theory.md §9) -- unit conventions, the Kirchhoff correlation, the
attached-flow indicial limits, and the behaviour of the paper's own
contribution (the delayed stall criterion).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twod_flutter.aero import LBModel, attached, dynamic_stall as ds, separation as sep
from twod_flutter.config import Airfoil, Case
from twod_flutter.units import RAD2DEG, cn_alpha_per_rad, rad


@pytest.fixture
def af() -> Airfoil:
    return Airfoil()


# --- units -------------------------------------------------------------
def test_cn_alpha_converts_to_near_two_pi():
    """0.108 /deg is the paper's value; in /rad it must land near 2*pi.

    This is the guard against the degrees/radians trap in theory.md §2.
    """
    assert cn_alpha_per_rad(0.108) == pytest.approx(6.188, abs=1e-3)
    assert abs(cn_alpha_per_rad(0.108) - 2 * np.pi) < 0.1


# --- Kirchhoff separation point, Eq. (11) -------------------------------
def test_kirchhoff_continuous_at_break(af):
    """Both branches of Eq. (11) must meet at |alpha| = alpha1."""
    a1 = af.alpha1_deg
    lo = sep.kirchhoff_f(rad(a1 - 1e-9), a1, af)
    hi = sep.kirchhoff_f(rad(a1 + 1e-9), a1, af)
    assert lo == pytest.approx(0.70, abs=1e-6)
    assert hi == pytest.approx(0.70, abs=1e-6)


def test_kirchhoff_is_even(af):
    """Eq. (11) is written in |alpha|, so it cannot break symmetry itself.

    This is *why* the paper needs C_m0 to produce asymmetric LCOs
    (theory.md §10.3).
    """
    for a in (5.0, 15.25, 25.0):
        assert sep.kirchhoff_f(rad(a), af.alpha1_deg, af) == pytest.approx(
            sep.kirchhoff_f(rad(-a), af.alpha1_deg, af)
        )


def test_kirchhoff_bounds_and_monotonicity(af):
    """Fully attached at zero incidence, decaying monotonically to ~0.04."""
    # At alpha = 0: f = 1 - 0.3*exp(-15.25/3.0) = 0.998, i.e. nearly attached.
    assert sep.kirchhoff_f(0.0, af.alpha1_deg, af) == pytest.approx(0.998, abs=1e-3)
    vals = [sep.kirchhoff_f(rad(a), af.alpha1_deg, af) for a in np.linspace(0, 40, 60)]
    assert np.all(np.diff(vals) <= 1e-12)
    assert vals[0] <= 1.0
    assert vals[-1] == pytest.approx(0.04, abs=1e-3)


# --- attached flow, Eq. (8) --------------------------------------------
def test_alpha_effective_reaches_alpha_in_steady_state(af):
    """Constant alpha must drive alpha_E -> alpha (phi(inf) = 1)."""
    mach, alpha = 0.12, rad(5.0)
    x = np.zeros(8)
    ds_step = 0.01
    for _ in range(200_000):
        x += attached.derivatives(x, alpha, 0.0, af, mach) * ds_step
    assert attached.alpha_effective(x, af, mach) == pytest.approx(alpha, rel=1e-4)


def test_steady_state_is_a_fixed_point(af):
    """attached.steady_state must actually have zero rate, not merely be close."""
    mach, alpha = 0.12, rad(8.0)
    x = attached.steady_state(alpha, af, mach)
    assert np.allclose(attached.derivatives(x, alpha, 0.0, af, mach), 0.0, atol=1e-12)
    assert attached.alpha_effective(x, af, mach) == pytest.approx(alpha, rel=1e-12)


def test_indicial_response_starts_at_zero(af):
    """phi(0) = 1 - A1 - A2 = 0: no circulatory lift at the instant of a step."""
    assert af.A1 + af.A2 == pytest.approx(1.0)
    assert attached.alpha_effective(np.zeros(8), af, 0.12) == pytest.approx(0.0)


def test_lift_curve_slope_matches_compressible_theory(af):
    """The state scaling must reproduce C_N^C = (2*pi/beta)*alpha in steady flow.

    Guards the xhat = x*(2V/c) rescaling of Leishman & Nguyen Eq. (17), where
    B carries no 2V/c factor and the raw states are dimensional.
    """
    mach, alpha = 0.12, rad(3.0)
    x = attached.steady_state(alpha, af, mach)
    # alpha_E -> alpha, so C_N^C/alpha recovers whatever slope is applied.
    slope = attached.circulatory_normal_force(x, af, mach) / alpha
    assert slope == pytest.approx(af.c_n_alpha_rad, rel=1e-9)
    # And the theoretical compressible value is within 3% of the airfoil's.
    assert abs(slope - 2 * np.pi / attached.beta(mach)) / slope < 0.03


def test_impulsive_loads_vanish_in_steady_flow(af):
    """Added-mass terms must contribute nothing when nothing is accelerating."""
    mach, alpha = 0.12, rad(8.0)
    x = attached.steady_state(alpha, af, mach)
    c_n_i, _ = attached.impulsive_loads(x, alpha, 0.0, af, mach)
    assert c_n_i == pytest.approx(0.0, abs=1e-10)


def test_k_constants_carry_the_25_percent_reduction(af):
    """Leishman & Nguyen reduce the theoretical K by 25% to match test data."""
    k_a, k_q, k_am, k_qm = attached._k_constants(0.12)
    # Theoretical K_alpha at M=0.12, their Eq. (A2) numerator inverted.
    sum_ab = 0.3 * 0.14 + 0.7 * 0.53
    bta = attached.beta(0.12)
    theo = 1.0 / ((1.0 - 0.12) + np.pi * bta * 0.12**2 * sum_ab)
    assert k_a == pytest.approx(0.75 * theo, rel=1e-12)
    # K_q relaxes faster than K_alpha (2*pi vs pi in the denominator).
    assert k_q < k_a


# --- the paper's contribution, Eq. (17) --------------------------------
def test_onset_lag_delays_stall(af):
    """x14 is a lag of x9, so it must cross the threshold later.

    This is the whole point of the low-Mach modification: the baseline
    criterion (Eq. 13, on x9) fires "too early".
    """
    x9, x14 = 0.0, 0.0
    step, crossed9, crossed14 = 0.01, None, None
    for i in range(200_000):
        s = i * step
        x9 += (0.5 * s - x9) / af.T_p * step  # x9 chases a ramp
        x14 += ds.onset_lag_derivative(x9, x14, af) * step
        if crossed9 is None and x9 > af.C_N1:
            crossed9 = s
        if crossed14 is None and x14 > af.C_N1:
            crossed14 = s
            break
    assert crossed9 is not None and crossed14 is not None
    assert crossed14 > crossed9


def test_vortex_shape_bounds(af):
    """V_x (Eq. 19) rises from 0, peaks at 1 at tau = T_v, then decays."""
    assert ds.vortex_shape(0.0, af) == pytest.approx(0.0)
    assert ds.vortex_shape(af.T_v, af) == pytest.approx(1.0)
    taus = np.linspace(0, 4 * af.T_vl, 400)
    vals = np.array([ds.vortex_shape(t, af) for t in taus])
    assert vals.min() >= 0.0
    assert vals.max() <= 1.0 + 1e-12
    assert vals[-1] == pytest.approx(0.0)


# --- integration smoke test --------------------------------------------
def test_forced_pitch_runs_and_stays_finite():
    from twod_flutter.analysis.forced_pitch import run

    case = Case.from_yaml("models/cases/naca0012_forced_pitch.yaml")
    case.forced_pitch.n_cycles = 2
    case.forced_pitch.steps_per_cycle = 1200
    r = run(case).last_cycle()

    assert np.all(np.isfinite(r.c_n))
    assert np.all(np.isfinite(r.c_m))
    # Separation point is a fraction of chord and must stay physical.
    assert r.f_delayed.min() >= 0.0
    assert r.f_delayed.max() <= 1.0
    # Dynamic stall must actually fire on this case (alpha peaks at 25 deg).
    assert r.stalled.any()


def test_lb_core_matches_published_verification():
    """The LB core must reproduce Chantharasenawong (2007) Fig. 2.9.

    This is the load-bearing validation: it checks our Leishman-Beddoes
    implementation against an independently published result, at that author's
    baseline settings and WITHOUT the source paper's low-Mach modification. It
    is what lets us attribute the remaining Fig. 3 discrepancy (theory.md §9.3)
    to the modification rather than to the core model.
    """
    from twod_flutter.analysis.forced_pitch import run

    case = Case.from_yaml("models/cases/naca0012_thesis_fig29.yaml")
    r = run(case).last_cycle()

    # Published: C_N peak 1.95, C_m min -0.30. Tolerance is 10%: the targets are
    # read off a figure, so tighter would be false precision.
    assert r.c_n.max() == pytest.approx(1.95, rel=0.10)
    assert r.c_m.min() == pytest.approx(-0.30, rel=0.10)
    # Peak must land in the right place, not just at the right height.
    i = int(np.argmax(r.c_n))
    assert 17.5 <= r.alpha_deg[i] <= 19.5


def test_vortex_overshoot_modes_both_run():
    """Both readings of Eq. (18) must integrate stably (theory.md §9.3)."""
    from twod_flutter.analysis.forced_pitch import run

    peaks = {}
    for mode in ("additive", "feed"):
        case = Case.from_yaml("models/cases/naca0012_forced_pitch.yaml")
        case.airfoil.vortex_overshoot_mode = mode
        case.forced_pitch.n_cycles = 2
        peaks[mode] = run(case).last_cycle().c_n.max()
    assert all(np.isfinite(v) for v in peaks.values())
    # The two readings differ by under 5% -- which is why the ambiguity was
    # ruled out as the cause of the Fig. 3 discrepancy.
    assert abs(peaks["additive"] - peaks["feed"]) / peaks["additive"] < 0.05


def test_airfoil_rejects_bad_overshoot_mode():
    with pytest.raises(ValueError, match="vortex_overshoot_mode"):
        Airfoil(vortex_overshoot_mode="nonsense")


def test_pitch_rate_factor_recovers_leishman_at_quarter_chord():
    """[LN]'s alpha_3/4 = alpha + q/2 assumes pitching about the quarter chord."""
    assert attached.pitch_rate_factor(attached.QUARTER_CHORD_AH) == pytest.approx(1.0)
    # Dimitriadis & Li Table 1 puts the pitch axis at 0.115 m = 0.3833c,
    # a_h = -0.2333, which shortens the arm to the 3/4-chord point.
    f = attached.pitch_rate_factor(-0.2333)
    assert f == pytest.approx(0.7333, abs=1e-4)
    # Using q/2 regardless would over-predict that term by ~36%.
    assert 1.0 / f == pytest.approx(1.364, abs=0.01)


def test_pitch_axis_leaves_forced_pitch_unchanged():
    """Prescribed-motion cases pitch about the quarter chord, so must not move.

    Guards the default: threading a_h through the aerodynamics must be inert
    unless a caller explicitly supplies a different pitch axis.
    """
    af = Airfoil()
    x = attached.steady_state(rad(6.0), af, 0.12)
    base = attached.evaluate(x, rad(6.0), 0.05, af, 0.12)
    dflt = attached.evaluate(x, rad(6.0), 0.05, af, 0.12, attached.QUARTER_CHORD_AH)
    assert np.allclose(base[0], dflt[0])
    assert base[1:] == pytest.approx(dflt[1:])
    # A different axis must actually change the answer.
    moved = attached.evaluate(x, rad(6.0), 0.05, af, 0.12, -0.2333)
    assert not np.allclose(base[0], moved[0])


def test_config_rejects_unknown_keys(tmp_path):
    """A misspelled key must fail loudly, not silently fall back to a default."""
    p = tmp_path / "bad.yaml"
    p.write_text("name: x\nc_n_alpha: 0.108\nT_pp: 1.7\n")
    with pytest.raises(ValueError, match="unknown key"):
        Airfoil.from_yaml(p)
