"""Structural model and coupled-system tests (theory.md §8, §10.3)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from twod_flutter.config import Case, Structure
from twod_flutter.structure import StructuralModel
from twod_flutter.system import AeroelasticSystem, N_TOTAL

#: Index of the vortex clock x11 in the coupled state vector. It is a pure ramp
#: (xdot = V/b, independent of every state), so it contributes a structural zero
#: eigenvalue that masks the true maximum. Excluded from stability analysis.
I_VORTEX_CLOCK = 10


@pytest.fixture
def struct() -> Structure:
    return Structure.from_yaml("models/structures/dimitriadis_naca0012.yaml")


def test_uncoupled_frequencies(struct):
    """omega = sqrt(k/m) for each DOF, from the paper's stated properties."""
    sm = StructuralModel(struct)
    wh, wt = sm.uncoupled_frequencies
    assert wh == pytest.approx(np.sqrt(struct.K_h / struct.mass))
    assert wt == pytest.approx(np.sqrt(struct.K_theta / struct.I_theta))
    # The two modes are far apart (ratio ~6.6), so they cannot coalesce and
    # classical bending-torsion flutter is impossible for this rig.
    assert wh / wt > 5.0


def test_yaml_unsigned_exponent_is_recovered(tmp_path):
    """YAML 1.1 parses '30.5e3' as a *string* -- it requires a signed exponent.

    Left alone that surfaces much later as an opaque matmul dtype error, so the
    loader coerces numeric fields. Python's float() accepts the string form, so
    the value is recovered exactly rather than rejected.
    """
    p = tmp_path / "quirk.yaml"
    p.write_text("name: x\nK_h: 30.5e3\n")
    assert Structure.from_yaml(p).K_h == pytest.approx(30500.0)


def test_yaml_rejects_genuinely_non_numeric(tmp_path):
    """A value that is not a number at all must fail loudly, with a hint."""
    p = tmp_path / "bad.yaml"
    p.write_text("name: x\nK_h: thirty thousand\n")
    with pytest.raises(ValueError, match="signed exponent"):
        Structure.from_yaml(p)


def test_quarter_chord_axis_cannot_diverge(struct):
    """a_h = -0.5 puts the elastic axis under the lift: no moment arm.

    This is the paper's implied configuration, and it is why Eq. (24) as
    printed cannot reproduce the 17.7 m/s divergence its own Ref. [3] reports.
    """
    struct.a_h = -0.5
    sm = StructuralModel(struct)
    assert sm.ac_to_ea == pytest.approx(0.0)
    assert sm.divergence_velocity(6.1879, 1.225) == float("inf")
    # And the moment transfer term vanishes, recovering Eq. (24) exactly.
    _, q_t = sm.generalised_forces(c_l=1.0, c_m=0.05, rho=1.225, V=10.0, chord=0.3)
    assert q_t == pytest.approx(0.5 * 1.225 * 100.0 * 0.3**2 * 0.05)


def test_divergence_velocity_matches_reference(struct):
    """a_h is calibrated so linear divergence lands on Ref. [3]'s 17.7 m/s."""
    sm = StructuralModel(struct)
    assert sm.ac_to_ea > 0.0
    assert sm.divergence_velocity(6.1879, 1.225) == pytest.approx(17.7, abs=0.05)


def test_coupled_system_linearises_to_divergence_at_reference_speed():
    """The full 18-state system must lose stability at ~17.7 m/s, aperiodically.

    This checks the calibration end to end rather than through the closed-form
    balance: it linearises the actual coupled right-hand side. The unstable mode
    must be *real* (divergence), not oscillatory -- there is no Hopf here.
    """
    case = Case.from_yaml("models/cases/naca0012_flutter.yaml")
    case.airfoil.C_m0 = 0.0  # trivial equilibrium at theta = h = 0
    keep = [i for i in range(N_TOTAL) if i != I_VORTEX_CLOCK]

    def max_eig(V: float) -> complex:
        case.flow.V = V
        case.flow.mach = V / case.flow.sound_speed
        s = AeroelasticSystem(case)
        x0 = s.initial_state(theta0=0.0)
        J = np.zeros((N_TOTAL, N_TOTAL))
        eps = 1e-7
        for i in range(N_TOTAL):
            xp = x0.copy(); xp[i] += eps
            xm = x0.copy(); xm[i] -= eps
            J[:, i] = (s.derivatives(0.0, xp) - s.derivatives(0.0, xm)) / (2 * eps)
        ev = np.linalg.eigvals(J[np.ix_(keep, keep)])
        return ev[int(np.argmax(ev.real))]

    assert max_eig(17.0).real < 0.0, "stable below the divergence speed"
    unstable = max_eig(18.0)
    assert unstable.real > 0.0, "unstable above it"
    assert abs(unstable.imag) < 1e-3, "the unstable mode is real: divergence, not Hopf"


def test_alpha_and_q_follow_equation_25():
    """alpha = theta + h_dot/V and q = theta_dot*c/V, with the paper's ordering."""
    case = Case.from_yaml("models/cases/naca0012_flutter.yaml")
    s = AeroelasticSystem(case)
    V, c = case.flow.V, case.structure.chord
    xs = np.array([0.1, 0.2, 0.3, 0.4])  # [h, theta, h_dot, theta_dot]
    alpha, q = s.struct.alpha_and_q(xs, V, c)
    assert alpha == pytest.approx(0.2 + 0.3 / V)
    assert q == pytest.approx(0.4 * c / V)
