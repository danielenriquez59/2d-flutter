# 2d-flutter

Python reimplementation of the airfoil aeroelastic stall-flutter model from:

> Shao S., Zhu Q., Zhang C., Ni X.,
> "Airfoil Aeroelastic Flutter Analysis Based on Modified Leishman-Beddoes Model at Low Mach Number,"
> *Chinese Journal of Aeronautics* **24**(5), 2011, 550–557.
> [doi:10.1016/S1000-9361(11)60064-2](https://doi.org/10.1016/S1000-9361(11)60064-2) (open access, CC BY-NC-ND)

An 18-state ODE system coupling a modified Leishman–Beddoes dynamic stall model (14 states,
valid for Ma < 0.3) to a 2-DOF pitch/plunge rigid airfoil (4 states), integrated in the time
domain to study stall-flutter bifurcation behaviour.

**→ Theory reference: [`docs/theory.md`](docs/theory.md)** — all 25 equations, constants,
validation targets, and the provenance of everything the paper leaves undefined.

The paper is not self-contained; two further sources are required to implement it:

- **[LN]** Leishman & Nguyen, *AIAA J.* **28**(5), 1990, 836–844 — attached flow, states `x1…x8`.
- **[CH]** Chantharasenawong, PhD thesis, Imperial College London, 2007 (the paper's Ref. [17]) —
  separated-flow airloads, the `σ₁`/`σ₂` switch tables, `α₁ₙ`, `c_v`, vortex-clock reset.

## Status

| Stage | State |
|---|---|
| Theory extraction | ✅ complete |
| 14-state aerodynamic model | ✅ complete, all deferred formulation sourced |
| **Validation §10.0 (LB core vs [CH] Fig. 2.9)** | ✅ **passing, within 3–5%** |
| Validation §10.1 (NACA0012 forced pitch, Figs. 3–4) | ⚠️ shape right, magnitude low — see below |
| Validation §10.2 (OA207 forced pitch) | ⬜ not started |
| 4-state structural model + coupling | ✅ complete (18-state system) |
| Validation §10.3 (flutter bifurcation) | ⚠️ structure reproduced, numbers not — see below |

### The LB core is verified

`models/cases/naca0012_thesis_fig29.yaml` reproduces **[CH]**'s own published LB verification
(his Fig. 2.9), at his baseline settings and without this paper's low-Mach modification:

| Quantity | [CH] Fig. 2.9 | Model | |
|---|---|---|---|
| `C_N` peak | 1.95 | 2.009 | +3.0% ✅ |
| `C_m` min | −0.30 | −0.281 | +6.3% ✅ |
| peak location | α ≈ 18–19° | α ≈ 18.8° | ✅ |

This is the load-bearing check: it validates our Leishman–Beddoes implementation against an
independent published result, and so separates *"is the core right?"* from *"is our reading of
Shao's Eqs. 17–20 right?"*

### The remaining discrepancy is in the low-Mach modification

`models/cases/naca0012_forced_pitch.yaml` reproduces the *character* of the paper's Figs. 3–4 —
the vortex-onset kink, the peak near α ≈ 24°, and the wide downstroke hysteresis — but
under-predicts magnitude:

| Quantity | Paper | Model | |
|---|---|---|---|
| `C_N` peak | 2.55 | 1.78 | −30% |
| `C_m` min | −0.62 | −0.32 | −48% |
| `C_N` peak location | α ≈ 24–25° | α ≈ 24.2° | ✅ |

The paper's own contribution does work: the modified criterion (Eq. 17) delays stall onset from
**α = 19.3°** (baseline Eq. 13) to **α = 21.6°** — a 2.25° delay, which is exactly the "stall
onset predicted too early" defect it set out to fix.

Since §10.0 passes, this is **not** a core-model error. Two candidate explanations have been
tested and **eliminated**:

1. **The Eq. (18) combination ambiguity** — the paper never states whether `ΔC_N^v` adds to the
   `x12` state or feeds it. Both readings are implemented (`vortex_overshoot_mode`) and differ by
   under 2%: `"additive"` → 1.783, `"feed"` → 1.752.
2. **Calibration of `B₁`** — `B₁ = 5`, five times the paper's value, reaches only `C_N` = 2.24 and
   drives `C_m` the *wrong* way.

So the residual is specific to this paper's low-Mach modification at deep-stall amplitude.
Remaining untested suspects are listed in [`docs/theory.md`](docs/theory.md) §9.3.

### Flutter: Eq. (24) is incomplete, and there is no Hopf

Two findings from the coupled system (details in [`docs/theory.md`](docs/theory.md) §10.3):

**Eq. (24) is missing a term, and the paper's own numbers prove it.** `I_θ` is defined about the
quarter chord, implying the elastic axis is there — but that puts the elastic axis directly under
the lift, giving zero moment arm and an *infinite* divergence speed, while the paper quotes
17.7 m/s from its Ref. [3]. The missing lift-transfer term is `½ρV²c·C_L·d`, `d = b(a_h + 0.5)`;
with `a_h = −0.5` it vanishes and Eq. (24) is recovered exactly. Inverting the divergence balance
at 17.7 m/s then **recovers `a_h = −0.2548`** (elastic axis at 0.373c) — turning a number the
paper states into one of the two structural parameters it omits.

**There is no Hopf bifurcation.** Linearising the 18-state system, both oscillatory modes stay
damped — damping increasing monotonically — until a *real* eigenvalue crosses zero at 17.70 m/s.
That is divergence, not flutter. The modes sit at 6.81 Hz and 1.02 Hz (ratio 6.6) and never
coalesce, so classical bending–torsion flutter is impossible for this rig.

The nonlinear sweep reproduces the paper's *structure* — subcriticality (1° decays, 20° sustains
an LCO below 18 m/s), the transition at the divergence speed, and a static offset growing with V
— but not its numbers: LCO amplitude is ~2.5× too large and no asymmetric LCO appears. The
amplitude error is very likely inherited from §10.1, where `C_m` is 48% low; `C_m` *is* the pitch
damping that sets LCO amplitude.

### The primary source resolves most of it

Ref. [3] (Dimitriadis & Li, *AIAA J.* 47(11), 2009) settles the structural questions and
**refutes** one hypothesis. Details in [`docs/theory.md`](docs/theory.md) §10.3.2.

- **The elastic-axis derivation was right.** Table 1 puts the pitch axis at 0.115 m = 0.3833c
  (`a_h = −0.2333`); our value inferred from the divergence balance was −0.2548, **within 3%**.
- **Divergence now matches.** With Table 1's per-span properties, linear theory gives **17.89 m/s**;
  the experiment's asymmetric-LCO onset is at **17.8 m/s**, and the paper states explicitly that
  this bifurcation occurs *at the static divergence airspeed*.
- **Units inconsistency in Shao.** Table 1 is total-for-0.9 m-span; Shao converts the mass to
  per-span but leaves `I_θ`, `K_h`, `K_θ` total. Correcting all four raises `ω_h` 6.81 → 7.18 Hz.
- **Two confirmations of our structural model.** The source says the response is "single mode"
  because plunge is far stiffer than pitch — our ratio is 6.9 — and the rig was built
  "without friction", so near-zero structural damping is by design.
- **The mean-incidence hypothesis is refuted.** The source states the equilibrium is at zero
  pitch and loses stability at 13 m/s.

**Pitch-axis correction applied.** [LN]'s `α_{3/4} = α + q/2` assumes pitching about the quarter
chord; for an axis at `a_h` the factor is `(0.5 − a_h)`. With this rig's `a_h = −0.2333` that is
**0.7333**, so the pitch-rate downwash had been over-predicted by **36%**. `a_h = −0.5` returns
1.0 and recovers [LN] exactly, so forced-pitch cases are unaffected. It is a correctness fix, not
a resolution: the torsion mode stays damped.

The experiment's sequence is fold at 12.2, subcritical Hopf at 13.0, divergence-driven asymmetric
LCO at 17.8. We reproduce the divergence bifurcation, put the fold ~2 m/s low, and miss the Hopf.

### The `C_m` hypothesis is refuted — in both directions

A diagnostic scale on the dynamic-stall moment (`c_m_stall_scale`) tests the standing hypothesis
that the 48% `C_m` shortfall explains the flutter discrepancies. It does not
([`docs/theory.md`](docs/theory.md) §10.3.2).

**Linearly, it does nothing.** At V = 13 m/s a **tenfold** increase moves the eigenvalue from
−0.17174 to −0.17150 — 0.014% — and never produces a Hopf. The reason is structural: at α ≈ 0 the
flow is attached (`f ≈ 0.998`), so `C_m^f ≈ K₀·C_N^f ≈ 0` and `C_m^v = 0`. **The deficient term is
identically inactive at the linearisation point.**

**Nonlinearly, the sign is backwards.** Correcting `C_m` toward the paper's value takes the LCO
from 101° to **146°** peak-to-peak against a target of ~40° — the error more than doubles. The
premise was wrong: the dynamic-stall moment is not damping, it is the **energy source** (Ref. [3]:
dynamic stall "allows the transfer of energy from the freestream to the wing"). Amplifying the
driver amplifies the limit cycle.

**And the amplitudes are outside the model's validity anyway** — ±50° at scale 1.0, ±73° at 1.92,
against a Kirchhoff range of ±60°. The amplitude comparison is an extrapolation, not a test.

Five candidates for the missing Hopf are now eliminated: the `C_m` shortfall, mean incidence,
`x_θ`, the Eq. (18) ambiguity, and the pitch-axis factor. Since the attached-flow model is
independently validated to 3–5% (§10.0), what remains is that the experimental Hopf may not be a
linear instability of the attached-flow equilibrium at all. Ref. [3] supports this: it attributes
the Hopf to dynamic stall, which cannot act at zero incidence, and a fold at 12.2 m/s with a
rapidly shrinking basin would present as a Hopf at 13 m/s under hand-administered impulses.

### Limitations

**[`docs/theory.md`](docs/theory.md) §15** documents the method's envelope in full. The headlines:

- **No Reynolds number anywhere in the model**, though dynamic stall is strongly Re-dependent.
  Both validation cases sit at Re ≈ 1–2 × 10⁶, which masks the omission.
- **Constants are fitted at Ma ≥ 0.30 but §10.1 runs at Ma = 0.12.** The low-Mach modification
  replaces only two of them (`T_b`, `C_N1`). This is an untested suspect for the §10.1 shortfall,
  and it explains why §10.0 — at the constants' native Mach — passes.
- **The model is non-smooth**, and the paper's conclusions are *about bifurcations*. Non-smooth
  systems admit transitions with no smooth analogue, so the "subcritical Hopf" classification
  presumes regularity the model lacks. Treat §10.3's labels as descriptive; check whether
  transition velocities move with step size.
- **Constants are NACA0012-specific.** Applying them to another section fails silently.
- **No structural damping, 2 DOF, rigid, strictly 2D**, and added mass enters as a load rather
  than in the mass matrix.

Correct constants also make the system **stiff** (ratio ~354; fastest state τ = 0.020
semi-chords), so explicit RK4 needs >891 steps/cycle here. The driver raises with the required
step count rather than returning a silently diverged run. This is precisely why the paper uses
implicit Newmark.

## Layout

```
src/twod_flutter/
    units.py          deg/rad discipline — the #1 correctness risk
    config.py         YAML → dataclasses, strict about unknown keys
    integrators.py    fixed-step RK4 + implicit Newmark average-velocity
    aero/
        attached.py       Eq. 8      states x1–x8   [LN]
        separation.py     Eqs. 9–12, 16   x9, x10, x13
        dynamic_stall.py  Eqs. 13–15, 17–20  x11, x12, x14
        airloads.py       Eqs. 2–7   assembly       [CH]
        model.py          the 14-state model
    analysis/
        forced_pitch.py   prescribed-motion driver
        plotting.py       figures
models/
    airfoils/         aerodynamic identity of a section
    structures/       mass / inertia / stiffness of the mount
    cases/            an analysis to run: airfoil + structure + flow + solver
scripts/              entry points that take a case file
tests/
```

Separation of concerns is deliberate: swap `airfoils/*.yaml` to change section, `structures/*.yaml`
to change test article, `cases/*.yaml` to change the analysis. Nothing physical is hard-coded in
the scripts.

## Usage

```bash
pip install -e ".[plots,dev]"

python scripts/run_forced_pitch.py models/cases/naca0012_thesis_fig29.yaml   # core check
python scripts/run_forced_pitch.py models/cases/naca0012_forced_pitch.yaml   # paper Figs. 3-4
pytest -q
```

The runner prints computed-vs-paper values and writes hysteresis loops to `results/`.

## Dependencies

NumPy and PyYAML for the model; matplotlib and pandas only for plotting. **No SciPy** — the
right-hand side has hard switches (Eq. 15's feeding condition, the stall trigger), which adaptive
solvers mis-handle by construction, so integration is fixed-step with events applied between
steps.

## Repository contents

| Path | |
|---|---|
| `docs/theory.md` | Theory reference and implementation plan |
| `references/Shao2011-*.pdf` | The source paper (open-access version) |
| `Airfoil Aeroelastic Flutter Analysis...pdf` | ⚠️ DRM-encrypted (Microsoft RMS), unreadable — superseded by `references/` |
