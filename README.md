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
| 4-state structural model + coupling | ⬜ not started |
| Validation §10.3 (flutter bifurcation) | ⬜ not started |

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

One item is still to source, and it blocks the flutter validation (§10.3) only: the structural
static mass moment `S` and density `ρ`, which must come from Ref. [3] (Dimitriadis & Li).

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
