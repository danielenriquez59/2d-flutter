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
validation targets, and the register of what the paper leaves undefined.

## Status

| Stage | State |
|---|---|
| Theory extraction | ✅ complete |
| 14-state aerodynamic model | ✅ runs end to end |
| Validation §10.1 (NACA0012 forced pitch) | ⚠️ shape right, magnitude low — see below |
| Validation §10.2 (OA207 forced pitch) | ⬜ not started |
| 4-state structural model + coupling | ⬜ not started |
| Validation §10.3 (flutter bifurcation) | ⬜ not started |

### Current validation result

`models/cases/naca0012_forced_pitch.yaml` reproduces the *character* of the paper's Figs. 3–4 —
the vortex-onset kink, the peak near α ≈ 24°, and the wide downstroke hysteresis — but
under-predicts magnitude:

| Quantity | Paper | Model | |
|---|---|---|---|
| `C_N` peak | 2.55 | 1.92 | −25% |
| `C_m` min | −0.62 | −0.35 | −44% |
| `C_N` peak location | α ≈ 24–25° | α ≈ 24.1° | ✅ |

The paper's own contribution does work: the modified criterion (Eq. 17) delays stall onset from
**α = 19.3°** (baseline Eq. 13) to **α = 21.6°** — a 2.25° delay, which is exactly the "stall
onset predicted too early" defect it set out to fix.

The magnitude shortfall is expected and is traced to the **unclosed gaps** — roughly 40% of the
model is deferred by the paper to its references, and those pieces are currently canonical
Leishman–Beddoes reconstructions rather than the paper's actual definitions. See
[`docs/theory.md`](docs/theory.md) §9. The dominant suspect is the vortex-lift path
(GAP-3, `c_v`) and the additive-vs-replacement ambiguity in Eq. (18).

## Layout

```
src/twod_flutter/
    units.py          deg/rad discipline — the #1 correctness risk
    config.py         YAML → dataclasses, strict about unknown keys
    integrators.py    fixed-step RK4 + implicit Newmark average-velocity
    aero/
        attached.py       Eq. 8      states x1–x8   [GAP-1]
        separation.py     Eqs. 9–12, 16   x9, x10, x13
        dynamic_stall.py  Eqs. 13–15, 17–20  x11, x12, x14
        airloads.py       Eqs. 2–7   assembly       [GAP-4]
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

python scripts/run_forced_pitch.py models/cases/naca0012_forced_pitch.yaml
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
