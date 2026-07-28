# 2d-flutter

Python reimplementation of the airfoil aeroelastic stall-flutter model from:

> Shao S., Zhu Q., Zhang C., Ni X.,
> "Airfoil Aeroelastic Flutter Analysis Based on Modified Leishman-Beddoes Model at Low Mach Number,"
> *Chinese Journal of Aeronautics* **24**(5), 2011, 550–557.
> [doi:10.1016/S1000-9361(11)60064-2](https://doi.org/10.1016/S1000-9361(11)60064-2) (open access, CC BY-NC-ND)

An 18-state ODE system coupling a modified Leishman–Beddoes dynamic stall model (14 states,
valid for Ma < 0.3) to a 2-DOF pitch/plunge rigid airfoil (4 states), integrated in the time
domain to study stall-flutter bifurcation behaviour.

## Status

Theory extraction complete. No code yet.

**→ Start here: [`docs/theory.md`](docs/theory.md)** — full transcription of all 25 equations,
constants, validation targets, and the gap register.

The paper is **not self-contained**: roughly 40% of the model (the 8 attached-flow states, several
coefficient definitions, and the separated-flow airload expressions) is deferred to its references.
See §9 of the theory document for the itemised list of what must be sourced before anything can run.

## Repository contents

| Path | |
|---|---|
| `docs/theory.md` | Theory reference and implementation plan |
| `references/Shao2011-*.pdf` | The source paper (open-access version) |
| `Airfoil Aeroelastic Flutter Analysis...pdf` | ⚠️ DRM-encrypted (Microsoft RMS), unreadable — superseded by `references/` |
