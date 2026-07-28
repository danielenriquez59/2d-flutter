# Theory Reference — Modified Leishman–Beddoes Stall Flutter Model

Working reference for reimplementing the airfoil aeroelastic stall-flutter model in Python.

**Source:** Shao Song, Zhu Qinghua, Zhang Chenglin, Ni Xianping,
"Airfoil Aeroelastic Flutter Analysis Based on Modified Leishman-Beddoes Model at Low Mach Number,"
*Chinese Journal of Aeronautics* **24**(5), 2011, pp. 550–557.
DOI [10.1016/S1000-9361(11)60064-2](https://doi.org/10.1016/S1000-9361(11)60064-2). Open access (CC BY-NC-ND).

Equation numbers below match the paper. Everything in §2–§7 is transcribed from the paper.

The paper is not self-contained — it defers roughly 40% of the model to its references. Those
pieces have all been recovered; **§9 records where each came from**, and is the section to read
before changing any coefficient.

**Supporting sources** (both required to implement the paper):

- **[LN]** Leishman J.G., Nguyen K.Q., "State-Space Representation of Unsteady Airfoil Behavior,"
  *AIAA Journal* **28**(5), 1990, 836–844. → attached flow, states `x1…x8`.
- **[CH]** Chantharasenawong C., "Nonlinear aeroelastic behaviour of aerofoils under dynamic
  stall," PhD thesis, Imperial College London, 2007 (the paper's Ref. [17]).
  → separated-flow airloads, the `σ₁`/`σ₂` switch tables, `α₁ₙ`, `c_v`, vortex-clock reset.

---

## 1. Scope and intent

The paper builds an 18-state ODE system coupling:

- a **14-state modified Leishman–Beddoes (L-B) unsteady aerodynamics / dynamic stall model**, valid at low Mach number (Ma < 0.3), and
- a **4-state 2-DOF (pitch + plunge) rigid airfoil structural model**.

It is integrated in the time domain, and stall-flutter behaviour is characterised by bifurcation
diagrams, LCO phase-plane plots, and time responses as freestream velocity `V` is swept.

The paper's three claimed contributions over baseline L-B:

1. A **delayed dynamic stall onset criterion** for low Mach (extra first-order lag, state `x14`) — §7.1.
2. A **normal-force overshoot correction** after stall onset — §7.2.
3. A matching **pitching-moment correction** — §7.2.

Baseline L-B empirical constants were fitted for Ma ∈ [0.3, 0.8]. Below Ma = 0.3 the original model
predicts stall onset **too early** and under-predicts the post-stall `C_N` **overshoot**. The
modifications target exactly those two defects.

---

## 2. Nomenclature and conventions

| Symbol | Meaning | Notes |
|---|---|---|
| `V` | freestream velocity | m/s |
| `c`, `b` | chord, semi-chord | `b = c/2` |
| `α` | angle of attack | **degrees** — see the unit warning below |
| `α_E` | effective angle of attack | |
| `q` | non-dimensional pitch rate | `q = θ̇ c / V` (Eq. 25) |
| `s` | non-dimensional time | `s = Vt/b`, semi-chords travelled |
| `C_N`, `C_m`, `C_C` | normal force, pitching moment, chord force coefficients | |
| `C_L`, `C_D` | lift, drag coefficients | |
| `f` | trailing-edge separation point, fraction of chord | `f = 1` attached, `f = 0` fully separated |
| `h`, `θ` | plunge displacement, pitch angle | |
| `a_h` | elastic-axis offset from mid-chord | in semi-chords |
| `x_θ` | mass-centre offset from elastic axis | in semi-chords |
| `K_h`, `K_θ` | plunge, pitch spring stiffness | |
| `C_m0` | static pitching moment at zero AoA | **symmetry-breaking parameter** |

Superscripts on airload coefficients: `I` = attached (impulsive + circulatory) flow,
`f` = trailing-edge separation, `v` = dynamic-stall / vortex-induced.

> ### ⚠ Unit warning — degrees, not radians
> The paper gives `C_Nα = 0.108` and `α₁ = 15.25`. A lift-curve slope of 2π/rad ≈ **0.1097 /deg**,
> so `C_Nα = 0.108` is **per degree** and `α₁ = 15.25` is **degrees**. `S₁ = 3.0` and `S₂ = 2.3` in
> Eq. (11) are likewise in degrees. Every angle fed to `f(α, α₁)` (Eq. 11) and to `x9/C_Nα`
> (Eq. 12) must be in degrees. `θ` and `h` in the structural equations are naturally in radians/metres.
> Pick one internal convention and convert explicitly at the boundary — this is the single most
> likely source of a silently wrong implementation.

---

## 3. Model skeleton (Eqs. 1–7)

The aerodynamic model is a set of ODEs

```
Ẋ = f_u(X, α, q),      Ẋ = dX/dt                                    (1)
```

with airloads recovered algebraically from the state:

```
C_i = g_i(X, α, q),     i = N, m, C                                 (2)
```

Lift and drag from normal and chord force:

```
C_L = C_N cos α_E + C_C sin α_E                                     (3)
C_D = C_N sin α  − C_C cos α                                        (4)
```

> **Note the asymmetry:** Eq. (3) uses `α_E`, Eq. (4) uses `α`. Transcribed verbatim from the paper.
> This is very likely a typo — both should use the same angle (conventionally `α_E`). It only
> affects `C_D`, which is not used by the aeroelastic coupling (§8 uses `C_L` and `C_m` only), so it
> is harmless for flutter results but should be flagged in code.

Airload decomposition:

```
C_N = C_N^I + C_N^f + C_N^v                                         (5)
C_m = C_m^I + C_m^f + C_m^v + C_m0                                  (6)
C_C = C_C^f                                                         (7)
```

`C_m0` is the constant offset "used for the break of the symmetry of airfoil" — it is the paper's
bifurcation-relevant asymmetry knob (§10.3).

---

## 4. Attached flow — states `x1 … x8` (Eq. 8)

```
x̃̇ = A x̃ + B u,     u = [α  q]ᵀ,     x̃ = [x1 x2 … x8]ᵀ            (8)
```

`α_E`, `C_N^I`, `C_m^I` and `C_C^I` are all recovered from `x̃`.

`A` and `B` are not printed in the paper — it says only that they are "constant matrices from
Ref. [16]". They are given in full by **[LN]** Eqs. (17)–(21) and Appendices A–C; see §9.1.

---

## 5. Separated flow — states `x9`, `x10`

### 5.1 Leading-edge pressure lag (Eqs. 9–10)

A first-order lag applied to the potential-flow normal force `C_N^p` produces the substitute
value `C_N′`, carried by state `x9`:

```
ẋ9 = ( −x9/T_p + (C_N^C + C_N^I)/T_p ) · V/b                        (9)

C_N^p = C_N^C + C_N^I                                              (10)
```

where `C_N^C` is the circulatory normal force and `T_p = 1.7` (non-dimensional, in semi-chords).
Equivalently `ẋ9 = (C_N^p − x9)/T_p · V/b`.

### 5.2 Static Kirchhoff separation point (Eq. 11)

```
              ⎧ 1 − 0.3 · exp( (|α| − α₁) / S₁ )        |α| ≤ α₁
f(α, α₁)  =   ⎨                                                     (11)
              ⎩ 0.04 + 0.66 · exp( (α₁ − |α|) / S₂ )    |α| > α₁
```

with `α₁ = 15.25`, `S₁ = 3.0`, `S₂ = 2.3` (degrees; Mach-dependent, from Ref. [16]).

The function is written in `|α|` so it is even — this is what makes the *aerodynamics* symmetric,
and hence why `C_m0` is needed to break symmetry.

### 5.3 Unsteady trailing-edge separation point (Eq. 12)

```
ẋ10 = [ f(x9/C_Nα, α₁ₙ) − x10 ] / (σ₁ T_f) · V/b                    (12)
```

with `C_Nα = 0.108`, `T_f = 3.0`. `σ₁` is a "modification coefficient of time constant `T_f`" and
`α₁ₙ` is described only as "the function of `x10`".

Neither `σ₁` nor `α₁ₙ` is defined in the paper. Both come from **[CH]** — `σ₁` is a switch
table, not a constant. See §9.2.

`x10` is the delayed separation point, written `f″` in Eq. (18).

---

## 6. Dynamic stall and reattachment — states `x11`, `x12`, `x13`

### 6.1 Baseline onset criterion (Eq. 13)

```
|C_N′| > C_N1                                                       (13)
```

i.e. `|x9| > C_N1`. **This is the original L-B criterion, superseded at low Mach by Eq. (17)** — see §7.1.

### 6.2 Vortex convection clock (Eq. 14)

```
ẋ11 = V/b                                                           (14)
```

`x11` is non-dimensional time since stall onset, in semi-chords — the `τ` of Eqs. (19)–(20).
The paper never states the reset rule, and Eq. (14) is meaningless without one. **[CH]** supplies
it: `τ_v` starts at zero when `|x9|` reaches `C_N1` **while increasing**, i.e. on the rising edge
of the stall criterion.

### 6.3 Vortex-induced normal force (Eq. 15)

```
        ⎧ ċ_v − (x12 / (σ₂ T_v)) · V/b     if  α·ċ_v ≥ 0  and  0 < x11 < 2·T_vl
ẋ12 =  ⎨                                                            (15)
        ⎩      − (x12 / (σ₂ T_v)) · V/b     otherwise
```

with `T_v = 6.0`, `T_vl = 7.0`. `c_v` is "the strength of vortex induced normal force" and `σ₂` is
the modification coefficient on `T_v`. `x12` represents `C_N^v`.

`c_v` and `σ₂` are not defined in the paper; both come from **[CH]**. See §9.2.

The switch means: while the vortex is being fed (`α` and `ċ_v` same sign, i.e. loading still
growing) and the vortex is still over the airfoil (`x11 < 2·T_vl`), the state accumulates; otherwise
it only decays.

### 6.4 Flow reattachment (Eq. 16)

```
ẋ13 = [ f(α, α₁ₙ) − x13 ] / (0.5 · T_f)                             (16)
```

`x13` is the separation point during reattachment; the pitching moment is adjusted by `x13` during
that phase.

> **Dimensional inconsistency:** Eqs. (9), (12), (15), (17) all carry an explicit `V/b` factor
> converting `d/ds → d/dt`; Eq. (16) does not. Transcribed verbatim, but this is almost certainly a
> missing `· V/b` in the paper. Implement it **with** the factor (i.e. `ẋ13 = [f(α,α₁ₙ) − x13]/(0.5 T_f) · V/b`)
> and note the deviation.
>
> Note also the effective time constant here is `0.5·T_f`, i.e. reattachment is modelled as twice as
> fast as separation.

---

## 7. Low-Mach modifications — the paper's contribution

### 7.1 Delayed stall onset criterion (Eq. 17)

At low Mach an *additional* time lag is needed before the disturbed flow develops a vortex strong
enough to trigger stall. A second lag is applied to `C_N′`, producing `C_N″`, carried by `x14`:

```
ẋ14 = (C_N′ − x14) / T_b · V/b   =   (x9 − x14) / T_b · V/b         (17)
```

**Modified onset criterion:** dynamic stall initiates when

```
|C_N″| > C_N1        i.e.   |x14| > 1.75
```

with `T_b = 2.2` and `C_N1 = 1.75` (from Ref. [14], Galbraith et al.).

This replaces Eq. (13) and is what fixes "stall onset predicted too early".

### 7.2 Normal force and pitching moment overshoot (Eqs. 18–20)

At low Mach the separated shear layer stays attached over the upper surface, causing an extra
`C_N` overshoot. Following Beddoes [11], the increment is proportional to the difference between the
delayed and static separation points:

```
ΔC_N^v = B₁ (f″ − f) V_x                                            (18)
```

with `B₁ = 1.0`, `f″ = x10` (delayed separation point), `f` the static value from Eq. (11).

Vortex shape function:

```
        ⎧ sin^{3/2}( π τ / (2 T_v) )            0 < τ ≤ T_v
V_x =   ⎨                                                           (19)
        ⎩ cos²( π (τ − T_v) / T_vl )            τ > T_v
```

Additional pitching moment from vortex convection:

```
ΔC_m^v = B₂ ( 1 − cos(π τ_v / T_v) ) ΔC_N^v                         (20)
```

with `B₂ = 0.2`.

> **Two ambiguities to resolve:**
> 1. Eq. (19) uses `τ`, Eq. (20) uses `τ_v`. Almost certainly the same quantity (`x11`). Treat as identical unless evidence emerges otherwise.
> 2. The paper does **not** state how `ΔC_N^v` (Eq. 18) combines with `C_N^v` from state `x12`
>    (Eq. 15). Two readings are implemented, selected by `vortex_overshoot_mode`:
>    `"additive"` (`C_N^v = x12 + ΔC_N^v`, the default) and `"feed"` (`ΔC_N^v` is injected into
>    the vortex feed `c_v` and integrated by Eq. 15). **Both were tested and neither closes the
>    §10.1 gap** — they differ by under 2% in `C_N` peak. The ambiguity is therefore *not* the
>    cause of the residual discrepancy; see §9.3.

---

## 8. Structural model — states `x15 … x18`

2-DOF rigid airfoil on bending and torsional springs (Fung [18]):

```
m ḧ + S θ̈ + K_h h = Q_h                                            (21)
S ḧ + I_θ θ̈ + K_θ θ = Q_θ                                          (22)
```

- `m` — mass per unit length
- `S` — static mass moment (`S = m·b·x_θ`)
- `I_θ` — polar moment of inertia **about the 1/4 chord**
- `Q_h`, `Q_θ` — external aerodynamic force and moment

State vector (note the ordering — displacements first, then rates):

```
[x15  x16  x17  x18]ᵀ = [h  θ  ḣ  θ̇]ᵀ                              (23)
```

```
x̂̇ = A_s x̂ + B_s u_s                                               (24)

u_s = [ 0   0   −½ρV²c C_L   ½ρV²c² C_m ]ᵀ
```

So `Q_h = −½ρV²c·C_L` and `Q_θ = +½ρV²c²·C_m`. `A_s` and `B_s` are not printed, but follow directly
from Eqs. (21)–(23). With `M = [[m, S], [S, I_θ]]` and `K = diag(K_h, K_θ)`:

```
A_s = ⎡  0₂ₓ₂     I₂ₓ₂ ⎤          B_s = ⎡ 0₂ₓ₂   0₂ₓ₂  ⎤
      ⎣ −M⁻¹K     0₂ₓ₂ ⎦                ⎣ 0₂ₓ₂   M⁻¹   ⎦
```

which is consistent with `u_s` having two leading zeros. **No structural damping appears anywhere in
the paper** — the model is undamped, and all dissipation is aerodynamic.

### Aerodynamic–structural coupling (Eq. 25)

```
⎡ α ⎤   ⎡ θ + ḣ/V ⎤   ⎡ x16 + x17/V ⎤
⎢   ⎥ = ⎢         ⎥ = ⎢             ⎥                               (25)
⎣ q ⎦   ⎣  θ̇c/V   ⎦   ⎣  x18·c/V    ⎦
```

This closes the loop: structure → `(α, q)` → L-B model → `(C_L, C_m)` → structure.
**Total system: 18 states.**

---

## 9. Sources for what the paper omits

The paper defers roughly 40% of the model to its references. Everything is now recovered; this
section records the provenance so a future change to any coefficient can be traced to a source.

| Item | Paper's pointer | Recovered from | §  |
|---|---|---|---|
| `A`, `B` matrices (Eq. 8) and the output map | Ref. [16] | **[LN]** Eqs. 17–21, App. A–C | 9.1 |
| `σ₁` (Eq. 12), `α₁ₙ` (Eqs. 12, 16) | Ref. [16] | **[CH]** Table 2.3, Eqs. 2.54–2.55 | 9.2 |
| `c_v`, `σ₂` (Eq. 15) | Ref. [16] | **[CH]** Table 2.4, `c_v'` expression | 9.2 |
| `C_N^f`, `C_m^f`, `C_C^f` | "linear function of `x10`" | **[CH]** Eqs. 2.7–2.9, Table 2.1 | 9.2 |
| `x11` reset rule (Eq. 14) | not stated | **[CH]** §2.4 — rising edge of `\|x9\| = C_N1` | 9.2 |
| Newmark average-velocity integrator | Ref. [15] | implemented in `integrators.py` | 11 |

One item is **still outstanding** and blocks §10.3 only: the structural static mass moment `S`
(equivalently `x_θ`) and air density `ρ` for the flutter case. Neither appears in the paper; both
must come from Ref. [3] (Dimitriadis & Li). The aerodynamic model is unaffected.

### 9.1 Attached flow, from [LN]

**[LN]**'s state numbering `x1…x8` matches this paper's exactly, and its `A` is **diagonal** —
which is what lets Eq. (8) be written with constant matrices.

| state | driven by | role |
|---|---|---|
| `x1`, `x2` | `α_{3/4} = α + q/2` | circulatory (Wagner, 2-pole) |
| `x3` | `α` | non-circulatory normal force |
| `x4` | `q` | non-circulatory normal force |
| `x5`, `x6` | `α` | non-circulatory moment |
| `x7` | `q` | circulatory moment |
| `x8` | `q` | non-circulatory moment |

Their Eq. (17): `ẋ₁ = (2V/c)β²(−b₁)x₁ + α_{3/4}`, output `C_N^C = (2π/β)(2V/c)β²[A₁b₁ A₂b₂]x`.
`B` carries **no** `2V/c` factor, so the raw states are dimensional; we integrate the scaled
states `x̂ = x·(2V/c)`, which makes every equation velocity-independent in `s`.

Non-circulatory time constants (their Eqs. A2, A6, A12), with `T_I = c/a`:

```
K_α  = [(1−M) + πβM²(A₁b₁+A₂b₂)]⁻¹
K_q  = [(1−M) + 2πβM²(A₁b₁+A₂b₂)]⁻¹
K_αM = (A₃b₄ + A₄b₃) / (b₃b₄(1−M))       A₃=1.5, A₄=−0.5, b₃=0.25, b₄=0.1
K_qM = 7 / [15(1−M) + 3πβM²b₅]           b₅=0.5
```

> **⚠ Trap:** their closing remark states the non-circulatory constants "were reduced by 25% from
> their theoretical values." **0.75 is a multiplier on `K(M)`, not the value of `K`.** Reading it
> as the value gives `K_α = 0.75` where the correct figure at Ma = 0.12 is `0.75 × 1.113 = 0.835`.

Circulatory constants `A₁=0.3, A₂=0.7, b₁=0.14, b₂=0.53` (Beddoes), with `A₁+A₂ = 1` so
`φ(0) = 0` — the *compressible* convention, where the non-circulatory term supplies the initial
load. This differs from Jones/Wagner (`A₁=0.165, A₂=0.335`, `φ(0) = 0.5`), which is why **[LN]**'s
Eq. (10) carries an extra `0.5α_{3/4}` term and their Eq. (18) does not.

**Consequence — the model is stiff.** State time constants at Ma = 0.12 span `τ = 0.020` (x6) to
`7.25` (x1) semi-chords, a **stiffness ratio of ~354**. Explicit RK4 needs `Δs < 2.78τ_min`, i.e.
**> 891 steps/cycle** at `k = 0.124`. This is the concrete reason the paper reports explicit
integration as unstable and adopts implicit Newmark — §11 is not a stylistic preference.

### 9.2 Separated flow and dynamic stall, from [CH]

**Cross-check first.** His Table 2.1 gives NACA0012 constants vs Mach. The **M = 0.30 column is
exactly this paper's constant set**, with angles in radians rather than degrees:

| [CH] at M=0.30 | Value | This paper |
|---|---|---|
| `C_Nα^S` | 6.1879 /rad | 0.108 /deg ✅ |
| `α₁₀` | 0.2662 rad | 15.25° ✅ |
| `S₁` | 0.0524 rad | 3.0° ✅ |
| `S₂` | 0.0401 rad | 2.3° ✅ |
| `T_P`, `T_f0`, `T_v0`, `T_vl` | 1.7, 3.0, 6.0, 7.0 | identical ✅ |

That is an independent confirmation of the transcription *and* of the degrees/radians reading in §2.

Additional constants: `K₀ = 0.0025`, `K₁ = −0.135`, `K₂ = 0.04`, `δ_α1 = 0.0367 rad = 2.103°`,
`η = 0.97`, `m* = 2`. (`C_N1 = 1.45` at M = 0.30; this paper's 1.75 is the low-Mach replacement.)

**Separated-flow airloads** — his Eqs. (2.7)–(2.9):

```
C_N^f = C_Nα · ((1+√f)/2)² · α_E
C_C^f = η · C_Nα · α_E² · √f
C_m^f = [K₀ + K₁(1−f) + K₂ sin(π f^{m*})] · C_N^f
```

**`σ₁ = T_f/T_f0`** is a switch table on vortex time and loading direction (`a·a' = α·dα/ds`),
his Table 2.3:

| | `0 ≤ τ_v ≤ T_vl` | `T_vl < τ_v ≤ 2T_vl` | `2T_vl < τ_v` |
|---|---|---|---|
| `a·a' ≥ 0` | 1 | 1/3 | 4 |
| `a·a' < 0` | 1/2 | 1/2 | 4 |

Reattachment phase (his Eq. 2.55): 1 if `x10 ≥ 0.7`, else 1/2.

**`α₁ₙ`** (his Eq. 2.54): `α₁₀` when loading, `α₁₀ − (1−x10)^0.25 · δ_α1` when unloading.

**`σ₂ = T_v/T_v0`**, his Table 2.4:

| | `0 ≤ τ_v ≤ T_vl` | `T_vl < τ_v ≤ 2T_vl` | `2T_vl < τ_v` |
|---|---|---|---|
| `a·a' ≥ 0` | 1 | 0.25 | 0.90 |
| `a·a' < 0` | 0.50 | 0.50 | 0.90 |

Reattachment phase: 1.

**`c_v = C_N^C(1 − K_N)`** — his explicit `c_v'` expression matches the analytic derivative used
in `dynamic_stall.vortex_feed_rate` term-for-term, including the `(1+√f)/(4√f)` factor.

**`x11` reset** — `τ_v` starts at zero when `|x9|` reaches `C_N1` *while increasing*.

### 9.3 Status of the §10.1 discrepancy

The LB core is **verified independently** against **[CH]** Fig. 2.9 (§10.0), to within 3–5%.
Against this paper's own Fig. 3, however, `C_N` peak reads 1.78 vs 2.55.

Decomposition at the `C_N` peak (α = 24.2°, `f'' = 0.208`, `τ_v = 3.7`):

| term | value |
|---|---|
| `C_N^I` impulsive | +0.053 |
| `C_N^f` separated circulatory | +1.267 |
| `x12` vortex state (Eq. 15) | +0.348 |
| `ΔC_N^v` overshoot (Eq. 18) | +0.115 |
| **total** | **1.783** (paper 2.550) |

Two candidate explanations have been **tested and eliminated**:

1. **The Eq. (18) combination ambiguity.** Both readings are implemented
   (`vortex_overshoot_mode`): `"additive"` gives `C_N` = 1.783, `"feed"` gives 1.752. They differ
   by under 2% — the ambiguity is not the cause.
2. **Calibration of `B₁`.** A sweep shows `B₁ = 5` (5× the paper's value) reaches only
   `C_N` = 2.24, and drives `C_m` the *wrong way* (−0.30 against the needed −0.62).

Since the LB core reproduces an independently published result and neither reading nor
calibration closes the gap, the discrepancy is specific to this paper's low-Mach modification
(Eqs. 17–20) at deep-stall amplitude. Remaining suspects, untested:

- The `C_m^v` centre-of-pressure travel term — `C_m` is proportionally further off than `C_N`.
- The Eq. (15) feeding window `0 < x11 < 2T_vl` combined with `σ₂` may truncate accumulation.
- The paper's Figs. 3–4 may be plotted against a different reduced-frequency convention; `k`
  is stated but the convention is not (§10.1).

---

## 10. Validation targets

Three independent checks, in increasing order of difficulty. **Do them in this order** — the
aeroelastic result (§10.3) is meaningless if §10.1 fails.

### 10.0 LB core verification against [CH] Fig. 2.9 — **PASSING**

Run first. This exercises the LB model at **[CH]**'s baseline settings, *without* this paper's
low-Mach modification, and so separates "is our LB core right?" from "is our reading of Eqs. 17–20
right?" — a distinction §10.1 cannot make on its own.

| Parameter | Value |
|---|---|
| motion | `α = 10° + 9° sin(0.1 S)`, `S = 2Vt/c` |
| `k` | 0.1 (from `0.1·S = 0.2Vt/c` ⟹ `ω = 0.2V/c`) |
| Ma | 0.30 (we run 0.29; 0.30 is the model's stated limit) |
| `C_N1` | 1.45 — **[CH]**'s M=0.30 value, not this paper's 1.75 |

| Quantity | [CH] Fig. 2.9 | Model | Error |
|---|---|---|---|
| `C_N` peak | 1.95 | 2.009 | **+3.0%** |
| `C_m` min | −0.30 | −0.281 | **+6.3%** |
| peak location | α ≈ 18–19° | α ≈ 18.8° | ✅ |

Case: `models/cases/naca0012_thesis_fig29.yaml`. **[CH]** notes a known discrepancy on the
reattachment branch that is also present in Leishman's own published results — do not chase it.

### 10.1 NACA0012 forced-pitch airloads (Figs. 3–4)

| Parameter | Value |
|---|---|
| Ma | 0.12 |
| reduced frequency `k` | 0.124 |
| chord `c` | 0.55 m |
| motion | `α = 15° + 10° sin(ωt)` |

Compare `C_N` vs `α` and `C_m` vs `α` hysteresis loops against experimental data of Ref. [14]
(Galbraith et al., Glasgow University).

Expected qualitative features from Fig. 3: `C_N` rises to a peak of ≈ **2.55** near α ≈ 24–25°,
then drops sharply to ≈ 1.2; the downstroke branch lies well below the upstroke. From Fig. 4:
`C_m` stays near 0 until α ≈ 22°, then plunges to ≈ **−0.62** near α ≈ 25° before recovering.

The paper notes its own known defect here: predicted `C_N` is "a little higher than experimental
data at the low angle of attack of flow reattachment phase" — do not chase that discrepancy.

> `k` convention is not stated. If `k = ωb/V` with `V = Ma·a ≈ 40.8 m/s` and `b = 0.275 m`,
> then `ω ≈ 18.4 rad/s`. Confirm against Ref. [14] before trusting.

### 10.2 OA207 forced-pitch airloads (Figs. 5–6)

| Parameter | Value |
|---|---|
| motion | `α = 13° + 10° sin(ωt)` |
| `k` | 0.105 |
| `V` | 34.24 m/s |
| `Re` | 2.0 × 10⁶ |

Reference data from Ref. [19]. Note the empirical constants in §5–§6 are NACA0012-tuned; the paper
does not say what was changed for OA207 — expect this case to require its own constant set.

### 10.3 NACA0012 stall flutter bifurcation (Figs. 7–12, Table 1)

| Parameter | Value |
|---|---|
| `c` | 0.3 m |
| `m` | 16.67 kg/m |
| `K_h` | 30.5 N/mm |
| `K_θ` | 13.1 N·m/rad |
| `I_θ` | 0.31 kg/m² *(as printed)* |
| `C_m0` | 0, 0.005, 0.010 |

Elastic axis is at the **quarter chord** (implied by `I_θ` being defined about 1/4 chord), i.e.
`a_h = −0.5`. **`S`, `x_θ` and `ρ` are not given** — the one outstanding item (§9), to be
recovered from Ref. [3] (Dimitriadis & Li), which is the source of the experiment.

**Target — reproduce Table 1:**

| Behaviour | `C_m0`=0 | `C_m0`=0.005 | `C_m0`=0.010 | Experiment [3] |
|---|---|---|---|---|
| Decaying oscillation | V < 12 | V < 12 | V < 12 | V < 12 |
| Hopf bifurcation | V = 12 | V = 12 | V = 12 | 12 < V < 13 |
| Symmetric LCO | V ≥ 12 | V ≥ 12 | V ≥ 12 | 12 < V < 21.4 |
| Asymmetric LCO | V > 26 | V > 20 | V > 18.4 | — |

(velocities in m/s)

**Physics to reproduce:**

- The Hopf bifurcation at V = 12 m/s is **subcritical** — symmetric LCO appears *suddenly* with
  finite amplitude, not growing from zero. A supercritical-looking result means something is wrong.
- Symmetric LCO is caused by **dynamic stall**. Diagnostic: the `x10`–`x14` phase-plane trajectory
  (Fig. 12) crosses `|x14| = C_N1 = 1.75`, i.e. the Eq. (17) criterion fires.
  *(The paper's text says the trajectory crosses "`x10 = ±C_N1`" — a typo; `C_N1` is a normal-force
  threshold on `x14`, and `x10` is a separation point bounded in [0,1].)*
- Asymmetric LCO requires **both** dynamic stall **and** static divergence (Ref. [3] gives static
  divergence at V = 17.7 m/s) **and** airfoil asymmetry (`C_m0 ≠ 0`). It never appears in the
  symmetric case. Larger `C_m0` → asymmetric LCO onset at lower V.
- At V = 20 m/s with `C_m0` = 0.010, symmetric and asymmetric LCOs **coexist** — the result is
  initial-condition dependent. Bifurcation diagrams must be swept from multiple ICs to see this.
- `C_m0 = 0.010` gives the best match to experiment.

---

## 11. Numerical integration

The paper uses the **implicit Newmark average-velocity state-space method** (Ref. [15], Kim & Lee,
*JSV* 2005) and explicitly justifies it: explicit integration of this system is unstable/inaccurate.
Conclusion (1) of the paper reiterates that this integrator "can effectively solve the issue of the
numerical integration instability".

**Recommendation:** do not treat this as blocking. Start with a stiff implicit solver from
`scipy.integrate.solve_ivp` (`Radau` or `BDF`) to get physics correct, then implement Newmark
average-velocity if fidelity to the paper's numerics turns out to matter. Two cautions:

- The RHS has **hard switches** (Eq. 15's conditional, the stall-onset trigger, the `V_x`/`f`
  branch points). Adaptive implicit solvers will struggle across discontinuities. Either use
  event detection to break integration at switch points, or accept fixed-step Newmark.
- The `x11` clock reset (§9.2) is a discrete event, not an ODE — it needs explicit event handling
  regardless of integrator choice.

---

## 12. Suggested implementation order

1. ~~Recover the deferred formulation from [LN] and [CH]~~ — **done**, see §9.
2. **Static airloads:** implement Eq. (11) and the steady Kirchhoff reconstruction. Verify
   `C_N` vs `α` matches a static NACA0012 curve. Cheap, catches the degrees/radians bug early.
3. **Attached-flow module** (Eq. 8) in isolation. Verify indicial step response against Wagner.
4. **Full L-B with forced pitch** — §10.1 and §10.2. This is the real validation gate.
5. **Structural module** (Eqs. 21–25) in isolation. Verify uncoupled natural frequencies against
   `√(K_h/m)` and `√(K_θ/I_θ)`.
6. **Couple and sweep** — §10.3. Bifurcation diagram machinery: sweep `V`, sample LCO peaks after
   transient decay, sweep from multiple ICs to capture coexisting attractors.

Proposed layout:

```
src/twod_flutter/
    attached.py      # Eq. 8  — states x1..x8
    separation.py    # Eqs. 9-12, 16 — states x9, x10, x13
    dynamic_stall.py # Eqs. 13-15, 17-20 — states x11, x12, x14
    airloads.py      # Eqs. 2-7 — assembly
    structure.py     # Eqs. 21-25 — states x15..x18
    system.py        # 18-state coupled RHS
    integrate.py     # Newmark average-velocity + solve_ivp fallback
    constants.py     # tables from §13
    bifurcation.py   # V sweep, LCO detection, phase planes
tests/
    test_static.py       # step 2
    test_forced_pitch.py # §10.1, §10.2
    test_flutter.py      # §10.3 / Table 1
```

---

## 13. Constants

All Mach-number-dependent, taken from Ref. [16] unless noted, and tuned for NACA0012 at Ma < 0.3.

| Constant | Value | Eq. | Role |
|---|---|---|---|
| `T_p` | 1.7 | (9) | leading-edge pressure lag |
| `α₁` | 15.25 ° | (11) | static separation break point |
| `S₁` | 3.0 ° | (11) | attached-branch decay |
| `S₂` | 2.3 ° | (11) | separated-branch decay |
| `C_Nα` | 0.108 /deg | (12) | normal-force curve slope |
| `T_f` | 3.0 | (12), (16) | separation-point lag |
| `T_v` | 6.0 | (15), (19), (20) | vortex decay |
| `T_vl` | 7.0 | (15), (19) | vortex convection over chord |
| `T_b` | 2.2 | (17) | **low-Mach onset lag** (Ref. [14]) |
| `C_N1` | 1.75 | (17) | **low-Mach onset threshold** (Ref. [14]) |
| `B₁` | 1.0 | (18) | `C_N` overshoot magnitude |
| `B₂` | 0.2 | (20) | `C_m` overshoot magnitude |
| `σ₁` | switch table | (12) | §9.2, [CH] Table 2.3 |
| `σ₂` | switch table | (15) | §9.2, [CH] Table 2.4 |
| `δ_α1` | 2.103 ° | (12) | §9.2, [CH] Table 2.1 |
| `K₀`, `K₁`, `K₂` | 0.0025, −0.135, 0.04 | (6) | §9.2, [CH] Table 2.1 |
| `η`, `m*` | 0.97, 2 | (7) | §9.2, [CH] Eq. 2.8 |

All time constants are non-dimensional (semi-chords of travel).

---

## 14. Known defects in the source document

Collected for convenience; each is discussed in context above.

1. **Eq. (4)** uses `α` where Eq. (3) uses `α_E`. Affects `C_D` only.
2. **Eq. (16)** is missing the `· V/b` factor present in every other rate equation. Dimensionally inconsistent as printed.
3. **Eqs. (19) vs (20)** use `τ` and `τ_v` for what is almost certainly the same quantity.
4. **§5.2 text** states the phase-plane trajectory "crosses the vertical lines of `x10 = ±C_N1`"; it should be `x14`. Confirmed by the same paragraph correctly identifying `x14` as the lagged leading-edge pressure quantity.
5. **`I_θ` units** printed as `kg/m²`; dimensionally a polar moment per unit span is `kg·m²/m = kg·m`.
6. **No definition** of how `ΔC_N^v`/`ΔC_m^v` (Eqs. 18, 20) enter Eqs. (5)–(6) relative to `x12`.
7. **Ref. [3] citation** is given as "AIAA Journal 2009; 41(11)" — volume 41 is 2003, not 2009. Check the actual Dimitriadis & Li reference when sourcing the structural parameters.

---

## 15. References (as cited by the paper)

Key ones for gap-closing are marked ★.

- [3] Dimitriadis G, Li J. "Bifurcation behavior of airfoil undergoing stall flutter oscillations in low-speed wind tunnel." *AIAA Journal* 2009; 41(11): 2577–2596. ★ *(source of the §10.3 experiment and the outstanding structural parameters)*
- [10] Leishman J G, Crouse G L. "State-space model for unsteady airfoil behavior and dynamic stall." *Proc. 30th AIAA/ASME/ASCE/AHS/ASC Structures*, 1989; 1319–1330. *(superseded for our purposes by [LN] 1990, which prints the matrices in full)*
- [11] Beddoes T S. "A third generation model for unsteady aerodynamics and dynamic stall." Westland Limited RP-908, 1993. ★ *(basis of Eqs. 18–20)*
- [12] Galbraith R A McD, Coton F N. "A new stall-onset criterion for low speed dynamic-stall." *J. Solar Energy Engineering* 2006; 128(4): 461–471.
- [13] Galbraith R A McD, Coton F N. "Improved dynamic stall onset criterion at low Mach numbers." *J. Aircraft* 2007; 44(3): 1049–1052.
- [14] Galbraith R A McD, Coton F N. "A modified dynamic stall model for low Mach numbers." *J. Solar Energy Engineering* 2008; 130(3): 31013–31022. ★ *(source of `T_b`, `C_N1`, and §10.1 experimental data)*
- [15] Kim B O, Lee A S. "A transient response analysis in the state-space applying the average velocity concept." *J. Sound and Vibration* 2005; 281(3–5): 1023–1035. *(the Newmark average-velocity integrator)*
- [16] Bauchau O A. *DYMORE User's and Theory Manual.* Georgia Institute of Technology, 2007. *(the paper's pointer for the deferred formulation; never obtained — [LN] and [CH] supplied all of it instead)*
- [17] Chantharasenawong C. "Nonlinear aeroelastic behaviour of aerofoils under dynamic stall." PhD thesis, University of London, 2007. ★ *(likely contains the same L-B state-space formulation in full)*
- [18] Fung Y C. *An Introduction to the Theory of Aeroelasticity.* Chapman & Hall, 1993; 210–212. *(Eqs. 21–22)*
- [19] Tan J F. "Analysis of rotor aerodynamic response under manoeuvring conditions." MSc thesis, NUAA, 2009. [in Chinese] *(§10.2 OA207 data)*

> **Note:** Ref. [17] (Chantharasenawong's thesis) proved to be the decisive source — its §2.4–2.5
> reproduces the full L-B state-space formulation including the switch tables the paper omits, and
> it is freely available from the author. The DYMORE manual, which the paper actually points to,
> was never needed. Leishman's *Principles of Helicopter Aerodynamics* (Ch. 8–9) is the standard
> textbook treatment of the same material.

---

## Appendix A — 18-state vector summary

| State | Symbol | Eq. | Description |
|---|---|---|---|
| `x1`–`x8` | — | (8) | attached-flow indicial lag states → `α_E`, `C_N^I`, `C_m^I`, `C_C^I` |
| `x9` | `C_N′` | (9) | leading-edge pressure lag |
| `x10` | `f″` | (12) | delayed trailing-edge separation point |
| `x11` | `τ` | (14) | vortex convection clock (semi-chords since onset) |
| `x12` | `C_N^v` | (15) | vortex-induced normal force |
| `x13` | — | (16) | separation point during reattachment (moment correction) |
| `x14` | `C_N″` | (17) | **low-Mach doubly-lagged pressure — stall trigger** |
| `x15` | `h` | (23) | plunge displacement |
| `x16` | `θ` | (23) | pitch angle |
| `x17` | `ḣ` | (23) | plunge rate |
| `x18` | `θ̇` | (23) | pitch rate |

Note the structural ordering is displacements-then-rates, so `α = x16 + x17/V` and `q = x18·c/V`.
