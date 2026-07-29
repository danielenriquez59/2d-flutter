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

**Before extending this model to a new airfoil, Reynolds number, or amplitude, read §15 —
limitations of the method.** In particular, §15.4 bears directly on how much weight the
bifurcation conclusions in §10.3 can carry.

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

#### The elastic axis, and a gap in Eq. (24)

`I_θ` is defined about the quarter chord, which implies the elastic axis sits there
(`a_h = −0.5`). But that placement puts the elastic axis **directly under the lift**, so the
lift has zero moment arm about it and the section can never diverge — while the paper quotes a
static divergence velocity of **17.7 m/s** from its own Ref. [3].

The two statements are incompatible, and the resolution identifies a missing term. Eq. (24)
gives the pitching moment as `Q_θ = ½ρV²c²·C_m` alone, with `C_m` taken about the quarter
chord. If the elastic axis is *not* at the quarter chord, the lift contributes an additional
moment about it, and Eq. (24) is missing that transfer:

```
Q_θ = ½ρV²c²·C_m  +  ½ρV²c·C_L·d,      d = b(a_h + 0.5)
```

With `a_h = −0.5` the added term vanishes and Eq. (24) is recovered exactly, so this is a
strict generalisation rather than a change of model.

**This recovers one of the two missing structural parameters from a number the paper does
state.** Inverting the linear divergence balance `K_θ = ½ρV²c·C_Nα·d` at V = 17.7 m/s:

```
d    = 13.1 / (½ × 1.225 × 17.7² × 0.3 × 6.1879) = 0.03677 m = 0.1226 c
x_ea = 0.25c + 0.1226c = 0.373 c
a_h  = d/b − 0.5 = −0.2548
```

Linearising the full 18-state system with that value returns divergence at **17.70 m/s**,
confirming the calibration end to end rather than only through the closed-form balance.

**Still outstanding:** the static mass moment `S` (equivalently `x_θ`) and the density `ρ`
(assumed 1.225 kg/m³). `x_θ = 0` puts the mass centre on the elastic axis, leaving the two
degrees of freedom coupled only aerodynamically — which is legitimate for a stall-flutter rig,
since stall flutter is fundamentally a torsional instability, but it is an assumption.

#### No Hopf bifurcation is present in the linearised system

Linearising the coupled system about the equilibrium at each velocity gives, for
`C_m0 = 0`:

| V (m/s) | max Re(λ) | f (Hz) |
|---|---|---|
| 10 | −0.106 | 6.80 |
| 14 | −0.206 | 6.81 |
| 17 | −0.290 | 6.81 |
| 18 | **+1.070** | **0.00** |

Both oscillatory modes stay damped — and damping *increases* monotonically with V — right up
to 17.7 m/s, where a **real** eigenvalue crosses zero. That is static divergence, not a Hopf.

This is physically consistent: the two modes sit at 6.81 Hz (plunge) and 1.02 Hz (torsion), a
ratio of 6.6, and they never coalesce, so classical bending–torsion flutter is impossible for
this rig. It is also consistent with stall flutter being a genuinely *nonlinear* phenomenon —
the negative aerodynamic damping that drives it appears only once the section oscillates into
stall, which cannot happen at the linearised equilibrium where the flow is fully attached.

**The paper, however, reports a subcritical Hopf at V = 12 m/s** — an oscillatory instability
of the equilibrium. Our model does not reproduce it.

#### 10.3.1 Nonlinear sweep result

`scripts/run_bifurcation.py`, `C_m0 = 0.010`, disturbances of 1° and 20°, 15 s per point:

| V (m/s) | θ₀ = 1° | θ₀ = 20° | LCO amplitude (deg p-p) | equilibrium offset (deg) |
|---|---|---|---|---|
| 10 | decays | **LCO** | 107.6 | +0.36 |
| 12 | decays | **LCO** | 102.1 | +0.65 |
| 14 | decays | **LCO** | 94.6 | +1.31 |
| 16 | decays | **LCO** | 89.6 | +3.81 |
| 18 | **LCO** | **LCO** | 85.8 | — |
| 20 | **LCO** | **LCO** | 83.4 | — |
| 26 | **LCO** | **LCO** | 79.7 | — |

**What matches.** Three qualitative features of the paper come out correctly:

1. **Subcriticality is reproduced.** Below 18 m/s a 1° disturbance decays while a 20° one settles
   onto a large-amplitude limit cycle. A stable equilibrium coexisting with a stable LCO is
   exactly the signature of a subcritical bifurcation, and it is why a single-initial-condition
   sweep would have found nothing.
2. **The transition at 18 m/s** — where even a 1° disturbance grows — coincides with the
   divergence speed of 17.7 m/s derived above. Above it the equilibrium is unstable, so every
   trajectory departs.
3. **The static offset grows with velocity and blows up at divergence**: +0.36° at V = 10 rising
   to +3.81° at V = 16, consistent with `C_m0 = 0.010` acting against a pitch stiffness that the
   aerodynamics is progressively cancelling.

**What does not match.**

| | Paper | Model |
|---|---|---|
| Symmetric LCO onset | 12 m/s (Hopf) | ≤ 10 m/s, no Hopf |
| Mechanism of onset | subcritical Hopf, dynamic stall | finite-amplitude only; divergence at 17.7 |
| Asymmetric LCO onset (`C_m0`=0.010) | 18.4 m/s | not observed |
| LCO amplitude | ~40° p-p (Fig. 10) | 80–108° p-p |

**Amplitude is over-predicted by roughly 2.5×, and this is very likely inherited from §10.1.**
The pitching moment there is 48% low — a larger relative error than `C_N`. `C_m` *is* the pitch
damping: LCO amplitude is set by where positive and negative aerodynamic work over the cycle
balance, so under-predicting the restoring moment lets the cycle grow further before it closes.
An amplitude error of this size is the expected consequence, which makes the flutter result a
*downstream* symptom of the unresolved §9.3 discrepancy rather than an independent defect.

The absent asymmetric LCO is plausibly the same thing: our offset is ~1–2° on an 80–108° cycle,
a ratio of 0.02. If the amplitude were correct at ~40°, the same offset would be a far larger
fraction and might well cross into the asymmetric classification.

**Reading the amplitude with caution.** At 80–108° peak-to-peak the section reaches |α| ≈ 40–54°,
deep in the asymptotic tail of the Kirchhoff correlation (`f → 0.04`) and close to its ~60°
validity limit (§15.5). Those excursions are inside the letter of the model's range but well
outside where its constants were fitted, so the amplitudes should be read as "large" rather than
as quantitative predictions.

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

#### 10.3.2 The primary source: Dimitriadis & Li (Ref. [3])

Obtaining Ref. [3] — *Bifurcation Behavior of Airfoil Undergoing Stall Flutter Oscillations in
Low-Speed Wind Tunnel*, AIAA Journal **47**(11), 2009 — settles most of the open structural
questions and **refutes one of our working hypotheses**.

**Table 1 gives the rig directly:**

| Parameter | Value |
|---|---|
| Wing span | 0.9 m |
| Chord | 0.3 m |
| **Pitch axis** | **0.115 m behind the leading edge** |
| Wing assembly mass | 15 kg |
| `I` about pitch axis | 0.31 kg·m² |
| Plunge stiffness | 30.5 N/mm |
| Pitch stiffness | 13.1 N·m/rad |

**The elastic-axis derivation was right.** The pitch axis at 0.115 m is `0.3833c`, giving
`a_h = −0.2333`. Our value inferred by inverting the divergence balance against the 17.7 m/s
Shao quotes was `−0.2548` — **within 3%**, from a completely independent route. It also confirms
the §10.3 finding that Eq. (24) must be missing the lift-transfer term, since the elastic axis is
demonstrably not at the quarter chord.

**A units inconsistency in Shao.** Table 1's quantities are **totals** for a 0.9 m span, while the
aerodynamic loads in Eqs. (21)–(22) are per unit span. Shao converts the *mass*
(15/0.9 = 16.67 kg/m) but leaves `I_θ`, `K_h` and `K_θ` at their total values — a mixed set.
Dividing all four by the span is self-consistent; it leaves the frequency ratio unchanged but
raises `ω_h` from 6.81 to 7.18 Hz.

**Divergence now matches the experiment.** With Table 1's pitch axis and per-span properties,
linear theory gives **17.89 m/s**. Dimitriadis & Li report that the second bifurcation — onto
*asymmetric* LCOs — occurs "at the system's static divergence airspeed", with the first
positive-pitch asymmetric LCOs at **17.8 m/s**. Our model's transition sits on it.

**Two independent confirmations of the structural model.** The paper states "the stiffness in
plunge is so much higher than the stiffness in pitch that there is no response at the plunge
natural frequency. Despite the fact that the system has 2 degrees of freedom, the response is
single mode." That is exactly our frequency ratio of 6.9 and single-DOF torsional character. And
the support was "designed to allow pitch and plunge motion of the wing model **without friction**"
— so near-zero structural damping is the intended design, and Shao's omission of it is faithful.

##### The mean-incidence hypothesis is refuted

§10.3.2 previously proposed that the missing Hopf required a mean incidence near stall. **The
primary source rules this out.** At V = 12.2 m/s it states "the wing's equilibrium position
around zero is still stable", and at V = 13 m/s "the LCOs are self-activated … the equilibrium
position is no longer stable". The equilibrium is at zero pitch and it genuinely loses stability
at 13 m/s.

The experimental bifurcation sequence is:

| V (m/s) | Behaviour | Bifurcation |
|---|---|---|
| < 12 | decaying | — |
| 12.2 | decaying **or** LCO (needs impulse) | **fold** of limit cycles |
| 13.0 | LCO, self-activated | **subcritical Hopf** |
| 13–18 | symmetric LCO | — |
| 17.8 | asymmetric LCO appears | at the **divergence** airspeed |
| 23.7 | negative-pitch asymmetric LCO | — |

Against that, our model gives: a fold below 10 m/s (LCO reachable with a 20° impulse at V = 10),
no Hopf, and the divergence transition at 17.9 m/s. **So the divergence bifurcation is
reproduced, the fold is roughly 2 m/s low, and the Hopf is absent.**

##### What the Hopf actually requires

Dimitriadis & Li attribute the Hopf to "the occurrence of dynamic stall, which allows the
transfer of energy from the freestream to the wing". But a Hopf is by definition a *linear*
instability of the equilibrium, and at zero incidence with attached flow there is no stall to
drive it — a genuine tension in the experimental interpretation, not only in our model.

The resolution is most likely quantitative rather than structural: the torsional mode's
aerodynamic damping is small and negative in a narrow band, and whether it crosses zero at 13 m/s
depends on the *magnitude* of the unsteady pitching moment. Our `C_m` is **48% low** (§9.3) — and
`C_m` is precisely the pitch damping. An error of that size is more than enough to keep a
marginally negative damping term negative. **The missing Hopf and the §9.3 shortfall are very
likely the same defect**, which also explains the over-large LCO amplitudes.

##### The C_m hypothesis is REFUTED for the Hopf

§10.3.2 proposed that the missing Hopf was the §9.3 `C_m` shortfall: pitch damping is marginally
negative, `C_m` *is* pitch damping, and ours is 48% low. A diagnostic scale on the dynamic-stall
moment (`c_m_stall_scale`, multiplying `C_m^f + C_m^v`) tests it directly.

**Calibration.** The knob is cleanly separable — `C_N` peak is unchanged at 1.783 for every
scale, and `C_m` min moves linearly. Scale ≈ **1.92** reproduces the paper's `C_m` min = −0.62.

**Linear stability — no effect whatsoever:**

| `c_m_stall_scale` | oscillatory max Re at V = 13 | onset |
|---|---|---|
| 1.00 | −0.17174 | 18.0, divergence |
| 1.92 | −0.17171 | 18.0, divergence |
| 3.00 | −0.17168 | 17.5, divergence |
| 5.00 | −0.17163 | 17.5, divergence |
| 10.00 | −0.17150 | 16.5, divergence |

A **tenfold** increase in the dynamic-stall moment moves the eigenvalue by 0.014%, and never
produces a Hopf.

**Why, in hindsight.** At the equilibrium `α ≈ 0` the flow is fully attached: `f ≈ 0.998`, so
`C_m^f = [K₀ + K₁(1−f) + K₂ sin(π f²)]·C_N^f ≈ K₀·C_N^f ≈ 0`, and `C_m^v = 0` because no vortex
has been shed. **The deficient term is identically inactive at the linearisation point.** Scaling
zero gives zero. The §9.3 shortfall is a *deep-stall* error and cannot, even in principle, affect
a linear stability boundary at zero incidence.

This is the same structural reason the mean-incidence hypothesis was attractive — and it is why
its refutation by Ref. [3] leaves the Hopf genuinely unexplained rather than merely mis-attributed.

##### What this leaves

The Hopf and the amplitude errors are now **decoupled**, not one defect:

- **The missing Hopf** is not the `C_m` shortfall, not the mean incidence, not `x_θ`, not the
  Eq. (18) ambiguity, and not the pitch-axis factor. All five are tested and eliminated. Since our
  attached-flow model is independently validated (§10.0, 3–5%), the remaining possibilities are
  that the experimental Hopf is not a linear instability of the attached-flow equilibrium at all,
  or that it requires physics outside this model class.

  There is direct support for the first reading in Ref. [3] itself: it attributes the Hopf to
  "the occurrence of dynamic stall", which cannot act at `α ≈ 0`. A fold at 12.2 m/s whose stable
  equilibrium has a rapidly shrinking basin above it would look experimentally like a Hopf at
  13 m/s under hand-administered impulses and residual tunnel turbulence, without the equilibrium
  ever formally losing stability.

- **The amplitude and fold-velocity errors** remain plausibly the `C_m` shortfall, because there
  the section *does* stall and the deficient term is active. §10.3.3 tests this.

##### Remaining gaps, revised

| # | Gap | Status |
|---|---|---|
| 1 | `C_m` 48% low (§9.3) | active in deep stall only; **cannot** explain the Hopf |
| 1b | The Hopf itself | **unexplained**; five candidates eliminated |
| 2 | Mean angle of attack | **refuted** — equilibrium is at zero |
| 3 | Structural damping | **resolved** — frictionless by design |
| 4 | `a_h`, per-span scaling | **resolved** — Table 1 |
| 5 | `x_θ` | still unstated; shown not to affect onset |
| 6 | `ρ` | still assumed 1.225 |
| 7 | `α_{3/4}` pitch-axis factor | **fixed** — see below |


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

## 15. Limitations of the method

The model reproduces dynamic-stall airloads well (§10.0) but it is a semi-empirical correlation
with a bounded envelope, and several of its properties bear directly on how much weight the
flutter conclusions in §10.3 can carry. Read this before extending the model to a new airfoil,
Reynolds number, or amplitude.

### 15.1 It is a correlation, not a prediction from geometry

The Leishman–Beddoes model contains no geometric input. Thickness, camber, and leading-edge
radius enter *only* through fitted constants. The table in §13 is **NACA0012-specific**; the
constants come from static and dynamic wind-tunnel data for that section.

Consequences:

- The OA207 case (§10.2) needs its own constant set. The paper does not state what it changed,
  which makes that validation case unreproducible as published.
- Applying NACA0012 constants to any other section is unsupported, and will fail silently — the
  model will produce plausible-looking hysteresis loops that are simply wrong.

### 15.2 Reynolds number does not appear anywhere

Re is not an input to any equation in the formulation, yet dynamic-stall onset, vortex strength
and reattachment are all strongly Re-dependent.

The omission is masked in this paper because both validation cases sit at the same order —
Re ≈ 1.5 × 10⁶ (NACA0012, §10.1) and 2.0 × 10⁶ (OA207, §10.2). Extrapolating to low Re (small
UAVs, sub-scale wind-tunnel models, MAVs) is not supported by anything in this model; that regime
has its own modified-LB literature.

### 15.3 Constants are tabulated at Ma ≥ 0.30 but used at Ma = 0.12

The empirical tables (§13, and [CH] Table 2.1) **bottom out at Ma = 0.30**. The §10.1 validation
runs at Ma = 0.12 — well below the fitted range.

The paper's low-Mach modification replaces only **two** constants: `T_b = 2.2` and `C_N1 = 1.75`.
Everything else — `T_p`, `T_f`, `T_v`, `T_vl`, `α₁`, `S₁`, `S₂`, `C_Nα` — is the Ma = 0.30 column
used unchanged at a quarter of that Mach number.

This is a plausible contributor to the §10.1 magnitude shortfall, and it is **untested** (§9.3).
It also explains why §10.0 — which runs at the constants' native Ma = 0.30 — passes while §10.1
does not.

### 15.4 The model is non-smooth, and the paper's conclusions are about bifurcations

**This is the most consequential limitation for §10.3.** The right-hand side contains hard
switches, not just nonlinearities:

| Switch | Discontinuous in |
|---|---|
| `σ₁`, `σ₂` tables (§9.2) | `τ_v` at `T_vl`, `2T_vl`; and in `sign(α·α')` |
| stall onset `\|x14\| > C_N1` | the state itself |
| `α₁ₙ` droop (Eq. 2.54) | `sign(α·α')` |
| Kirchhoff `f` (Eq. 11) | gradient discontinuous at `\|α\| = α₁` |
| `V_x` (Eq. 19) | branch change at `τ = T_v` |
| Eq. (15) feeding condition | `α·ċ_v` sign and the `τ_v` window |

Two distinct consequences:

**(a) Numerical.** No classical smooth-ODE convergence theory applies. Adaptive step-size solvers
mis-handle the switches by construction, which is why integration here is fixed-step (§11). Our
event handling applies switches *between* steps, so the crossing instant is resolved only to
`O(Δs)` — accuracy near a switch is first-order regardless of the integrator's formal order.

**(b) Mathematical, and more serious.** Non-smooth dynamical systems admit bifurcations with **no
smooth analogue** — grazing, sliding, and other discontinuity-induced transitions. The paper
classifies its results in the language of smooth theory: a *subcritical Hopf bifurcation*, and
symmetric/asymmetric *limit cycles*. That classification presumes smoothness the model does not
have.

Some of the reported structure may therefore be an artefact of the switches rather than of the
physics. This is a documented concern for exactly this model — see Galvanetto, Peiró &
Chantharasenawong, "An assessment of some effects of the nonsmoothness of the Leishman–Beddoes
dynamic stall model on the nonlinear dynamics of a typical aerofoil section."

**Practical guidance:** treat §10.3's bifurcation *classification* as descriptive rather than as a
rigorous dynamical-systems claim. The velocities at which behaviour changes are meaningful and
worth reproducing; the labels attached to those transitions are less so. When reproducing the
bifurcation diagrams, check sensitivity to step size — a transition velocity that moves with `Δs`
is a numerical artefact of switch resolution, not a physical boundary.

### 15.5 The static Kirchhoff reconstruction has a limited range

- `f(α, α₁)` is written in `|α|` and is therefore **even**: it cannot represent camber. The paper
  bolts asymmetry back on through `C_m0`, which is a moment offset, not a genuine cambered-section
  model.
- Kirchhoff lift diverges from experiment beyond **α ≈ 60°** ([CH] Fig. 2.4, against Sheldahl &
  Klimas data); calculations should be confined to `|α| ≤ 60°`.
- §10.1 peaks at α = 25°, comfortably inside — but a growing flutter LCO can exit the valid range
  *silently*, since nothing in the formulation flags it. Worth an explicit runtime check when
  running §10.3 sweeps.

### 15.6 Reattachment is the weakest branch of the model

Both source documents concede this independently:

- **[CH]**: "A major discrepancy is observed during the reattachment period, but this is also
  present in the original numerical results reported by Leishman."
- **Shao**: predicted `C_N` is "a little higher than experimental data at the low angle of attack
  of flow reattachment phase", which the paper argues makes flutter analysis conservative.

That argument holds for a *flutter boundary* but not for *LCO amplitude*. LCO amplitude is set by
the balance of energy input and extraction integrated over the whole cycle, so an error confined
to the reattachment branch still biases both the amplitude and the velocities at which
bifurcations occur.

### 15.7 The structural model is deliberately minimal

- **2 DOF, rigid section, linear springs.** No chordwise flexibility, no higher modes, no
  geometric stiffening.
- **No structural damping at all** (`C_h = C_θ = 0`) — every dissipative mechanism in the model is
  aerodynamic. Real mounts have hysteretic and bearing damping, so predicted flutter onset will be
  earlier than reality and LCO amplitudes larger.
- **`α = θ + ḣ/V` (Eq. 25) is a small-angle, quasi-steady inflow relation.** It is a linearisation
  valid for `ḣ/V` small; at large LCO amplitude near stall that assumption is doing real work.
- **`q = θ̇c/V` carries pitch rate only.** Plunge acceleration's contribution to the 3/4-chord
  downwash is not carried as a separate input to Eq. (8).
- **Added mass enters only as a load,** through the impulsive terms, which are functions of `α`
  and `q`. Physically, added mass modifies the *inertia* of the coupled system — it belongs in the
  mass matrix. Representing it as an external load is an approximation that can shift computed
  flutter boundaries. (There is a dedicated literature on this: "Assessment of added mass effects
  on flutter boundaries using the Leishman–Beddoes dynamic stall model.")
- **Elastic axis at the quarter chord is inferred**, not stated — it follows only from `I_θ` being
  defined about `c/4` (§10.3).
- **Strictly 2D.** No finite span, no three-dimensional stall relief, no sweep — all of which
  delay and soften stall on a real wing or blade.

### 15.8 Implementation-specific caveats

Things true of *this codebase* rather than of the method, carried here so they are not forgotten:

| Item | Status |
|---|---|
| Eq. (16) | implemented **with** the `V/b` factor the paper omits (§14, defect 2) |
| Eq. (18) combination | ambiguous; both readings implemented, neither matches (§9.3) |
| Phase switching | keyed on `x14` (this paper's criterion); **[CH]** keys on `x9` — our choice |
| Vortex clock reset | inferred from **[CH]**, never stated by this paper |
| §10.1 magnitude | **unvalidated** — `C_N` 30% low, `C_m` 48% low; any flutter result inherits this |
| Newmark integrator | implemented but unused in validation runs (RK4 at ≥891 steps/cycle) |
| Structural model | defined in config only; coupling and §10.3 not yet implemented |

### 15.9 Validity envelope, at a glance

| Quantity | Supported range | Note |
|---|---|---|
| Mach | `< 0.3` (enforced in `config.Flow`) | constants themselves fitted at `≥ 0.30` (§15.3) |
| Reynolds | ~1–2 × 10⁶ as validated | **not a model input** (§15.2) |
| Angle of attack | `\|α\| ≤ 60°` | Kirchhoff reconstruction limit (§15.5) |
| Airfoil | NACA0012 only | constants are per-section (§15.1) |
| Structure | 2-DOF rigid, undamped, linear | (§15.7) |
| Flow | 2D, attached through deep stall | no finite span or 3D relief (§15.7) |

---

## 16. References (as cited by the paper)

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
