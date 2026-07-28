# Theory Reference — Modified Leishman–Beddoes Stall Flutter Model

Working reference for reimplementing the airfoil aeroelastic stall-flutter model in Python.

**Source:** Shao Song, Zhu Qinghua, Zhang Chenglin, Ni Xianping,
"Airfoil Aeroelastic Flutter Analysis Based on Modified Leishman-Beddoes Model at Low Mach Number,"
*Chinese Journal of Aeronautics* **24**(5), 2011, pp. 550–557.
DOI [10.1016/S1000-9361(11)60064-2](https://doi.org/10.1016/S1000-9361(11)60064-2). Open access (CC BY-NC-ND).

Equation numbers below match the paper. Everything in §2–§7 is transcribed from the paper.
Everything marked **[GAP]** is *not* in the paper and must be sourced elsewhere before the model can run —
see §9, which is the single most important section for planning implementation work.

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

**[GAP-1] `A` and `B` are not printed in the paper.** It says only "`A` and `B` are constant matrices
from Ref. [16]" (Bauchau, *DYMORE User's and Theory Manual*, Georgia Tech, 2007). The output map
`x̃ → (α_E, C_N^I, C_m^I)` is likewise not given. See §9.

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

**[GAP-2] Neither `σ₁` nor `α₁ₙ` is defined in the paper.** See §9.

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

`x11` is non-dimensional time since stall onset, in semi-chords. It is the `τ` appearing in
Eqs. (19)–(20). It must be **reset to zero at each stall onset event** and held/frozen otherwise —
the paper does not state the reset logic explicitly, but Eq. (14) is meaningless without it.

### 6.3 Vortex-induced normal force (Eq. 15)

```
        ⎧ ċ_v − (x12 / (σ₂ T_v)) · V/b     if  α·ċ_v ≥ 0  and  0 < x11 < 2·T_vl
ẋ12 =  ⎨                                                            (15)
        ⎩      − (x12 / (σ₂ T_v)) · V/b     otherwise
```

with `T_v = 6.0`, `T_vl = 7.0`. `c_v` is "the strength of vortex induced normal force" and `σ₂` is
the modification coefficient on `T_v`. `x12` represents `C_N^v`.

**[GAP-3] `c_v` and `σ₂` are not defined in the paper.** See §9.

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
> 2. The paper does **not** state how `ΔC_N^v` (Eq. 18) combines with `C_N^v` from state `x12` (Eq. 15) in the assembly of Eq. (5) — additive (`C_N^v = x12 + ΔC_N^v`) or replacement. Additive is the reading most consistent with the word "overshoot"/"additional", and is the recommended starting assumption; validate against Fig. 3 (§10.1). Same question for `ΔC_m^v` vs `C_m^v`.

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

## 9. Gap register — what must be sourced before anything runs

This is the critical planning output. The paper is **not self-contained**; it is a delta on top of
Refs. [10], [11–14] and [16]. Roughly 40% of the model is by reference.

| # | Missing | Paper's pointer | Impact |
|---|---|---|---|
| ~~**GAP-1**~~ | ~~`A`, `B` matrices (Eq. 8)~~ | **CLOSED** — see §9.1 | ✅ resolved |
| **GAP-2** | `σ₁` (Eq. 12) and `α₁ₙ` (Eqs. 12, 16) | Ref. [16] | **Blocking.** Controls separation dynamics. |
| **GAP-3** | `c_v` and `σ₂` (Eq. 15) | Ref. [16] | **Blocking.** Controls vortex lift. |
| **GAP-4** | `C_N^f`, `C_m^f`, `C_C^f` as functions of `x10`/`x13` | stated only as "linear function of `x10`" | **Blocking.** Separated-flow airloads. |
| **GAP-5** | `x11` reset logic at stall onset | not stated | **Blocking.** Eq. (14) is incomplete without it. |
| **GAP-6** | Newmark average-velocity state-space integrator | Ref. [15] Kim & Lee 2005 | Non-blocking — see §11. |
| **GAP-7** | `S` (or `x_θ`), `a_h`, `ρ`, and structural ICs for the flutter case | not given | **Blocking for §10.3** only. |

### 9.1 GAP-1: CLOSED

Source: **Leishman J.G., Nguyen K.Q., "State-Space Representation of Unsteady Airfoil
Behavior," *AIAA Journal* 28(5), 1990, 836–844.** Its state numbering `x1…x8` matches this
paper's exactly, and its `A` is **diagonal** — which is what lets Eq. (8) be written with
constant matrices.

| state | driven by | role |
|---|---|---|
| `x1`, `x2` | `α_{3/4} = α + q/2` | circulatory (Wagner, 2-pole) |
| `x3` | `α` | non-circulatory normal force |
| `x4` | `q` | non-circulatory normal force |
| `x5`, `x6` | `α` | non-circulatory moment |
| `x7` | `q` | circulatory moment |
| `x8` | `q` | non-circulatory moment |

Their Eq. (17): `ẋ₁ = (2V/c)β²(−b₁)x₁ + α_{3/4}`, output `C_N^C = (2π/β)(2V/c)β²[A₁b₁ A₂b₂]x`.
Note `B` carries **no** `2V/c` factor, so the raw states are dimensional; we integrate the
scaled states `x̂ = x·(2V/c)`, which makes every equation velocity-independent in `s`.

Non-circulatory time constants (their Eqs. A2, A6, A12), all with `T_I = c/a`:

```
K_α  = [(1−M) + πβM²(A₁b₁+A₂b₂)]⁻¹
K_q  = [(1−M) + 2πβM²(A₁b₁+A₂b₂)]⁻¹
K_αM = (A₃b₄ + A₄b₃) / (b₃b₄(1−M))       A₃=1.5, A₄=−0.5, b₃=0.25, b₄=0.1
K_qM = 7 / [15(1−M) + 3πβM²b₅]           b₅=0.5
```

> **⚠ Trap:** their closing remark states the non-circulatory constants "were reduced by 25%
> from their theoretical values." So **0.75 is a multiplier on `K(M)`, not the value of `K`**.
> Reading it as the value gives `K_α = 0.75` where the correct figure at Ma = 0.12 is
> `0.75 × 1.113 = 0.835`.

Circulatory constants confirmed as `A₁=0.3, A₂=0.7, b₁=0.14, b₂=0.53` (Beddoes), with
`A₁+A₂ = 1` so `φ(0) = 0` — the *compressible* convention, where the non-circulatory term
supplies the initial load. This differs from Jones/Wagner (`A₁=0.165, A₂=0.335`, `φ(0) = 0.5`),
which is why their Eq. (10) carries an extra `0.5α_{3/4}` term and Eq. (18) does not.

**Consequence — the model is stiff.** With the correct constants the state time constants at
Ma = 0.12 span `τ = 0.020` (x6) to `7.25` (x1) semi-chords, a **stiffness ratio of ~354**.
Explicit RK4 then needs `Δs < 2.78τ_min = 0.057`, i.e. **> 891 steps/cycle** at `k = 0.124`.
This is the concrete reason the source paper reports explicit integration as unstable and
adopts implicit Newmark (Ref. [15]) — §11 is not a stylistic preference.

**Empirical result:** closing GAP-1 changed the §10.1 validation by <0.2% (`C_N` peak
1.917 → 1.920). The attached-flow model is therefore **not** the source of the remaining
magnitude error; that lies in GAP-3/GAP-4 (the vortex and separated-flow loads).

### Canonical forms to verify against Refs. [10]/[16]

The following are the standard L-B expressions. They are **not from this paper** and must be
confirmed against the DYMORE manual before being trusted — recorded here as the starting hypothesis
so the gap work has a concrete target to check:

- **Kirchhoff normal force:** `C_N^f = C_Nα · ((1 + √f″)/2)² · α_E`
- **Chord force:** `C_C^f = η · C_Nα · α_E² · √f″`
- **Moment:** `C_m^f = (K₀ + K₁(1 − f″) + K₂ sin(π f″^{k})) · C_N^f` (Beddoes form; `K₀,K₁,K₂` airfoil constants)
- **Vortex feed:** `c_v = C_N^C · [1 − ((1 + √f″)/2)²]` — the difference between attached-flow and Kirchhoff normal force, i.e. the lift "missing" due to separation, which is what rolls up into the vortex
- **Attached-flow circulatory indicial response:** Wagner-type two-lag approximation
  `φ(s) = 1 − A₁e^{−b₁s} − A₂e^{−b₂s}`, typically `A₁ = 0.3, A₂ = 0.7, b₁ = 0.14, b₂ = 0.53`
- **Attached-flow non-circulatory:** added-mass / Theodorsen terms in the incompressible limit
  (the usual compressible `4/M`-scaled deficiency functions degenerate as `M → 0`, so the low-Mach
  limit needs care — this is a known trap at Ma = 0.12)
- **`σ₁`, `σ₂`:** in Beddoes' formulation these are *switches*, not constants — they take different
  values depending on whether flow is separating vs. reattaching and whether the airfoil is stalled.
  Expect a small lookup table, not a scalar.

---

## 10. Validation targets

Three independent checks, in increasing order of difficulty. **Do them in this order** — the
aeroelastic result (§10.3) is meaningless if §10.1 fails.

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
`a_h = −0.5`. **`S`, `x_θ` and `ρ` are not given (GAP-7)** — they must be recovered from Ref. [3]
(Dimitriadis & Li), which is the source of the experiment.

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
- The `x11` clock reset (GAP-5) is a discrete event, not an ODE — it needs explicit event handling
  regardless of integrator choice.

---

## 12. Suggested implementation order

1. **Close GAP-1 through GAP-4** from the DYMORE manual (Ref. [16]) and Leishman & Crouse (Ref. [10]).
   Nothing runs until this is done. Record every recovered expression back into this document.
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
| `σ₁` | **undefined** | (12) | GAP-2 |
| `σ₂` | **undefined** | (15) | GAP-3 |

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
7. **Ref. [3] citation** is given as "AIAA Journal 2009; 41(11)" — volume 41 is 2003, not 2009. Check the actual Dimitriadis & Li reference when sourcing GAP-7.

---

## 15. References (as cited by the paper)

Key ones for gap-closing are marked ★.

- [3] Dimitriadis G, Li J. "Bifurcation behavior of airfoil undergoing stall flutter oscillations in low-speed wind tunnel." *AIAA Journal* 2009; 41(11): 2577–2596. ★ *(source of the §10.3 experiment and GAP-7)*
- [10] Leishman J G, Crouse G L. "State-space model for unsteady airfoil behavior and dynamic stall." *Proc. 30th AIAA/ASME/ASCE/AHS/ASC Structures*, 1989; 1319–1330. ★ *(GAP-1)*
- [11] Beddoes T S. "A third generation model for unsteady aerodynamics and dynamic stall." Westland Limited RP-908, 1993. ★ *(basis of Eqs. 18–20)*
- [12] Galbraith R A McD, Coton F N. "A new stall-onset criterion for low speed dynamic-stall." *J. Solar Energy Engineering* 2006; 128(4): 461–471.
- [13] Galbraith R A McD, Coton F N. "Improved dynamic stall onset criterion at low Mach numbers." *J. Aircraft* 2007; 44(3): 1049–1052.
- [14] Galbraith R A McD, Coton F N. "A modified dynamic stall model for low Mach numbers." *J. Solar Energy Engineering* 2008; 130(3): 31013–31022. ★ *(source of `T_b`, `C_N1`, and §10.1 experimental data)*
- [15] Kim B O, Lee A S. "A transient response analysis in the state-space applying the average velocity concept." *J. Sound and Vibration* 2005; 281(3–5): 1023–1035. *(GAP-6, integrator)*
- [16] Bauchau O A. *DYMORE User's and Theory Manual.* Georgia Institute of Technology, 2007. ★★ *(GAP-1, 2, 3, 4 — the single most important missing source)*
- [17] Chantharasenawong C. "Nonlinear aeroelastic behaviour of aerofoils under dynamic stall." PhD thesis, University of London, 2007. ★ *(likely contains the same L-B state-space formulation in full)*
- [18] Fung Y C. *An Introduction to the Theory of Aeroelasticity.* Chapman & Hall, 1993; 210–212. *(Eqs. 21–22)*
- [19] Tan J F. "Analysis of rotor aerodynamic response under manoeuvring conditions." MSc thesis, NUAA, 2009. [in Chinese] *(§10.2 OA207 data)*

> **Practical note on GAP closure:** Ref. [17] (Chantharasenawong's thesis) is likely the most
> productive single source — PhD theses on exactly this topic normally reproduce the full L-B
> state-space equations in an appendix, whereas the DYMORE manual is a code manual and may be
> harder to obtain. Leishman's *Principles of Helicopter Aerodynamics* (Ch. 8–9) is the standard
> textbook treatment and covers most of GAP-1 through GAP-4.

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
