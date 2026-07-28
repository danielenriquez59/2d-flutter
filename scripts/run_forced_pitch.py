#!/usr/bin/env python3
"""Run a forced-pitch airload case and report against the paper's figures.

    python scripts/run_forced_pitch.py models/cases/naca0012_forced_pitch.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np  # noqa: E402

from twod_flutter.analysis.forced_pitch import run  # noqa: E402
from twod_flutter.config import Case  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("case", help="path to a case YAML under models/cases/")
    p.add_argument("--integrator", default="rk4", choices=["rk4", "newmark"])
    p.add_argument("--out", default=None, help="output PNG (default: results/<case>.png)")
    p.add_argument("--no-plot", action="store_true")
    p.add_argument("--csv", default=None, help="also write the last cycle to CSV")
    args = p.parse_args()

    case = Case.from_yaml(args.case)
    print(f"case      : {case.name}")
    print(f"airfoil   : {case.airfoil.name}")
    print(f"flow      : V = {case.flow.V:.2f} m/s, Ma = {case.flow.mach:.3f}")
    fp = case.forced_pitch
    print(
        f"motion    : alpha = {fp.alpha_mean_deg}deg + {fp.alpha_amp_deg}deg "
        f"sin(wt), k = {fp.k}"
    )

    result = run(case, integrator=args.integrator)
    cyc = result.last_cycle()

    print(f"\nintegrated: {fp.n_cycles} cycles x {fp.steps_per_cycle} steps")
    print(f"period    : {result.period * 1e3:.2f} ms")

    if not np.all(np.isfinite(cyc.c_n)):
        print("\n!! non-finite C_N -- the run diverged", file=sys.stderr)
        return 1

    i_peak = int(np.argmax(cyc.c_n))
    i_mmin = int(np.argmin(cyc.c_m))
    print("\n--- computed (last cycle) ---")
    print(f"  C_N peak : {cyc.c_n[i_peak]:+.3f} at alpha = {cyc.alpha_deg[i_peak]:.2f} deg")
    print(f"  C_N min  : {cyc.c_n.min():+.3f}")
    print(f"  C_m min  : {cyc.c_m[i_mmin]:+.3f} at alpha = {cyc.alpha_deg[i_mmin]:.2f} deg")
    print(f"  stalled  : {100.0 * cyc.stalled.mean():.1f}% of the cycle")

    ref = case.reference
    if ref:
        print("\n--- paper (Figs. 3-4) ---")
        rows = [
            ("C_N peak", ref.get("c_n_peak"), cyc.c_n[i_peak]),
            ("C_m min", ref.get("c_m_min"), cyc.c_m[i_mmin]),
        ]
        for label, expected, got in rows:
            if expected is None:
                continue
            err = 100.0 * (got - expected) / abs(expected)
            flag = "ok" if abs(err) < 15.0 else "OFF"
            print(f"  {label:9s}: paper {expected:+.3f}  model {got:+.3f}  ({err:+.1f}%) {flag}")

    if args.csv:
        out = Path(args.csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savetxt(
            out,
            np.column_stack([cyc.alpha_deg, cyc.c_n, cyc.c_m, cyc.c_c, cyc.f_delayed]),
            delimiter=",",
            header="alpha_deg,C_N,C_m,C_C,f_delayed",
            comments="",
        )
        print(f"\ncsv       : {out}")

    if not args.no_plot:
        from twod_flutter.analysis.plotting import plot_forced_pitch

        out = Path(args.out) if args.out else REPO_ROOT / "results" / f"{case.name}.png"
        ref_plot = dict(ref)
        ref_plot["C_N1"] = case.airfoil.C_N1
        path = plot_forced_pitch(
            result, out, title=f"{case.name} -- {case.airfoil.name}", reference=ref_plot
        )
        print(f"plot      : {path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
