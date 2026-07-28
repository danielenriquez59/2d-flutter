#!/usr/bin/env python3
"""Sweep freestream velocity and reproduce the paper's Table 1 (theory.md §10.3).

    python scripts/run_bifurcation.py models/cases/naca0012_flutter.yaml \
        --vmin 8 --vmax 26 --dv 2 --cm0 0.010
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np  # noqa: E402

from twod_flutter.analysis import bifurcation  # noqa: E402
from twod_flutter.config import Case  # noqa: E402
from twod_flutter.structure import StructuralModel  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("case")
    p.add_argument("--vmin", type=float, default=8.0)
    p.add_argument("--vmax", type=float, default=26.0)
    p.add_argument("--dv", type=float, default=2.0)
    p.add_argument("--cm0", type=float, default=None, help="override C_m0")
    p.add_argument("--t-end", type=float, default=20.0)
    p.add_argument("--steps-per-period", type=int, default=4000)
    p.add_argument("--disturbances", type=float, nargs="+", default=[1.0, 8.0, 15.0])
    p.add_argument("--out", default=None, help="CSV of every branch found")
    args = p.parse_args()

    case = Case.from_yaml(args.case)
    if args.cm0 is not None:
        case.airfoil.C_m0 = args.cm0

    sm = StructuralModel(case.structure)
    wh, wt = sm.uncoupled_frequencies
    print(f"case      : {case.name}   C_m0 = {case.airfoil.C_m0}")
    print(f"structure : omega_h = {wh:.3f} rad/s, omega_theta = {wt:.3f} rad/s")
    print(f"sweep     : V = {args.vmin}..{args.vmax} step {args.dv}, "
          f"disturbances {args.disturbances} deg, t_end {args.t_end}s")
    print(flush=True)

    vs = np.arange(args.vmin, args.vmax + 1e-9, args.dv)
    res = bifurcation.sweep(
        case, vs,
        disturbances_deg=tuple(args.disturbances),
        t_end=args.t_end,
        steps_per_period=args.steps_per_period,
        progress=True,
    )

    print()
    print(bifurcation.compare_to_table_1(res, case.airfoil.C_m0))

    if any(b.out_of_range for b in res.branches):
        print("\n  !! some runs exceeded |alpha| = 60 deg, outside the Kirchhoff"
              "\n     reconstruction's valid range (theory.md §15.5)")

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w") as fh:
            fh.write("V,theta0_deg,kind,amplitude_deg,offset_deg\n")
            for b in res.branches:
                fh.write(
                    f"{b.V},{b.theta0_deg},{b.kind},"
                    f"{b.amplitude_deg},{b.offset_deg}\n"
                )
        print(f"\ncsv       : {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
