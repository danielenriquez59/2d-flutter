"""Plot helpers. Imports matplotlib lazily so the core model stays dependency-light."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .forced_pitch import ForcedPitchResult


def plot_forced_pitch(
    result: ForcedPitchResult,
    out_path: str | Path,
    *,
    title: str = "",
    reference: dict | None = None,
) -> Path:
    """Airload hysteresis loops, in the layout of the paper's Figs. 3-4."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cyc = result.last_cycle()
    a = cyc.alpha_deg
    ref = reference or {}

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))

    ax = axes[0]
    ax.plot(a, cyc.c_n, color="tab:red", lw=1.6, label="Calculation")
    if "c_n_peak" in ref:
        ax.axhline(ref["c_n_peak"], ls=":", c="gray", lw=1)
        ax.annotate(
            f"paper peak {ref['c_n_peak']}",
            xy=(a.min(), ref["c_n_peak"]),
            fontsize=8,
            color="gray",
            va="bottom",
        )
    ax.set_xlabel(r"$\alpha$ / deg")
    ax.set_ylabel(r"$C_N$")
    ax.set_title("Normal force (cf. Fig. 3)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.plot(a, cyc.c_m, color="tab:blue", lw=1.6)
    if "c_m_min" in ref:
        ax.axhline(ref["c_m_min"], ls=":", c="gray", lw=1)
        ax.annotate(
            f"paper min {ref['c_m_min']}",
            xy=(a.min(), ref["c_m_min"]),
            fontsize=8,
            color="gray",
            va="bottom",
        )
    ax.set_xlabel(r"$\alpha$ / deg")
    ax.set_ylabel(r"$C_m$")
    ax.set_title("Pitching moment (cf. Fig. 4)")
    ax.grid(alpha=0.3)

    # Diagnostic: the stall trigger. x14 crossing +/-C_N1 is what fires the
    # modified low-Mach criterion (Eq. 17).
    ax = axes[2]
    ax.plot(a, cyc.x14, color="tab:green", lw=1.4, label=r"$x_{14}=C_N''$")
    ax.plot(a, cyc.x9, color="tab:orange", lw=1.0, ls="--", label=r"$x_9=C_N'$")
    if "C_N1" in ref:
        for sign in (1, -1):
            ax.axhline(sign * ref["C_N1"], ls=":", c="k", lw=1)
    ax.set_xlabel(r"$\alpha$ / deg")
    ax.set_ylabel("lagged normal force")
    ax.set_title("Stall trigger (Eq. 17)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    if title:
        fig.suptitle(title)
    fig.tight_layout()

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path
