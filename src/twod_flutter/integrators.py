"""Fixed-step time integration.

No SciPy: the right-hand side contains hard switches (Eq. 15's feeding
condition, the stall trigger, the branch point in ``V_x``), and adaptive
solvers mis-handle discontinuities by design -- they shrink the step trying to
resolve a jump that has no derivative to resolve. A fixed-step scheme with
events applied *between* steps is both simpler and better behaved here, and it
keeps the dependency footprint to NumPy.

The paper itself uses an implicit Newmark average-velocity scheme (Ref. [15])
and reports that explicit integration was unstable. RK4 at the step sizes used
here (~2000 steps/cycle) is well inside its stability limit for these time
constants; :func:`newmark_average_velocity` is provided for when fidelity to
the paper's numerics matters.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

RHS = Callable[[float, np.ndarray], np.ndarray]


def rk4_step(f: RHS, t: float, x: np.ndarray, dt: float) -> np.ndarray:
    """One classical fourth-order Runge-Kutta step."""
    k1 = f(t, x)
    k2 = f(t + 0.5 * dt, x + 0.5 * dt * k1)
    k3 = f(t + 0.5 * dt, x + 0.5 * dt * k2)
    k4 = f(t + dt, x + dt * k3)
    return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def newmark_average_velocity(
    f: RHS, t: float, x: np.ndarray, dt: float, *, tol: float = 1e-10, max_iter: int = 50
) -> np.ndarray:
    """One implicit average-velocity (trapezoidal) step, Ref. [15].

    Solves ``x_{n+1} = x_n + dt/2 * (f(t_n, x_n) + f(t_{n+1}, x_{n+1}))`` by
    fixed-point iteration. Unconditionally stable for linear problems, which is
    what the paper is relying on.
    """
    f_n = f(t, x)
    x_next = x + dt * f_n  # explicit Euler predictor
    for _ in range(max_iter):
        x_new = x + 0.5 * dt * (f_n + f(t + dt, x_next))
        if np.max(np.abs(x_new - x_next)) < tol:
            return x_new
        x_next = x_new
    return x_next


INTEGRATORS = {
    "rk4": rk4_step,
    "newmark": newmark_average_velocity,
}


def get_integrator(name: str):
    try:
        return INTEGRATORS[name]
    except KeyError:
        raise ValueError(
            f"unknown integrator {name!r}; choose from {sorted(INTEGRATORS)}"
        ) from None
