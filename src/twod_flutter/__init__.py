"""2D airfoil stall flutter via a modified Leishman-Beddoes model.

Implements Shao et al., *Chinese Journal of Aeronautics* 24(5), 2011, 550-557.
See ``docs/theory.md`` for the full formulation, the constants, and -- crucially
-- the register of what the paper leaves undefined (§9).
"""

from .aero import LBModel
from .config import Airfoil, Case, Flow, ForcedPitch, Structure

__version__ = "0.1.0"

__all__ = [
    "Airfoil",
    "Case",
    "Flow",
    "ForcedPitch",
    "LBModel",
    "Structure",
]
