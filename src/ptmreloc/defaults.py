"""Default reassignment parameters.

Taken from ``sampling/Optimizing_Reassignment.py`` (Monte Carlo sampling on the
MCF10A EGF time-course phosphoproteome, median of the best 5% trials). They were
derived with the ``legacy`` feature settings (the :class:`FeatureConfig` defaults).
"""
from __future__ import annotations

from typing import Dict

DEFAULT_THRESHOLD = 30.0

#: median weights of the top-5% trials by the composite (GS x FUN) score — the package default
DEFAULT_WEIGHTS: Dict[str, float] = {
    "w_H": 0.40935512896496834,
    "w_L": 0.0028834480249440245,
    "w_D": 0.004439231598796614,
}

#: named weight sets usable wherever weights are accepted, e.g. ``weights="htp_only"``
WEIGHT_PRESETS: Dict[str, Dict[str, float]] = {
    "medians": DEFAULT_WEIGHTS,
    "fun_medians": {"w_H": 0.3373114823599035, "w_L": 0.004482122763962499, "w_D": 0.003735925450381492},
    "gs_medians": {"w_H": 0.42056700185773666, "w_L": 0.0031175186176344164, "w_D": 0.004656634333672645},
    "htp_only": {"w_H": 1.0, "w_L": 0.0, "w_D": 0.0},
}
