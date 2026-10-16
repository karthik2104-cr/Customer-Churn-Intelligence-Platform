"""Convert numpy and pandas values into JSON-safe objects."""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return to_jsonable(value.tolist())
    if isinstance(value, np.floating):
        return to_jsonable(float(value))
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if hasattr(value, "item"):
        try:
            return to_jsonable(value.item())
        except Exception:
            return str(value)
    return str(value)
