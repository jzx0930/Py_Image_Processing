import numpy as np
from processing.registry import get_method


def apply_step(image: np.ndarray, step: dict) -> np.ndarray:
    method = get_method(step["method"])
    return method.fn(image, step["params"])


def run_pipeline(original: np.ndarray, steps: list[dict]) -> np.ndarray:
    result = original.copy()
    for step in steps:
        try:
            result = apply_step(result, step)
        except Exception:
            pass
    return result


def make_step_label(step: dict) -> str:
    method = get_method(step["method"])
    if not step["params"]:
        return f"{method.category} / {method.name}"
    param_str = "  ".join(f"{k}={v}" for k, v in step["params"].items())
    return f"{method.category} / {method.name}  ({param_str})"
