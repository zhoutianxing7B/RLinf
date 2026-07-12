"""Deterministic sampling helpers for bounded rollout training."""


def evenly_spaced_indices(total: int, target: int) -> list[int]:
    """Return ``target`` unique indices spanning ``[0, total)``."""
    if total < 0 or target < 0:
        raise ValueError("total and target must be non-negative")
    if target > total:
        raise ValueError(f"target ({target}) cannot exceed total ({total})")
    if target == 0:
        return []
    if target == 1:
        return [0]
    return [index * (total - 1) // (target - 1) for index in range(target)]
