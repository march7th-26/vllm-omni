# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Zero-based sparsity schedules shared by full and rectangular attention."""

import math


def normalize_schedule(values: list[float] | tuple[float, ...] | None) -> tuple[float, ...]:
    if values is None:
        return ()
    if not isinstance(values, (list, tuple)):
        raise ValueError("step_sparsities must be a list or tuple")
    schedule = tuple(float(value) for value in values)
    if any(not math.isfinite(value) or not 0 < value < 1 for value in schedule):
        raise ValueError("step_sparsities must contain finite values strictly between 0 and 1")
    return schedule


def resolve_sparsity(scalar: float, schedule: tuple[float, ...], step: int | None, total_steps: int | None) -> float:
    if not schedule:
        return scalar
    if total_steps != len(schedule):
        raise ValueError(f"step_sparsities length {len(schedule)} != denoise steps {total_steps}")
    if type(step) is not int or not 0 <= step < len(schedule):
        raise ValueError(f"step_sparsities has no entry for denoise step {step!r}")
    return schedule[step]
