from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Callable, Dict, List

import numpy as np


@dataclass
class PolicyResult:
    value_table: Dict[int, Dict[float, float]]
    action_table: Dict[int, Dict[float, float]]
    delta_grid: List[float]


def discretize_delta(delta_min: float, delta_max: float, step: float) -> List[float]:
    grid = np.arange(delta_min, delta_max + 1e-9, step)
    return [float(np.round(val, 4)) for val in grid]


def plan_policy(
    hazard_fn: Callable[[Dict, int, float], float],
    terminal_reward_fn: Callable[[float, Dict], float],
    p0_fn: Callable[[Dict], float],
    constraint_fn: Callable[[Dict, float], float],
    action_set: List[float],
    delta_grid: List[float],
    t_max: int,
    holding_cost: float,
    x: Dict,
) -> PolicyResult:
    value_table: Dict[int, Dict[float, float]] = {}
    action_table: Dict[int, Dict[float, float]] = {}

    for t in range(t_max, 0, -1):
        value_table[t] = {}
        action_table[t] = {}
        for delta in delta_grid:
            best_value = -np.inf
            best_action = action_set[0]
            for action in action_set:
                delta_prime = constraint_fn(x, delta + action)
                hazard = float(np.clip(hazard_fn(x, t, delta_prime), 0.0, 1.0))
                p0 = p0_fn(x)
                p_t = p0 * np.exp(delta_prime)
                terminal_reward = terminal_reward_fn(p_t, x)
                next_value = 0.0
                if t < t_max:
                    next_delta = float(np.round(delta_prime, 4))
                    next_value = value_table[t + 1].get(next_delta, 0.0)
                q_val = hazard * terminal_reward + (1 - hazard) * (
                    -holding_cost + next_value
                )
                if q_val > best_value:
                    best_value = q_val
                    best_action = action
            value_table[t][float(np.round(delta, 4))] = float(best_value)
            action_table[t][float(np.round(delta, 4))] = float(best_action)

    return PolicyResult(
        value_table=value_table,
        action_table=action_table,
        delta_grid=delta_grid,
    )


def save_policy(result: PolicyResult, path: str) -> None:
    payload = {
        "value_table": result.value_table,
        "action_table": result.action_table,
        "delta_grid": result.delta_grid,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
