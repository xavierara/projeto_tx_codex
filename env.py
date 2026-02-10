from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Tuple

import numpy as np


@dataclass
class EnvState:
    x: Dict
    t: int
    delta: float


@dataclass
class StepResult:
    state: EnvState
    reward: float
    done: bool
    event: bool


class CarPricingEnv:
    def __init__(
        self,
        hazard_fn: Callable[[Dict, int, float], float],
        p0_fn: Callable[[Dict], float],
        constraint_fn: Callable[[Dict, float], float],
        holding_cost: float,
        t_max: int,
        rng: np.random.Generator,
        terminal_reward_fn: Callable[[float, Dict], float],
    ) -> None:
        self.hazard_fn = hazard_fn
        self.p0_fn = p0_fn
        self.constraint_fn = constraint_fn
        self.holding_cost = holding_cost
        self.t_max = t_max
        self.rng = rng
        self.terminal_reward_fn = terminal_reward_fn
        self.state: EnvState | None = None

    def reset(self, x: Dict, delta: float) -> EnvState:
        self.state = EnvState(x=x, t=1, delta=delta)
        return self.state

    def step(self, action: float) -> StepResult:
        if self.state is None:
            raise RuntimeError("Environment must be reset before step.")
        x = self.state.x
        t = self.state.t
        delta_prime = self.constraint_fn(x, self.state.delta + action)
        p0 = self.p0_fn(x)
        p_t = p0 * np.exp(delta_prime)
        hazard = self.hazard_fn(x, t, delta_prime)
        hazard = float(np.clip(hazard, 0.0, 1.0))

        event = self.rng.random() < hazard
        if event:
            reward = self.terminal_reward_fn(p_t, x)
            done = True
            next_state = EnvState(x=x, t=t, delta=delta_prime)
        else:
            reward = -self.holding_cost
            next_t = t + 1
            done = next_t > self.t_max
            next_state = EnvState(x=x, t=next_t, delta=delta_prime)
        self.state = next_state
        return StepResult(state=next_state, reward=reward, done=done, event=event)


def default_terminal_reward(p_t: float, x: Dict) -> float:
    return float(p_t)


def margin_proxy_reward(p_t: float, x: Dict, alpha: float = 0.0) -> float:
    p0 = x.get("p0")
    if p0 is None:
        return float(p_t)
    return float(p_t - alpha * p0)


def conservative_hazard(
    probs: np.ndarray, k: float
) -> float:
    mean = probs.mean()
    std = probs.std()
    return float(np.clip(mean - k * std, 0.0, 1.0))


def build_constraint_fn(
    delta_min: float, delta_max: float
) -> Callable[[Dict, float], float]:
    def _constraint(_: Dict, delta: float) -> float:
        return float(np.clip(delta, delta_min, delta_max))

    return _constraint


def compute_price(p0: float, delta: float) -> float:
    return float(p0 * np.exp(delta))


def update_delta(p0: float, price: float) -> float:
    return float(np.log(price / p0))
