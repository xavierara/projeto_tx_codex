from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Callable, Dict, List

import numpy as np

from env import CarPricingEnv, EnvState


@dataclass
class EpisodeResult:
    total_reward: float
    weeks: int
    terminal_price: float | None
    event: bool


def run_episode(
    env: CarPricingEnv,
    policy_fn: Callable[[EnvState], float],
    initial_state: EnvState,
) -> EpisodeResult:
    state = env.reset(initial_state.x, initial_state.delta)
    total_reward = 0.0
    event_price = None
    while True:
        action = policy_fn(state)
        step = env.step(action)
        total_reward += step.reward
        state = step.state
        if step.event:
            event_price = initial_state.x["p0"] * np.exp(state.delta)
        if step.done:
            break
    return EpisodeResult(
        total_reward=total_reward,
        weeks=state.t,
        terminal_price=event_price,
        event=step.event,
    )


def simulate(
    env_builder: Callable[[Dict], CarPricingEnv],
    policy_fn: Callable[[EnvState], float],
    states: List[EnvState],
) -> List[EpisodeResult]:
    results = []
    for state in states:
        env = env_builder(state.x)
        results.append(run_episode(env, policy_fn, state))
    return results


def summarize(results: List[EpisodeResult]) -> Dict[str, float]:
    total_rewards = [r.total_reward for r in results]
    weeks = [r.weeks for r in results]
    terminal_prices = [r.terminal_price for r in results if r.terminal_price is not None]
    events = [r.event for r in results]
    return {
        "expected_total_reward": float(np.mean(total_rewards)),
        "expected_time_to_event": float(np.mean(weeks)),
        "event_rate": float(np.mean(events)),
        "expected_terminal_price": float(np.mean(terminal_prices)) if terminal_prices else 0.0,
    }


def save_summary(summary: Dict[str, float], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
