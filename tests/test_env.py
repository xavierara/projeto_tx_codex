import numpy as np

from env import CarPricingEnv, default_terminal_reward, build_constraint_fn


def test_env_transition_terminal():
    rng = np.random.default_rng(0)

    def hazard_fn(_, __, ___):
        return 1.0

    def p0_fn(_):
        return 100.0

    env = CarPricingEnv(
        hazard_fn=hazard_fn,
        p0_fn=p0_fn,
        constraint_fn=build_constraint_fn(-0.5, 0.5),
        holding_cost=1.0,
        t_max=4,
        rng=rng,
        terminal_reward_fn=default_terminal_reward,
    )
    state = env.reset({"p0": 100.0}, delta=0.0)
    step = env.step(0.0)
    assert step.done is True
    assert step.event is True
    assert step.reward == 100.0
    assert step.state.t == state.t


def test_env_transition_hold():
    rng = np.random.default_rng(0)

    def hazard_fn(_, __, ___):
        return 0.0

    def p0_fn(_):
        return 100.0

    env = CarPricingEnv(
        hazard_fn=hazard_fn,
        p0_fn=p0_fn,
        constraint_fn=build_constraint_fn(-0.5, 0.5),
        holding_cost=2.0,
        t_max=2,
        rng=rng,
        terminal_reward_fn=default_terminal_reward,
    )
    env.reset({"p0": 100.0}, delta=0.0)
    step = env.step(0.0)
    assert step.done is False
    assert step.event is False
    assert step.reward == -2.0
    assert step.state.t == 2
