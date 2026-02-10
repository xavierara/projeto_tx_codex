from plan_policy import discretize_delta, plan_policy


def test_policy_returns_valid_actions():
    action_set = [-0.1, 0.0, 0.1]
    delta_grid = discretize_delta(-0.2, 0.2, 0.1)

    def hazard_fn(_, __, ___):
        return 0.2

    def p0_fn(_):
        return 100.0

    def constraint_fn(_, delta):
        return delta

    def terminal_reward_fn(p_t, _):
        return p_t

    result = plan_policy(
        hazard_fn=hazard_fn,
        terminal_reward_fn=terminal_reward_fn,
        p0_fn=p0_fn,
        constraint_fn=constraint_fn,
        action_set=action_set,
        delta_grid=delta_grid,
        t_max=2,
        holding_cost=1.0,
        x={"p0": 100.0},
    )
    for t in result.action_table:
        for delta in result.action_table[t]:
            assert result.action_table[t][delta] in action_set
