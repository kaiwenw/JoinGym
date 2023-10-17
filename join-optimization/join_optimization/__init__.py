from gymnasium.envs.registration import register

register(
    id='join_optimization_left-v0',
    entry_point='join_optimization.envs:JoinOptEnvLeft',
)

register(
    id='join_optimization_bushy-v0',
    entry_point='join_optimization.envs:JoinOptEnvBushy',
)