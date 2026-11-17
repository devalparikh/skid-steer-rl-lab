import numpy as np
from stable_baselines3.common.env_checker import check_env

from skid_steer_rl import LoaderEnv


def test_api_and_seeded_reset():
    env = LoaderEnv()
    check_env(env)
    a, _ = env.reset(seed=12)
    goal = env.goal.copy()
    for _ in range(5):
        env.step(np.array([0.5, -0.2]))
    b, _ = env.reset(seed=12)
    np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(goal, env.goal)
    env.close()


def test_wheels_move_machine_and_remain_finite():
    env = LoaderEnv()
    env.reset(seed=10)
    for _ in range(20):
        observation, reward, _, _, _ = env.step(np.array([0.5, 0.5]))
        assert np.isfinite(observation).all()
        assert np.isfinite(reward)
    assert env.data.qpos[0] > 0.3
    assert 0.5 < env.data.qpos[2] < 0.8
    env.close()


def test_timeout_is_truncation_and_parking_requires_hold():
    env = LoaderEnv()
    env.reset(seed=3)
    env.ticks = 239
    _, _, terminated, truncated, _ = env.step(np.zeros(2))
    assert truncated and not terminated
    env.reset(seed=3)
    env.goal[:] = env.data.qpos[:2]
    env.distance = 0
    for tick in range(10):
        _, _, terminated, truncated, info = env.step(np.zeros(2))
        assert terminated == (tick == 9)
    assert info["success"] and not truncated
    env.close()
