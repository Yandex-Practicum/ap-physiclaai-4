"""Контракт проприоцептивного наблюдения PandaPickCubeEnv."""

from types import SimpleNamespace

import numpy as np
import pytest

from env import PandaPickCubeEnv


def test_get_proprio_returns_joint_positions_and_velocities():
    env = PandaPickCubeEnv.__new__(PandaPickCubeEnv)
    env.data = SimpleNamespace(
        qpos=np.arange(20, dtype=np.float64),
        qvel=np.arange(30, 50, dtype=np.float64),
    )

    proprio = env.get_proprio()

    expected = np.concatenate([
        np.arange(9, dtype=np.float32),
        np.arange(30, 37, dtype=np.float32),
    ])
    np.testing.assert_array_equal(proprio, expected)
    assert proprio.shape == (16,)
    assert proprio.dtype == np.float32
    assert proprio.flags.c_contiguous


def test_get_proprio_returns_a_copy():
    env = PandaPickCubeEnv.__new__(PandaPickCubeEnv)
    env.data = SimpleNamespace(
        qpos=np.arange(20, dtype=np.float64),
        qvel=np.arange(30, 50, dtype=np.float64),
    )

    proprio = env.get_proprio()
    proprio[0] = -999

    assert env.data.qpos[0] == pytest.approx(0.0)
