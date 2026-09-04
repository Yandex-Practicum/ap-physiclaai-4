"""Контракт синхронного сбора изображения и проприоцепции."""

import numpy as np
import torch
import torch.nn as nn

from collect_data import collect_episode


class DummyEnv:
    episode_length = 3

    def __init__(self):
        self.t = 0

    def reset(self, seed=None):
        self.t = 0
        return np.full((84, 84, 3), self.t, dtype=np.uint8)

    def step(self, action):
        self.t += 1
        obs = np.full((84, 84, 3), self.t, dtype=np.uint8)
        done = self.t >= self.episode_length
        return obs, done, done

    def get_privileged_state(self):
        return np.full(29, self.t, dtype=np.float32)

    def get_proprio(self):
        return np.full(16, self.t, dtype=np.float32)


class ZeroPolicy(nn.Module):
    def forward(self, state):
        return torch.zeros((state.shape[0], 8), dtype=torch.float32)


def test_collect_episode_returns_synchronized_proprio():
    result = collect_episode(DummyEnv(), ZeroPolicy(), device="cpu", rng_seed=7)
    obs, state, proprio, actions, dones, success = result

    assert obs.shape == (3, 84, 84, 3)
    assert state.shape == (3, 8)
    assert proprio.shape == (3, 16)
    assert actions.shape == (3, 8)
    assert dones.shape == (3,)
    assert success == 1

    np.testing.assert_array_equal(obs[:, 0, 0, 0], [0, 1, 2])
    np.testing.assert_array_equal(proprio[:, 0], [0.0, 1.0, 2.0])
