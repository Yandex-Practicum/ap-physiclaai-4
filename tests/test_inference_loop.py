"""Контракты closed-loop инференса для image/both/proprio режимов."""

import numpy as np
import pytest
import torch
import torch.nn as nn

import inference as inference_module
from inference import load_bc_policy, run_episode_bc


class DummyEnv:
    episode_length = 10

    def __init__(self, succeed_at=5):
        self.succeed_at = succeed_at
        self.t = 0

    def reset(self, seed=None):
        self.t = 0
        return np.full((84, 84, 3), self.t, dtype=np.uint8)

    def step(self, action):
        assert np.asarray(action).shape == (8,)
        self.t += 1
        obs = np.full((84, 84, 3), self.t, dtype=np.uint8)
        success = self.t >= self.succeed_at
        done = success or self.t >= self.episode_length
        return obs, success, done

    def get_proprio(self):
        return np.full(16, self.t, dtype=np.float32)


class RecordingPolicy(nn.Module):
    def __init__(self, use_image, use_proprio):
        super().__init__()
        self.use_image = use_image
        self.use_proprio = use_proprio
        self.calls = []
        self.grad_was_enabled = None

    def forward(self, obs=None, proprio=None):
        self.grad_was_enabled = torch.is_grad_enabled()
        image_value = None if obs is None else float(obs[0, 0, 0, 0])
        proprio_value = None if proprio is None else float(proprio[0, 0])
        self.calls.append((image_value, proprio_value))
        return torch.zeros((1, 8), dtype=torch.float32)


def test_image_mode_prepares_bchw_float_image():
    env = DummyEnv(succeed_at=2)
    policy = RecordingPolicy(use_image=True, use_proprio=False)

    success, steps = run_episode_bc(env, policy, device="cpu", seed=0)

    assert bool(success) is True
    assert steps == 2
    assert policy.calls[0] == (0.0, None)
    assert policy.calls[1][0] == pytest.approx(1.0 / 255.0)
    assert policy.calls[1][1] is None


def test_both_mode_reads_synchronized_modalities():
    env = DummyEnv(succeed_at=3)
    policy = RecordingPolicy(use_image=True, use_proprio=True)

    run_episode_bc(env, policy, device="cpu", seed=0)

    for image_value, proprio_value in policy.calls:
        assert round(image_value * 255) == proprio_value


def test_proprio_mode_does_not_pass_image():
    env = DummyEnv(succeed_at=2)
    policy = RecordingPolicy(use_image=False, use_proprio=True)

    run_episode_bc(env, policy, device="cpu", seed=0)

    assert policy.calls == [(None, 0.0), (None, 1.0)]


def test_timeout_returns_false_and_inference_uses_no_grad():
    env = DummyEnv(succeed_at=999)
    policy = RecordingPolicy(use_image=False, use_proprio=True)

    success, steps = run_episode_bc(env, policy, device="cpu", seed=0)

    assert bool(success) is False
    assert steps == env.episode_length
    assert policy.grad_was_enabled is False


class FakeLoadedPolicy:
    def __init__(
        self,
        action_dim,
        use_image,
        use_proprio,
        proprio_encoder_type="mlp",
        **kwargs,
    ):
        self.action_dim = action_dim
        self.use_image = use_image
        self.use_proprio = use_proprio
        self.proprio_encoder_type = proprio_encoder_type

    def load_state_dict(self, state_dict):
        self.state_dict_value = state_dict

    def to(self, device):
        return self

    def eval(self):
        return self


def test_load_bc_policy_restores_obs_mode(monkeypatch):
    monkeypatch.setattr(inference_module, "BCPolicy", FakeLoadedPolicy)
    monkeypatch.setattr(
        inference_module.torch,
        "load",
        lambda *args, **kwargs: {
            "obs_mode": "both",
            "proprio_encoder_type": "identity",
            "model_state_dict": {"weight": torch.tensor(1.0)},
        },
    )

    policy = load_bc_policy("/unused", "cpu")

    assert policy.use_image is True
    assert policy.use_proprio is True
    assert policy.proprio_encoder_type == "identity"


def test_load_bc_policy_defaults_old_checkpoint_to_image(monkeypatch):
    monkeypatch.setattr(inference_module, "BCPolicy", FakeLoadedPolicy)
    monkeypatch.setattr(
        inference_module.torch,
        "load",
        lambda *args, **kwargs: {"model_state_dict": {}},
    )

    policy = load_bc_policy("/unused", "cpu")

    assert policy.use_image is True
    assert policy.use_proprio is False
    assert policy.proprio_encoder_type == "mlp"
