"""Контракты трёх конфигураций мультимодальной BCPolicy."""

import pytest
import torch
import torch.nn as nn

import model as model_module
from model import BCPolicy


class FakeImageEncoder(nn.Module):
    num_features = 512

    def forward(self, image):
        return torch.ones((image.shape[0], self.num_features), dtype=image.dtype)


@pytest.fixture(autouse=True)
def fake_image_encoder(monkeypatch):
    monkeypatch.setattr(
        model_module.timm,
        "create_model",
        lambda *args, **kwargs: FakeImageEncoder(),
    )


def test_image_mode_accepts_only_image():
    policy = BCPolicy(use_image=True, use_proprio=False)
    action = policy(obs=torch.zeros(2, 3, 84, 84))
    assert action.shape == (2, 8)


def test_proprio_mode_accepts_only_proprio_and_registers_stats():
    mean = torch.arange(16, dtype=torch.float32)
    std = torch.arange(1, 17, dtype=torch.float32)
    policy = BCPolicy(
        use_image=False,
        use_proprio=True,
        proprio_mean=mean,
        proprio_std=std,
    )

    action = policy(proprio=torch.zeros(2, 16))

    assert action.shape == (2, 8)
    buffers = dict(policy.named_buffers())
    torch.testing.assert_close(buffers["proprio_mean"], mean)
    torch.testing.assert_close(buffers["proprio_std"], std)


def test_both_mode_accepts_image_and_proprio():
    policy = BCPolicy(
        use_image=True,
        use_proprio=True,
        proprio_mean=torch.zeros(16),
        proprio_std=torch.ones(16),
    )
    action = policy(
        obs=torch.zeros(2, 3, 84, 84),
        proprio=torch.zeros(2, 16),
    )
    assert action.shape == (2, 8)


def test_rejects_configuration_without_modalities():
    with pytest.raises(ValueError, match="модальност"):
        BCPolicy(use_image=False, use_proprio=False)
