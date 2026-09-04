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
    assert policy.decoder[0].in_features == 512 + 64


def test_both_identity_ablation_concatenates_normalized_proprio_directly():
    mean = torch.arange(16, dtype=torch.float32)
    std = torch.full((16,), 2.0)
    policy = BCPolicy(
        use_image=True,
        use_proprio=True,
        proprio_mean=mean,
        proprio_std=std,
        proprio_encoder_type="identity",
    )
    captured = {}

    def capture_decoder_input(_module, args):
        captured["features"] = args[0].detach().clone()

    policy.decoder.register_forward_pre_hook(capture_decoder_input)
    proprio = mean.unsqueeze(0).repeat(2, 1) + 2.0
    action = policy(
        obs=torch.zeros(2, 3, 84, 84),
        proprio=proprio,
    )

    assert action.shape == (2, 8)
    assert policy.decoder[0].in_features == 512 + 16
    torch.testing.assert_close(captured["features"][:, :512], torch.ones(2, 512))
    expected_proprio = (proprio - mean) / (std + 1e-6)
    torch.testing.assert_close(captured["features"][:, 512:], expected_proprio)


def test_rejects_configuration_without_modalities():
    with pytest.raises(ValueError, match="модальност"):
        BCPolicy(use_image=False, use_proprio=False)


def test_rejects_unknown_proprio_encoder_type():
    with pytest.raises(ValueError, match="proprio_encoder_type"):
        BCPolicy(proprio_encoder_type="unknown")
