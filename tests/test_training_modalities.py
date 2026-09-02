"""Контракты LeRobot-выборки и статистик проприоцепции."""

import sys
from types import ModuleType, SimpleNamespace

import torch
from torch.utils.data import Dataset

import train_bc
from train_bc import EpisodeDataset


class FakeLeRobotDataset:
    def __init__(self, **kwargs):
        self.meta = SimpleNamespace(total_episodes=1)
        self.samples = [
            {
                "observation.images.front": torch.ones(3, 84, 84),
                "observation.proprio": torch.arange(16, dtype=torch.float32),
                "action": torch.zeros(8),
            }
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]


def install_fake_lerobot(monkeypatch):
    lerobot = ModuleType("lerobot")
    datasets = ModuleType("lerobot.datasets")
    dataset_module = ModuleType("lerobot.datasets.lerobot_dataset")
    dataset_module.LeRobotDataset = FakeLeRobotDataset
    monkeypatch.setitem(sys.modules, "lerobot", lerobot)
    monkeypatch.setitem(sys.modules, "lerobot.datasets", datasets)
    monkeypatch.setitem(
        sys.modules,
        "lerobot.datasets.lerobot_dataset",
        dataset_module,
    )


def test_episode_dataset_returns_named_lerobot_modalities(monkeypatch):
    install_fake_lerobot(monkeypatch)
    dataset = EpisodeDataset("/unused", obs_mode="both")

    sample = dataset[0]

    assert set(sample) == {"image", "proprio", "action"}
    assert sample["image"].shape == (3, 84, 84)
    assert sample["proprio"].shape == (16,)
    assert sample["action"].shape == (8,)


class ProprioDataset(Dataset):
    def __init__(self):
        self.values = [
            torch.zeros(16),
            torch.full((16,), 2.0),
        ]

    def __len__(self):
        return len(self.values)

    def __getitem__(self, index):
        return {
            "proprio": self.values[index],
            "action": torch.zeros(8),
        }


def test_compute_proprio_stats_uses_dataset_values():
    mean, std = train_bc.compute_proprio_stats(ProprioDataset())
    torch.testing.assert_close(mean, torch.ones(16))
    torch.testing.assert_close(std, torch.ones(16))
