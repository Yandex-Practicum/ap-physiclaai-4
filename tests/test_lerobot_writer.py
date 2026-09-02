"""Unit-тест для LeRobotWriter (Урок 4).

Проверяет, что студент реализовал запись в LeRobotDataset v3.0: создаётся
структура папок meta/ data/ videos/, видео сохраняется в .mp4, датасет читается
через API LeRobot и содержит правильное число эпизодов/кадров и корректные типы.

Требует пакет lerobot (есть в контейнере практики). Запуск:
    pytest tests/test_lerobot_writer.py
"""
import glob
import os
import tempfile

import numpy as np
import torch

from collect_data import LeRobotWriter


def _fake_episode(T=6):
    obs = (np.random.rand(T, 84, 84, 3) * 255).astype(np.uint8)
    state = np.random.rand(T, 8).astype(np.float32)
    proprio = np.random.rand(T, 16).astype(np.float32)
    act = np.random.rand(T, 8).astype(np.float32)
    return obs, state, proprio, act


def test_writer_produces_loadable_dataset():
    base = tempfile.mkdtemp()
    save_dir = os.path.join(base, "ds")  # не должна существовать заранее

    writer = LeRobotWriter(save_dir)
    obs, state, proprio, act = _fake_episode(T=6)
    writer.add_episode(obs, state, proprio, act)
    obs2, state2, proprio2, act2 = _fake_episode(T=4)
    writer.add_episode(obs2, state2, proprio2, act2)
    writer.finalize()

    # 1. Структура папок LeRobotDataset v3.0
    for sub in ("meta", "data", "videos"):
        assert os.path.isdir(os.path.join(save_dir, sub)), f"нет папки {sub}/"

    # 2. Видео сохранено в формате .mp4
    mp4s = glob.glob(os.path.join(save_dir, "videos", "**", "*.mp4"), recursive=True)
    assert mp4s, "видео .mp4 не найдено — проверьте use_videos=True и dtype 'video'"

    # 3. Датасет читается через API и содержит корректные данные
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    ds = LeRobotDataset(
        repo_id="local/practice4",
        root=save_dir,
        video_backend="torchcodec",
    )
    assert ds.meta.info["codebase_version"] == "v3.0"
    assert ds.meta.total_episodes == 2
    assert ds.meta.total_frames == 10

    sample = ds[0]
    assert sample["action"].shape == (8,)
    assert sample["action"].dtype == torch.float32
    assert sample["observation.state"].shape == (8,)
    assert sample["observation.state"].dtype == torch.float32
    assert sample["observation.proprio"].shape == (16,)
    assert sample["observation.proprio"].dtype == torch.float32
    # изображение декодируется из видео как (C, H, W)
    assert tuple(sample["observation.images.front"].shape) == (3, 84, 84)
    assert sample["observation.images.front"].dtype == torch.float32
    assert 0.0 <= float(sample["observation.images.front"].min())
    assert float(sample["observation.images.front"].max()) <= 1.0


def test_training_dataset_reads_lerobot():
    base = tempfile.mkdtemp()
    save_dir = os.path.join(base, "ds")

    writer = LeRobotWriter(save_dir)
    obs, state, proprio, act = _fake_episode(T=4)
    writer.add_episode(obs, state, proprio, act)
    writer.finalize()

    from train_bc import EpisodeDataset

    dataset = EpisodeDataset(save_dir, obs_mode="both")
    sample = dataset[0]
    assert tuple(sample["image"].shape) == (3, 84, 84)
    assert tuple(sample["proprio"].shape) == (16,)
    assert tuple(sample["action"].shape) == (8,)
    assert sample["image"].dtype == torch.float32
    assert sample["proprio"].dtype == torch.float32
    assert sample["action"].dtype == torch.float32
