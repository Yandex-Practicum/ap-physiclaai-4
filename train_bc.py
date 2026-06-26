"""Обучение визуомоторной BC-модели на собранном датасете.

Вход модели — изображение с камеры гриппера. Это baseline-конфигурация
практики; в уроках вы расширите observation space (добавите проприоцепцию).

Запуск:
    python3 train_bc.py --train_dir dataset/train --eval_dir dataset/eval --exp_name bc_baseline --epochs 100
"""

import argparse
import glob
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from model import BCPolicy


def parse_args():
    parser = argparse.ArgumentParser(description="Обучение визуомоторной BC-модели")
    parser.add_argument("--train_dir", type=str, required=True)
    parser.add_argument("--eval_dir", type=str, required=True)
    parser.add_argument("--exp_name", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


class EpisodeDataset(Dataset):
    """Датасет из .npz: каждый шаг = (obs, action)."""

    def __init__(self, data_dir: str):
        self.files = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
        if not self.files:
            raise ValueError(f"Нет .npz файлов в {data_dir}")

        obs_l, act_l = [], []
        for f in self.files:
            ep = np.load(f)
            obs = ep["obs"]
            acts = ep["actions"]
            T = min(obs.shape[0], acts.shape[0])
            obs_l.append(obs[:T])
            act_l.append(acts[:T])

        self.observations = np.concatenate(obs_l, axis=0)
        self.actions = np.concatenate(act_l, axis=0)
        print(f"Loaded {len(self.files)} episodes, {len(self)} steps from {data_dir}")

    def __len__(self):
        return len(self.observations)

    def __getitem__(self, idx):
        obs = torch.from_numpy(self.observations[idx].copy()).float() / 255.0
        action = torch.from_numpy(self.actions[idx].copy())
        return obs, action


def evaluate(model, loader, criterion, device):
    model.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for obs, actions in loader:
            actions = actions.to(device)
            pred = model(obs.to(device))
            total += criterion(pred, actions).item()
            n += 1
    return total / max(n, 1)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    train_dataset = EpisodeDataset(args.train_dir)
    eval_dataset = EpisodeDataset(args.eval_dir)

    pin = (device == "cuda")
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=pin, drop_last=True)
    eval_loader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=pin)

    model = BCPolicy(action_dim=8).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    log_dir = os.path.join("logs", args.exp_name)
    ckpt_dir = os.path.join(log_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)

    best_eval_loss, best_epoch = float("inf"), 0
    print(f"\nОбучение: {args.exp_name}")
    print(f"  Train: {len(train_dataset)} steps | Eval: {len(eval_dataset)} steps")
    print(f"  Epochs: {args.epochs} | Batch: {args.batch_size} | LR: {args.lr}\n")

    def save(path, epoch, tr, ev):
        torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "eval_loss": ev, "train_loss": tr}, path)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        tot, n = 0.0, 0
        for obs, actions in train_loader:
            actions = actions.to(device)
            pred = model(obs.to(device))
            loss = criterion(pred, actions)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            tot += loss.item(); n += 1

        train_loss = tot / max(n, 1)
        eval_loss = evaluate(model, eval_loader, criterion, device)
        writer.add_scalar("train/loss", train_loss, epoch)
        writer.add_scalar("eval/loss", eval_loss, epoch)

        if eval_loss < best_eval_loss:
            best_eval_loss, best_epoch = eval_loss, epoch
            save(os.path.join(ckpt_dir, "best.pt"), epoch, train_loss, eval_loss)

        print(f"Epoch {epoch:3d}/{args.epochs} | train_loss: {train_loss:.4f} | "
              f"eval_loss: {eval_loss:.4f} | {time.time() - t0:.0f}s")

    save(os.path.join(ckpt_dir, "last.pt"), args.epochs, train_loss, eval_loss)
    writer.close()
    print(f"\nBest checkpoint: {ckpt_dir}/best.pt (epoch {best_epoch}, eval_loss={best_eval_loss:.4f})")


if __name__ == "__main__":
    main()
