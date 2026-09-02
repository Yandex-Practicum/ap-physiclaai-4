"""BC-модель (CNN + MLP) и RL-политика (MLP) для PandaPickCube."""

import torch
import torch.nn as nn
import timm


class BCPolicy(nn.Module):
    """Мультимодальная BC-политика: image/proprio → вектор действия.

    TODO (Практика 4):
      - поддержите конфигурации image, both и proprio через флаги;
      - создавайте CNN только при use_image;
      - создавайте proprio-энкодер 16→64→64 только при use_proprio;
      - храните proprio mean/std через register_buffer;
      - стройте decoder от суммы размеров включённых веток.

    Для режима image сохраните имена модулей encoder/decoder и прежние размеры:
    это позволяет загружать legacy checkpoint без поля obs_mode.
    """

    def __init__(
        self,
        action_dim: int = 8,
        encoder_name: str = "resnet18",
        proprio_dim: int = 16,
        use_image: bool = True,
        use_proprio: bool = False,
        proprio_mean=None,
        proprio_std=None,
    ):
        super().__init__()
        if not use_image and not use_proprio:
            raise ValueError("Нужно включить хотя бы одну модальность.")
        self.use_image = use_image
        self.use_proprio = use_proprio
        raise NotImplementedError(
            "Реализуйте условные image/proprio ветки BCPolicy (Практика 4)."
        )

    def forward(self, obs=None, proprio=None) -> torch.Tensor:
        """Выполнить forward по включённым модальностям.

        Изображение из LeRobot уже имеет формат BCHW float32 [0,1].

        TODO (Практика 4):
          1) для image-ветки проверьте BCHW-контракт и получите CNN-фичи;
          2) для proprio-ветки проверьте форму (B,16), нормализуйте значения
             как (proprio - mean) / (std + 1e-6) и примените MLP;
          3) объедините доступные фичи через torch.cat(..., dim=-1);
          4) передайте результат в decoder.
        """
        raise NotImplementedError(
            "Реализуйте мультимодальный forward BCPolicy (Практика 4)."
        )


class RLPolicy(nn.Module):
    """MLP-политика для privileged state → action."""

    def __init__(self, state_dim: int = 29, action_dim: int = 8,
                 hidden_dims: tuple = (512, 256, 128)):
        super().__init__()
        layers = []
        in_dim = state_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(in_dim, h), nn.ELU()])
            in_dim = h
        layers.append(nn.Linear(in_dim, action_dim))
        layers.append(nn.Tanh())
        self.net = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)
