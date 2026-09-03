"""BC-модель (CNN + MLP) и RL-политика (MLP) для PandaPickCube."""

import torch
import torch.nn as nn
import timm


class BCPolicy(nn.Module):
    """Мультимодальная BC-политика: image/proprio → вектор действия."""

    def __init__(
        self,
        action_dim: int = 8,
        encoder_name: str = "resnet18",
        proprio_dim: int = 16,
        use_image: bool = True,
        use_proprio: bool = False,
        proprio_mean=None,
        proprio_std=None,
        proprio_encoder_type: str = "mlp",
    ):
        super().__init__()
        if not use_image and not use_proprio:
            raise ValueError("Нужно включить хотя бы одну модальность.")
        if proprio_encoder_type not in {"mlp", "identity"}:
            raise ValueError("proprio_encoder_type должен быть mlp или identity.")
        self.use_image = use_image
        self.use_proprio = use_proprio
        self.proprio_encoder_type = proprio_encoder_type

        feature_dim = 0
        if self.use_image:
            self.encoder = timm.create_model(
                encoder_name, pretrained=True, num_classes=0
            )
            feature_dim += self.encoder.num_features

        if self.use_proprio:
            if proprio_mean is None:
                proprio_mean = torch.zeros(proprio_dim, dtype=torch.float32)
            if proprio_std is None:
                proprio_std = torch.ones(proprio_dim, dtype=torch.float32)
            proprio_mean = torch.as_tensor(proprio_mean, dtype=torch.float32)
            proprio_std = torch.as_tensor(proprio_std, dtype=torch.float32)
            if proprio_mean.shape != (proprio_dim,) or proprio_std.shape != (proprio_dim,):
                raise ValueError(
                    f"proprio_mean и proprio_std должны иметь форму ({proprio_dim},)."
                )
            self.register_buffer("proprio_mean", proprio_mean.clone())
            self.register_buffer("proprio_std", proprio_std.clone())

            if proprio_encoder_type == "mlp":
                self.proprio_encoder = nn.Sequential(
                    nn.Linear(proprio_dim, 64),
                    nn.ReLU(),
                    nn.Linear(64, 64),
                    nn.ReLU(),
                )
                feature_dim += 64
            else:
                self.proprio_encoder = nn.Identity()
                feature_dim += proprio_dim

        self.decoder = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Tanh(),
        )

    def forward(self, obs=None, proprio=None) -> torch.Tensor:
        """Выполнить forward по включённым модальностям."""
        features = []
        if self.use_image:
            if obs is None:
                raise ValueError("Для image-ветки требуется obs.")
            if obs.ndim != 4 or obs.shape[1] != 3:
                raise ValueError("obs должен иметь форму (B, 3, H, W).")
            features.append(self.encoder(obs))

        if self.use_proprio:
            if proprio is None:
                raise ValueError("Для proprio-ветки требуется proprio.")
            if proprio.ndim != 2 or proprio.shape[1] != self.proprio_mean.numel():
                raise ValueError(
                    "proprio должен иметь форму "
                    f"(B, {self.proprio_mean.numel()})."
                )
            normalized = (proprio - self.proprio_mean) / (self.proprio_std + 1e-6)
            features.append(self.proprio_encoder(normalized))

        if len(features) > 1:
            if features[0].shape[0] != features[1].shape[0]:
                raise ValueError("Размер batch у image и proprio должен совпадать.")
            fused = torch.cat(features, dim=-1)
        else:
            fused = features[0]
        return self.decoder(fused)


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
