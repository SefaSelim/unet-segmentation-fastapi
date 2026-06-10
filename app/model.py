from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


class DoubleConv(nn.Module):
    """Two Conv-BatchNorm-ReLU blocks matching the trained checkpoint keys."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UNet(nn.Module):
    """U-Net architecture used by the saved segmentation checkpoint."""

    def __init__(self, in_channels: int = 3, out_channels: int = 1) -> None:
        super().__init__()
        self.down1 = DoubleConv(in_channels, 64)
        self.down2 = DoubleConv(64, 128)
        self.down3 = DoubleConv(128, 256)
        self.down4 = DoubleConv(256, 512)

        self.bottleneck = DoubleConv(512, 1024)

        self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.conv4 = DoubleConv(1024, 512)
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv3 = DoubleConv(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv2 = DoubleConv(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv1 = DoubleConv(128, 64)

        self.final = nn.Conv2d(64, out_channels, kernel_size=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    @staticmethod
    def _match_size(x: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        if x.shape[-2:] == reference.shape[-2:]:
            return x
        return F.interpolate(x, size=reference.shape[-2:], mode="bilinear", align_corners=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        down1 = self.down1(x)
        down2 = self.down2(self.pool(down1))
        down3 = self.down3(self.pool(down2))
        down4 = self.down4(self.pool(down3))

        bottleneck = self.bottleneck(self.pool(down4))

        up4 = self._match_size(self.up4(bottleneck), down4)
        conv4 = self.conv4(torch.cat([up4, down4], dim=1))
        up3 = self._match_size(self.up3(conv4), down3)
        conv3 = self.conv3(torch.cat([up3, down3], dim=1))
        up2 = self._match_size(self.up2(conv3), down2)
        conv2 = self.conv2(torch.cat([up2, down2], dim=1))
        up1 = self._match_size(self.up1(conv2), down1)
        conv1 = self.conv1(torch.cat([up1, down1], dim=1))

        return self.final(conv1)


def load_model(model_path: str | Path) -> UNet:
    model = UNet()
    checkpoint = _torch_load_cpu(Path(model_path))
    state_dict = checkpoint.get("model_state_dict", checkpoint)

    if not isinstance(state_dict, dict):
        raise ValueError("The model file does not contain a valid state dict.")

    cleaned_state_dict = {
        key.removeprefix("module."): value for key, value in state_dict.items()
    }

    model.load_state_dict(cleaned_state_dict)
    model.eval()
    return model


def _torch_load_cpu(model_path: Path):
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    try:
        return torch.load(model_path, map_location=torch.device("cpu"), weights_only=False)
    except TypeError:
        return torch.load(model_path, map_location=torch.device("cpu"))
