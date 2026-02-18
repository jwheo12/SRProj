import math

import torch
import torch.nn as nn


class SRCNN(nn.Module):
    def __init__(self, num_channels=3, feature_dim=64, map_dim=32):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(num_channels, feature_dim, (9, 9), (1, 1), (4, 4)),
            nn.ReLU(True),
        )
        self.map = nn.Sequential(
            nn.Conv2d(feature_dim, map_dim, (5, 5), (1, 1), (2, 2)),
            nn.ReLU(True),
        )
        self.reconstruction = nn.Conv2d(map_dim, num_channels, (5, 5), (1, 1), (2, 2))
        self._initialize_weights()

    def forward(self, x):
        out = self.features(x)
        out = self.map(out)
        out = self.reconstruction(out)
        return out

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.normal_(
                    module.weight.data,
                    0.0,
                    math.sqrt(2 / (module.out_channels * module.weight.data[0][0].numel())),
                )
                nn.init.zeros_(module.bias.data)
        nn.init.normal_(self.reconstruction.weight.data, 0.0, 0.001)
        nn.init.zeros_(self.reconstruction.bias.data)


class ResidualDenseBlock(nn.Module):
    def __init__(self, num_feat=64, growth_channels=32):
        super().__init__()
        self.conv1 = nn.Conv2d(num_feat, growth_channels, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat + growth_channels, growth_channels, 3, 1, 1)
        self.conv3 = nn.Conv2d(
            num_feat + growth_channels * 2, growth_channels, 3, 1, 1
        )
        self.conv4 = nn.Conv2d(
            num_feat + growth_channels * 3, growth_channels, 3, 1, 1
        )
        self.conv5 = nn.Conv2d(num_feat + growth_channels * 4, num_feat, 3, 1, 1)
        self.act = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x1 = self.act(self.conv1(x))
        x2 = self.act(self.conv2(torch.cat((x, x1), dim=1)))
        x3 = self.act(self.conv3(torch.cat((x, x1, x2), dim=1)))
        x4 = self.act(self.conv4(torch.cat((x, x1, x2, x3), dim=1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), dim=1))
        return x + x5 * 0.2


class RRDB(nn.Module):
    def __init__(self, num_feat=64, growth_channels=32):
        super().__init__()
        self.rdb1 = ResidualDenseBlock(num_feat=num_feat, growth_channels=growth_channels)
        self.rdb2 = ResidualDenseBlock(num_feat=num_feat, growth_channels=growth_channels)
        self.rdb3 = ResidualDenseBlock(num_feat=num_feat, growth_channels=growth_channels)

    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return x + out * 0.2


class RRDBRefiner(nn.Module):
    def __init__(self, num_in_ch=3, num_feat=64, num_block=16, growth_channels=32):
        super().__init__()
        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = nn.Sequential(
            *[
                RRDB(num_feat=num_feat, growth_channels=growth_channels)
                for _ in range(num_block)
            ]
        )
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_in_ch, 3, 1, 1)
        self.act = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        feat = self.conv_first(x)
        body_feat = self.conv_body(self.body(feat))
        feat = feat + body_feat
        out = self.conv_last(self.act(self.conv_hr(feat)))
        # Residual learning on bicubic-upsampled input.
        return x + out


def build_model(cfg):
    model_name = cfg.get("MODEL_NAME", "rrdb_refiner").lower()
    if model_name == "srcnn":
        return SRCNN()

    if model_name in {"rrdb", "rrdb_refiner"}:
        return RRDBRefiner(
            num_in_ch=3,
            num_feat=cfg.get("MODEL_FEATURES", 64),
            num_block=cfg.get("MODEL_BLOCKS", 16),
            growth_channels=cfg.get("MODEL_GROWTH_CHANNELS", 32),
        )

    raise ValueError(f"Unsupported MODEL_NAME: {cfg.get('MODEL_NAME')}")
