import torch.nn.functional as F
import torch.nn as nn
import torch
import math

from modules.nn_modules import (
    AttentionBlock,
    TripleResBlock,
    conv,
)

class FENet(nn.Module):
    def __init__(self, N=192, M=320, split_ctx=None, quality=None, config=None):
        super().__init__()
        self.split_ctx = split_ctx
        self.quality = quality
        self.config = config
        self.levels = self.config["levels"]

        self.g_a = self.config["net"](F=self.config["F"], N=N, M=M)
        self.Gain = torch.nn.Parameter(
            torch.ones(size=[self.levels, M]), requires_grad=True
        )

    def get_gain(self, q):
        # if q = 0.9
        q = q - 1  # -0.1
        lower_index = int(math.floor(q))  # -1
        upper_index = int(math.ceil(q))  # 0
        decimal = q - lower_index  # 0.9

        if lower_index < 0:
            y_quant = torch.abs(self.Gain[upper_index]) * decimal
        else:
            y_quant = torch.abs(self.Gain[lower_index]).pow(1 - decimal) * torch.abs(
                self.Gain[upper_index]
            ).pow(decimal)
        y_quant = y_quant.unsqueeze(0).unsqueeze(2).unsqueeze(3)
        return y_quant

    def cal_feature_padding_size(self, shape):
        if self.split_ctx in ["obj", "seg"]:
            ps_list = [64, 32, 16, 8]
        elif self.split_ctx in ["alt1", "dn53"]:
            ps_list = [32, 16, 8]
        else:
            raise NotImplementedError
        ori_size = []
        paddings = []
        unpaddings = []
        padded_size = []

        ori_size.append(shape)
        for i in range(len(ps_list) - 1):
            h, w = ori_size[-1]
            ori_size.append(((h + 1) // 2, (w + 1) // 2))

        for i, ps in enumerate(ps_list):
            h = ori_size[i][0]
            w = ori_size[i][1]

            h_pad_len = ps - h % ps if h % ps != 0 else 0
            w_pad_len = ps - w % ps if w % ps != 0 else 0

            paddings.append(
                (
                    w_pad_len // 2,
                    w_pad_len - w_pad_len // 2,
                    h_pad_len // 2,
                    h_pad_len - h_pad_len // 2,
                )
            )
            unpaddings.append(
                (
                    0 - (w_pad_len // 2),
                    0 - (w_pad_len - w_pad_len // 2),
                    0 - (h_pad_len // 2),
                    0 - (h_pad_len - h_pad_len // 2),
                )
            )

        for i, p in enumerate(paddings):
            h = ori_size[i][0]
            w = ori_size[i][1]
            h_pad_len = p[2] + p[3]
            w_pad_len = p[0] + p[1]
            padded_size.append((h + h_pad_len, w + w_pad_len))

        return {
            "ori_size": ori_size,
            "paddings": paddings,
            "unpaddings": unpaddings,
            "padded_size": padded_size,
        }

    def feature_padding(self, features, pad_info):
        paddings = pad_info["paddings"]
        padded_features = [
            F.pad(f, paddings[i], mode="reflect") for i, f in enumerate(features)
        ]
        return padded_features

    def forward(self, features):
        _, _, h, w = features[0].shape
        pad_info = self.cal_feature_padding_size((h, w))
        features = self.feature_padding(features, pad_info)

        y = self.g_a(features)
        y = y * self.get_gain(self.quality)
        return {
            "y": y,
            "maxChShape": (h, w),
        }


class FENet_FPN(nn.Module):
    def __init__(self, F, N, M) -> None:
        super().__init__()
        self.block1 = nn.Sequential(conv(F, N), TripleResBlock(N))

        self.block2 = nn.Sequential(
            conv(F + N, N),
            TripleResBlock(N),
            AttentionBlock(N),
        )

        self.block3 = nn.Sequential(
            conv(F + N, N),
            TripleResBlock(N),
        )

        self.block4 = nn.Sequential(
            conv(F + N, M),
            AttentionBlock(M),
        )

    def forward(self, p_layer_features):
        # p_layer_features contains padded features p2, p3, p4, p5
        p2, p3, p4, p5 = tuple(p_layer_features)
        y = self.block1(p2)
        y = self.block2(torch.cat([y, p3], dim=1))
        y = self.block3(torch.cat([y, p4], dim=1))
        y = self.block4(torch.cat([y, p5], dim=1))
        return y


class FENet_DKN(nn.Module):
    def __init__(self, F, N, M) -> None:
        super().__init__()
        self.block1 = nn.Sequential(
            conv(F, N),
            TripleResBlock(N),
            AttentionBlock(N),
        )

        self.block2 = nn.Sequential(
            conv(2 * F + N, N),
            TripleResBlock(N),
        )

        self.block3 = nn.Sequential(
            conv(4 * F + N, M),
            AttentionBlock(M),
        )

    def forward(self, p_layer_features):
        p3, p4, p5 = tuple(p_layer_features)
        y = self.block1(p3)
        y = self.block2(torch.cat([y, p4], dim=1))
        y = self.block3(torch.cat([y, p5], dim=1))
        return y
