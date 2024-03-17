import torch.nn.functional as F
import torch.nn as nn
import torch
import math

from modules.nn_modules import (
    AttentionBlock,
    TripleResBlock,
    conv1x1,
    deconv,
)

class DRNet(nn.Module):
    def __init__(self, N=192, M=320, split_ctx=None, quality=None, config=None):
        super().__init__()
        self.split_ctx = split_ctx
        self.quality = quality
        self.config = config
        self.levels = self.config["levels"]

        self.g_s = self.config["net"](F=self.config["F"], N=N, M=M)
        self.InverseGain = torch.nn.Parameter(
            torch.ones(size=[self.levels, M]), requires_grad=True
        )

    def get_inverse_gain(self, q):
        q = q - 1  # -0.1

        lower_index = int(math.floor(q))
        upper_index = int(math.ceil(q))
        decimal = q - lower_index
        if lower_index < 0:
            y_quant_inv = torch.abs(self.InverseGain[upper_index]) * (1 / decimal)
        else:
            y_quant_inv = torch.abs(self.InverseGain[lower_index]).pow(
                1 - decimal
            ) * torch.abs(self.InverseGain[upper_index]).pow(decimal)
        y_quant_inv = y_quant_inv.unsqueeze(0).unsqueeze(2).unsqueeze(3)
        return y_quant_inv

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

    def feature_unpadding(self, features, pad_info):
        unpaddings = pad_info["unpaddings"]
        unpadded_features = [F.pad(f, unpaddings[i]) for i, f in enumerate(features)]
        return unpadded_features

    def forward(self, y_hat, feature_size):  # feature_size: (h, w)
        pad_info = self.cal_feature_padding_size(feature_size)
        y_hat = y_hat * self.get_inverse_gain(self.quality)

        recon_features = self.g_s(y_hat)
        recon_features = self.feature_unpadding(recon_features, pad_info)
        return recon_features


class DRNet_FPN(nn.Module):
    def __init__(self, F=256, N=192, M=320) -> None:
        super().__init__()

        class FeatureMixingBlock(nn.Module):
            def __init__(self, N) -> None:
                super().__init__()
                self.conv1 = nn.Sequential(
                    nn.Conv2d(N, N, kernel_size=5, stride=2, padding=2), nn.LeakyReLU()
                )

                self.conv2 = nn.Sequential(
                    nn.Conv2d(N * 2, N, kernel_size=3, stride=1, padding=1),
                    nn.LeakyReLU(),
                )

            def forward(self, high, low):
                high = self.conv1(high)
                return self.conv2(torch.cat([high, low], dim=1)) + low

        self.p5Decoder = nn.Sequential(deconv(M, N), TripleResBlock(N), conv1x1(N, F))

        self.p4Decoder = nn.Sequential(
            deconv(M, N),
            TripleResBlock(N),
            deconv(N, N),
            TripleResBlock(N),
            conv1x1(N, F),
        )

        self.p3Decoder = nn.Sequential(
            deconv(M, N),
            TripleResBlock(N),
            deconv(N, N),
            AttentionBlock(N),
            TripleResBlock(N),
            deconv(N, N),
            TripleResBlock(N),
            conv1x1(N, F),
        )
        self.p2Decoder = nn.Sequential(
            deconv(M, N),
            TripleResBlock(N),
            deconv(N, N),
            AttentionBlock(N),
            TripleResBlock(N),
            deconv(N, N),
            TripleResBlock(N),
            deconv(N, N),
            TripleResBlock(N),
            conv1x1(N, F),
        )

        self.decoder_attention = AttentionBlock(M)

        self.fmb23 = FeatureMixingBlock(F)
        self.fmb34 = FeatureMixingBlock(F)
        self.fmb45 = FeatureMixingBlock(F)

    def forward(self, y_hat):
        y_hat = self.decoder_attention(y_hat)
        p2 = self.p2Decoder(y_hat)
        p3 = self.fmb23(p2, self.p3Decoder(y_hat))
        p4 = self.fmb34(p3, self.p4Decoder(y_hat))
        p5 = self.fmb45(p4, self.p5Decoder(y_hat))
        return [p2, p3, p4, p5]


class DRNet_DKN(nn.Module):
    def __init__(self, F=256, N=192, M=320) -> None:
        super().__init__()

        class FeatureMixingBlock(nn.Module):
            def __init__(self, F) -> None:
                super().__init__()
                self.conv1 = nn.Sequential(
                    nn.Conv2d(F, F, kernel_size=5, stride=2, padding=2), nn.LeakyReLU()
                )

                self.conv2 = nn.Sequential(
                    nn.Conv2d(F * 3, F * 2, kernel_size=3, stride=1, padding=1),
                    nn.LeakyReLU(),
                )

            def forward(self, high, low):
                high = self.conv1(high)
                return self.conv2(torch.cat([high, low], dim=1)) + low

        self.p5Decoder = nn.Sequential(
            deconv(M, N), TripleResBlock(N), conv1x1(N, F * 4)
        )

        self.p4Decoder = nn.Sequential(
            deconv(M, N),
            TripleResBlock(N),
            deconv(N, N),
            TripleResBlock(N),
            conv1x1(N, F * 2),
        )

        self.p3Decoder = nn.Sequential(
            deconv(M, N),
            TripleResBlock(N),
            deconv(N, N),
            AttentionBlock(N),
            TripleResBlock(N),
            deconv(N, N),
            TripleResBlock(N),
            conv1x1(N, F),
        )
        self.decoder_attention = AttentionBlock(M)

        self.fmb34 = FeatureMixingBlock(F)
        self.fmb45 = FeatureMixingBlock(F * 2)

    def forward(self, y_hat):
        y_hat = self.decoder_attention(y_hat)
        p3 = self.p3Decoder(y_hat)
        p4 = self.fmb34(p3, self.p4Decoder(y_hat))
        p5 = self.fmb45(p4, self.p5Decoder(y_hat))
        return [p3, p4, p5]