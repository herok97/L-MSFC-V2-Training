# Common
import math

import torch
import torch.nn.functional as F

# CompressAI
from compressai.ans import BufferedRansEncoder, RansDecoder
from compressai.models.priors import CompressionModel, GaussianConditional
from compressai.models.utils import update_registered_buffers
from tqdm import tqdm

# Bitstream
from .utils.stream_helper import (
    decode_feature,
    encode_feature,
    get_downsampled_shape,
)

SCALES_MIN = 0.11
SCALES_MAX = 256
SCALES_LEVELS = 64


def get_scale_table(min=SCALES_MIN, max=SCALES_MAX, levels=SCALES_LEVELS):
    return torch.exp(torch.linspace(math.log(min), math.log(max), levels))


class BaseCodec(CompressionModel):
    def __init__(self, F, N, M, task, quality, **kwargs):
        super().__init__(entropy_bottleneck_channels=N)
        self.F = int(F)
        self.N = int(N)
        self.M = int(M)
        self.task = task
        self.quality = quality
        self.gaussian_conditional = GaussianConditional(None)

    def get_feature_key_dict(self):
        if self.task in ["detection", "segmentation", "A", "B", "C", "D"]:
            return {"p2": None, "p3": None, "p4": None, "p5": None}

        elif self.task in ["TVD"]:
            return {36: None, 61: None, 74: None}

        elif self.task in ["HiEve720p", "HiEve1080p"]:
            return {75: None, 90: None, 105: None}
        else:
            raise NotImplementedError

    def load_state_dict(self, state_dict, strict=True):
        update_registered_buffers(
            self.gaussian_conditional,
            "gaussian_conditional",
            ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
            state_dict,
        )
        super().load_state_dict(state_dict, strict)

    def update(self, scale_table=None, force=False):
        if scale_table is None:
            scale_table = get_scale_table()
        updated = self.gaussian_conditional.update_scale_table(scale_table, force=force)
        updated |= super().update(force=force)
        return updated

    def cal_feature_padding_size(self, shape):
        if self.task in ["detection", "segmentation", "A", "B", "C", "D"]:
            ps_list = [64, 32, 16, 8]
        elif self.task in ["HiEve720p", "HiEve1080p", "TVD", "HiEve"]:
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

    def feature_unpadding(self, features, pad_info):
        unpaddings = pad_info["unpaddings"]
        unpadded_features = [F.pad(f, unpaddings[i]) for i, f in enumerate(features)]
        return unpadded_features
