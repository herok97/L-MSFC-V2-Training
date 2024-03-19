import math

import torch
import torch.nn as nn
from compressai.ans import BufferedRansEncoder, RansDecoder
from compressai.models.priors import CompressionModel, GaussianConditional
from compressai.models.utils import update_registered_buffers
from modules.nn_modules import conv, conv3x3
from modules.nn_modules_quant import (
    QuantCheckboardMaskedConv2d,
    QuantReLU,
    prepare_quant_modules,
    qconv,
    qconv1x1,
    qconv3x3,
    qdeconv,
    quantize_modules,
)
from tqdm import tqdm

from utils.readwrite import decode_feature, encode_feature, get_downsampled_shape


class QuantSCCTX(CompressionModel):
    def __init__(self, N=192, M=320, task=None, **kwargs):
        super().__init__(entropy_bottleneck_channels=N)
        self.N = int(N)
        self.M = int(M)

        self.task = task
        self.gaussian_conditional = GaussianConditional(None)
        self.groups = [0, 16, 16, 32, 64, 192]
        self.num_slices = 5

        self.h_a = nn.Sequential(
            conv3x3(M, N),
            nn.ReLU(inplace=True),
            conv(N, N),
            nn.ReLU(inplace=True),
            conv(N, N),
        )

        self.h_s = nn.Sequential(
            qdeconv(N, N),
            QuantReLU(inplace=True),
            qdeconv(N, N * 3 // 2),
            QuantReLU(inplace=True),
            qconv3x3(N * 3 // 2, 2 * M),
        )

        self.cc_transforms = nn.ModuleList(
            nn.Sequential(
                qconv(
                    self.groups[min(1, i) if i > 0 else 0]
                    + self.groups[i if i > 1 else 0],
                    224,
                    stride=1,
                    kernel_size=5,
                ),
                QuantReLU(inplace=True),
                qconv(224, 128, stride=1, kernel_size=5),
                QuantReLU(inplace=True),
                qconv(128, self.groups[i + 1] * 2, stride=1, kernel_size=5),
            )
            for i in range(1, self.num_slices)
        )
        self.context_prediction = nn.ModuleList(
            QuantCheckboardMaskedConv2d(
                in_channels=self.groups[i + 1],
                out_channels=2 * self.groups[i + 1],
                kernel_size=5,
                padding=2,
                stride=1,
            )
            for i in range(self.num_slices)
        )
        self.ParamAggregation = nn.ModuleList(
            nn.Sequential(
                qconv1x1(
                    640
                    + self.groups[i + 1 if i > 0 else 0] * 2
                    + self.groups[i + 1] * 2,
                    640,
                ),
                QuantReLU(inplace=True),
                qconv1x1(640, 512),
                QuantReLU(inplace=True),
                qconv1x1(512, self.groups[i + 1] * 2),
            )
            for i in range(self.num_slices)
        )

        self.stream = None

    def prepare_quant_modules(self, nbit):
        prepare_quant_modules(
            modules=[
                self.h_s,
                self.cc_transforms,
                self.context_prediction,
                self.ParamAggregation,
            ],
            nbit=nbit,
        )

    def quantize_model(self):
        quantize_modules(
            modules=[
                self.h_s,
                self.cc_transforms,
                self.context_prediction,
                self.ParamAggregation,
            ]
        )

    def compress(self, y):
        B, C, H, W = y.size()
        z = self.h_a(y)

        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        latent_means, latent_scales = self.h_s(z_hat).chunk(2, 1)
        y_slices = torch.split(y, self.groups[1:], 1)

        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.reshape(-1).int().tolist()
        offsets = self.gaussian_conditional.offset.reshape(-1).int().tolist()
        encoder = BufferedRansEncoder()
        symbols_list = []
        indexes_list = []
        y_strings = []
        y_hat_slices = []

        ctx_params_anchor_split = torch.split(
            torch.zeros(B, C * 2, H, W).to(y.device),
            [2 * i for i in self.groups[1:]],
            1,
        )

        for slice_index, y_slice in enumerate(y_slices):
            if slice_index == 0:
                support_slices = []
            elif slice_index == 1:
                support_slices = y_hat_slices[0]
                support_slices_ch = self.cc_transforms[slice_index - 1](support_slices)
                (
                    support_slices_ch_mean,
                    support_slices_ch_scale,
                ) = support_slices_ch.chunk(2, 1)

            else:
                support_slices = torch.concat(
                    [y_hat_slices[0], y_hat_slices[slice_index - 1]], dim=1
                )
                support_slices_ch = self.cc_transforms[slice_index - 1](support_slices)
                (
                    support_slices_ch_mean,
                    support_slices_ch_scale,
                ) = support_slices_ch.chunk(2, 1)
            support = (
                torch.concat([latent_means, latent_scales], dim=1)
                if slice_index == 0
                else torch.concat(
                    [
                        support_slices_ch_mean,
                        support_slices_ch_scale,
                        latent_means,
                        latent_scales,
                    ],
                    dim=1,
                )
            )
            y_anchor = y_slices[slice_index].clone()

            (means_anchor, scales_anchor,) = self.ParamAggregation[slice_index](
                torch.concat([ctx_params_anchor_split[slice_index], support], dim=1)
            ).chunk(2, 1)

            B_anchor, C_anchor, H_anchor, W_anchor = y_anchor.size()

            y_anchor_encode = torch.zeros(
                B_anchor, C_anchor, H_anchor, W_anchor // 2
            ).to(y.device)
            means_anchor_encode = torch.zeros(
                B_anchor, C_anchor, H_anchor, W_anchor // 2
            ).to(y.device)
            scales_anchor_encode = torch.zeros(
                B_anchor, C_anchor, H_anchor, W_anchor // 2
            ).to(y.device)
            y_anchor_decode = torch.zeros(B_anchor, C_anchor, H_anchor, W_anchor).to(
                y.device
            )

            y_anchor_encode[:, :, 0::2, :] = y_anchor[:, :, 0::2, 0::2]
            y_anchor_encode[:, :, 1::2, :] = y_anchor[:, :, 1::2, 1::2]
            means_anchor_encode[:, :, 0::2, :] = means_anchor[:, :, 0::2, 0::2]
            means_anchor_encode[:, :, 1::2, :] = means_anchor[:, :, 1::2, 1::2]
            scales_anchor_encode[:, :, 0::2, :] = scales_anchor[:, :, 0::2, 0::2]
            scales_anchor_encode[:, :, 1::2, :] = scales_anchor[:, :, 1::2, 1::2]

            indexes_anchor = self.gaussian_conditional.build_indexes(
                scales_anchor_encode
            )
            anchor_quantized = self.gaussian_conditional.quantize(
                y_anchor_encode, "symbols", means_anchor_encode
            )

            symbols_list.extend(anchor_quantized.reshape(-1).tolist())
            indexes_list.extend(indexes_anchor.reshape(-1).tolist())
            y_anchor_decode[:, :, 0::2, 0::2] = (
                anchor_quantized[:, :, 0::2, :] + means_anchor_encode[:, :, 0::2, :]
            )
            y_anchor_decode[:, :, 1::2, 1::2] = (
                anchor_quantized[:, :, 1::2, :] + means_anchor_encode[:, :, 1::2, :]
            )

            masked_context = self.context_prediction[slice_index](y_anchor_decode)
            means_non_anchor, scales_non_anchor = self.ParamAggregation[slice_index](
                torch.concat([masked_context, support], dim=1)
            ).chunk(2, 1)

            y_non_anchor_encode = torch.zeros(
                B_anchor, C_anchor, H_anchor, W_anchor // 2
            ).to(y.device)
            means_non_anchor_encode = torch.zeros(
                B_anchor, C_anchor, H_anchor, W_anchor // 2
            ).to(y.device)
            scales_non_anchor_encode = torch.zeros(
                B_anchor, C_anchor, H_anchor, W_anchor // 2
            ).to(y.device)
            y_non_anchor_decode = torch.zeros(
                B_anchor, C_anchor, H_anchor, W_anchor
            ).to(y.device)

            non_anchor = y_slices[slice_index].clone()
            y_non_anchor_encode[:, :, 0::2, :] = non_anchor[:, :, 0::2, 1::2]
            y_non_anchor_encode[:, :, 1::2, :] = non_anchor[:, :, 1::2, 0::2]
            means_non_anchor_encode[:, :, 0::2, :] = means_non_anchor[:, :, 0::2, 1::2]
            means_non_anchor_encode[:, :, 1::2, :] = means_non_anchor[:, :, 1::2, 0::2]
            scales_non_anchor_encode[:, :, 0::2, :] = scales_non_anchor[
                :, :, 0::2, 1::2
            ]
            scales_non_anchor_encode[:, :, 1::2, :] = scales_non_anchor[
                :, :, 1::2, 0::2
            ]

            indexes_non_anchor = self.gaussian_conditional.build_indexes(
                scales_non_anchor_encode
            )
            non_anchor_quantized = self.gaussian_conditional.quantize(
                y_non_anchor_encode, "symbols", means_non_anchor_encode
            )

            symbols_list.extend(non_anchor_quantized.reshape(-1).tolist())
            indexes_list.extend(indexes_non_anchor.reshape(-1).tolist())
            y_non_anchor_decode[:, :, 0::2, 1::2] = (
                non_anchor_quantized[:, :, 0::2, :]
                + means_non_anchor_encode[:, :, 0::2, :]
            )
            y_non_anchor_decode[:, :, 1::2, 0::2] = (
                non_anchor_quantized[:, :, 1::2, :]
                + means_non_anchor_encode[:, :, 1::2, :]
            )

            y_slice_hat = y_anchor_decode + y_non_anchor_decode
            y_hat_slices.append(y_slice_hat)

        encoder.encode_with_indexes(
            symbols_list, indexes_list, cdf, cdf_lengths, offsets
        )
        y_string = encoder.flush()
        y_strings.append(y_string)
        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}

    def decompress(self, strings, h, w):
        assert isinstance(strings, list) and len(strings) == 2
        pad_info = self.cal_feature_padding_size((h, w))
        padded_h = pad_info["padded_size"][0][0]
        padded_w = pad_info["padded_size"][0][1]

        if self.task in ["obj", "seg"]:
            downsampled_ratio = 64
        elif self.task in ["alt1", "dn53"]:
            downsampled_ratio = 32
        else:
            raise NotImplementedError

        z_shape = get_downsampled_shape(padded_h, padded_w, downsampled_ratio)
        z_hat = self.entropy_bottleneck.decompress(strings[1], z_shape)
        B, _, _, _ = z_hat.size()

        latent_means, latent_scales = self.h_s(z_hat).chunk(2, 1)

        y_strings = strings[0][0]

        ctx_params_anchor = torch.zeros(
            (B, self.M * 2, z_hat.shape[2] * 4, z_hat.shape[3] * 4)
        ).to(z_hat.device)
        ctx_params_anchor_split = torch.split(
            ctx_params_anchor, [2 * i for i in self.groups[1:]], 1
        )

        y_hat_slices = []
        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.reshape(-1).int().tolist()
        offsets = self.gaussian_conditional.offset.reshape(-1).int().tolist()
        decoder = RansDecoder()
        decoder.set_stream(y_strings)

        for slice_index in range(len(self.groups) - 1):
            if slice_index == 0:
                support_slices = []
            elif slice_index == 1:
                support_slices = y_hat_slices[0]
                support_slices_ch = self.cc_transforms[slice_index - 1](support_slices)
                (
                    support_slices_ch_mean,
                    support_slices_ch_scale,
                ) = support_slices_ch.chunk(2, 1)

            else:
                support_slices = torch.concat(
                    [y_hat_slices[0], y_hat_slices[slice_index - 1]], dim=1
                )
                support_slices_ch = self.cc_transforms[slice_index - 1](support_slices)
                (
                    support_slices_ch_mean,
                    support_slices_ch_scale,
                ) = support_slices_ch.chunk(2, 1)
            support = (
                torch.concat([latent_means, latent_scales], dim=1)
                if slice_index == 0
                else torch.concat(
                    [
                        support_slices_ch_mean,
                        support_slices_ch_scale,
                        latent_means,
                        latent_scales,
                    ],
                    dim=1,
                )
            )
            (means_anchor, scales_anchor,) = self.ParamAggregation[slice_index](
                torch.concat([ctx_params_anchor_split[slice_index], support], dim=1)
            ).chunk(2, 1)

            B_anchor, C_anchor, H_anchor, W_anchor = means_anchor.size()

            means_anchor_encode = torch.zeros(
                B_anchor, C_anchor, H_anchor, W_anchor // 2
            ).to(z_hat.device)
            scales_anchor_encode = torch.zeros(
                B_anchor, C_anchor, H_anchor, W_anchor // 2
            ).to(z_hat.device)

            means_anchor_encode[:, :, 0::2, :] = means_anchor[:, :, 0::2, 0::2]
            means_anchor_encode[:, :, 1::2, :] = means_anchor[:, :, 1::2, 1::2]
            scales_anchor_encode[:, :, 0::2, :] = scales_anchor[:, :, 0::2, 0::2]
            scales_anchor_encode[:, :, 1::2, :] = scales_anchor[:, :, 1::2, 1::2]

            indexes_anchor = self.gaussian_conditional.build_indexes(
                scales_anchor_encode
            )
            anchor_quantized = decoder.decode_stream(
                indexes_anchor.reshape(-1).tolist(), cdf, cdf_lengths, offsets
            )
            anchor_quantized = (
                torch.Tensor(anchor_quantized)
                .reshape(scales_anchor_encode.shape)
                .to(scales_anchor_encode.device)
            )
            y_anchor_decode = torch.zeros(B_anchor, C_anchor, H_anchor, W_anchor).to(
                z_hat.device
            )

            y_anchor_decode[:, :, 0::2, 0::2] = (
                anchor_quantized[:, :, 0::2, :] + means_anchor_encode[:, :, 0::2, :]
            )
            y_anchor_decode[:, :, 1::2, 1::2] = (
                anchor_quantized[:, :, 1::2, :] + means_anchor_encode[:, :, 1::2, :]
            )

            masked_context = self.context_prediction[slice_index](y_anchor_decode)
            means_non_anchor, scales_non_anchor = self.ParamAggregation[slice_index](
                torch.concat([masked_context, support], dim=1)
            ).chunk(2, 1)

            means_non_anchor_encode = torch.zeros(
                B_anchor, C_anchor, H_anchor, W_anchor // 2
            ).to(z_hat.device)
            scales_non_anchor_encode = torch.zeros(
                B_anchor, C_anchor, H_anchor, W_anchor // 2
            ).to(z_hat.device)

            means_non_anchor_encode[:, :, 0::2, :] = means_non_anchor[:, :, 0::2, 1::2]
            means_non_anchor_encode[:, :, 1::2, :] = means_non_anchor[:, :, 1::2, 0::2]
            scales_non_anchor_encode[:, :, 0::2, :] = scales_non_anchor[
                :, :, 0::2, 1::2
            ]
            scales_non_anchor_encode[:, :, 1::2, :] = scales_non_anchor[
                :, :, 1::2, 0::2
            ]

            indexes_non_anchor = self.gaussian_conditional.build_indexes(
                scales_non_anchor_encode
            )
            non_anchor_quantized = decoder.decode_stream(
                indexes_non_anchor.reshape(-1).tolist(), cdf, cdf_lengths, offsets
            )
            non_anchor_quantized = (
                torch.Tensor(non_anchor_quantized)
                .reshape(scales_non_anchor_encode.shape)
                .to(scales_non_anchor_encode.device)
            )
            y_non_anchor_decode = torch.zeros(
                B_anchor, C_anchor, H_anchor, W_anchor
            ).to(z_hat.device)
            y_non_anchor_decode[:, :, 0::2, 1::2] = (
                non_anchor_quantized[:, :, 0::2, :]
                + means_non_anchor_encode[:, :, 0::2, :]
            )
            y_non_anchor_decode[:, :, 1::2, 0::2] = (
                non_anchor_quantized[:, :, 1::2, :]
                + means_non_anchor_encode[:, :, 1::2, :]
            )

            y_slice_hat = y_anchor_decode + y_non_anchor_decode
            y_hat_slices.append(y_slice_hat)

        y_hat = torch.cat(y_hat_slices, dim=1)
        return {"y_hat": y_hat}

    def encode(self, nbframes, y, output_path, tShape):
        self.update()
        self.height, self.width = tShape

        # y: [C, H, W]
        for i in range(nbframes):
            y_i = y[i].unsqueeze(0)
            self._encode(nbframes, y_i, output_path, self.height, self.width)
        self.reset()

    @torch.no_grad()
    def _encode(self, nbframes, y, output_path, height, width):
        encoded = self.compress(y)
        y_string = encoded["strings"][0][0]
        z_string = encoded["strings"][1][0]
        self.stream = encode_feature(
            nbframes, height, width, y_string, z_string, output_path, self.stream
        )

    def reset(self):
        assert self.stream is not None
        assert self.height is not None
        assert self.width is not None
        self.stream.close()
        self.stream = self.height = self.width = None

    def decode(self, bitstream_fd):
        self.update()

        y_hat_list = []

        # set self.nbframes
        y_hat_list.append(self._decode(bitstream_fd)["y_hat"])

        for it in range(self.nbframes - 1):
            y_hat_list.append(self._decode(bitstream_fd)["y_hat"])

        maxChShape = (self.height, self.width)
        self.reset()
        return torch.cat(y_hat_list, dim=0), maxChShape

    @torch.no_grad()
    def _decode(self, bitstream_path):
        nbframes, height, width, y_string, z_string, stream = decode_feature(
            bitstream_path, self.stream
        )
        self.stream = stream
        if nbframes is not None and height is not None and width is not None:
            self.nbframes, self.height, self.width = nbframes, height, width
        decoded = self.decompress([y_string, z_string], self.height, self.width)
        return decoded

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
            scale_table = torch.exp(torch.linspace(math.log(0.11), math.log(256), 64))
        updated = self.gaussian_conditional.update_scale_table(scale_table, force=force)
        updated |= super().update(force=force)
        return updated

    def cal_feature_padding_size(self, shape):
        if self.task in ["obj", "seg"]:
            ps_list = [64, 32, 16, 8]
        elif self.task in ["alt1", "dn53"]:
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
