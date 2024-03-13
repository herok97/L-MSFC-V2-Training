import torch

from .base import BaseCodec
from compressai.ops import ste_round

# Modules
from .modules import (
    get_cc_transforms,
    get_context_prediction,
    get_hyper_enc_dec,
    get_paramAggregation,
    Quantizer,
)
from .networks import (
    DRNet_FPN,
    FENet_FPN,
    DRNet_DKN,
    FENet_DKN,
)


class LMSFC_V2_FPN_FULL(BaseCodec):
    def __init__(self, F=256, N=192, M=320, task=None, quality=None, **kwargs):
        super().__init__(F=F, N=N, M=M, task=task, quality=quality)
        self.groups = [0, 16, 16, 32, 64, 192]  # support depth
        self.num_slices = 5
        self.levels = 8

        self.g_a = FENet_FPN(F=self.F, N=self.N, M=self.M)
        self.g_s = DRNet_FPN(F=self.F, N=self.N, M=self.M)
        self.h_a, self.h_s = get_hyper_enc_dec(self.N, self.M)

        self.cc_transforms = get_cc_transforms(self.groups, self.num_slices)
        self.context_prediction = get_context_prediction(self.groups, self.num_slices)
        self.ParamAggregation = get_paramAggregation(self.groups, self.num_slices)

        self.Gain = torch.nn.Parameter(
            torch.ones(size=[self.levels, M]), requires_grad=True
        )
        self.InverseGain = torch.nn.Parameter(
            torch.ones(size=[self.levels, M]), requires_grad=True
        )
        self.quantizer = Quantizer()
        self.stream = None

    def forward(
        self, features, noisequant=False, quality=None
    ):  # features: [p2, p3, p4, p5]
        _, _, p2_h, p2_w = features[0].shape
        pad_info = self.cal_feature_padding_size((p2_h, p2_w))
        features = self.feature_padding(features, pad_info)

        y = self.g_a(features)
        y = y * self.Gain[quality].unsqueeze(0).unsqueeze(2).unsqueeze(3)

        B, C, H, W = y.size()
        z = self.h_a(y)

        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        if not noisequant:
            z_offset = self.entropy_bottleneck._get_medians()
            z_tmp = z - z_offset
            z_hat = ste_round(z_tmp) + z_offset

        latent_means, latent_scales = self.h_s(z_hat).chunk(2, 1)

        # Charm + Ckbd
        anchor = torch.zeros_like(y).to(features[0].device)
        non_anchor = torch.zeros_like(y).to(features[0].device)

        anchor[:, :, 0::2, 0::2] = y[:, :, 0::2, 0::2]
        anchor[:, :, 1::2, 1::2] = y[:, :, 1::2, 1::2]
        non_anchor[:, :, 0::2, 1::2] = y[:, :, 0::2, 1::2]
        non_anchor[:, :, 1::2, 0::2] = y[:, :, 1::2, 0::2]

        y_slices = torch.split(y, self.groups[1:], 1)

        anchor_split = torch.split(anchor, self.groups[1:], 1)
        non_anchor_split = torch.split(non_anchor, self.groups[1:], 1)
        ctx_params_anchor_split = torch.split(
            torch.zeros(B, C * 2, H, W).to(features[0].device),
            [2 * i for i in self.groups[1:]],
            1,
        )
        y_hat_slices = []
        y_hat_slices_for_gs = []
        y_likelihood = []
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
            ##support mean and scale
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
            ### checkboard process 1
            y_anchor = anchor_split[slice_index]
            (
                means_anchor,
                scales_anchor,
            ) = self.ParamAggregation[slice_index](
                torch.concat([ctx_params_anchor_split[slice_index], support], dim=1)
            ).chunk(2, 1)

            scales_hat_split = torch.zeros_like(y_anchor).to(features[0].device)
            means_hat_split = torch.zeros_like(y_anchor).to(features[0].device)

            scales_hat_split[:, :, 0::2, 0::2] = scales_anchor[:, :, 0::2, 0::2]
            scales_hat_split[:, :, 1::2, 1::2] = scales_anchor[:, :, 1::2, 1::2]
            means_hat_split[:, :, 0::2, 0::2] = means_anchor[:, :, 0::2, 0::2]
            means_hat_split[:, :, 1::2, 1::2] = means_anchor[:, :, 1::2, 1::2]
            if noisequant:
                y_anchor_quantilized = self.quantizer.quantize(y_anchor, "noise")
                y_anchor_quantilized_for_gs = self.quantizer.quantize(y_anchor, "ste")
            else:
                y_anchor_quantilized = (
                    self.quantizer.quantize(y_anchor - means_anchor, "ste")
                    + means_anchor
                )
                y_anchor_quantilized_for_gs = (
                    self.quantizer.quantize(y_anchor - means_anchor, "ste")
                    + means_anchor
                )

            y_anchor_quantilized[:, :, 0::2, 1::2] = 0
            y_anchor_quantilized[:, :, 1::2, 0::2] = 0
            y_anchor_quantilized_for_gs[:, :, 0::2, 1::2] = 0
            y_anchor_quantilized_for_gs[:, :, 1::2, 0::2] = 0

            ### checkboard process 2
            masked_context = self.context_prediction[slice_index](y_anchor_quantilized)
            means_non_anchor, scales_non_anchor = self.ParamAggregation[slice_index](
                torch.concat([masked_context, support], dim=1)
            ).chunk(2, 1)

            scales_hat_split[:, :, 0::2, 1::2] = scales_non_anchor[:, :, 0::2, 1::2]
            scales_hat_split[:, :, 1::2, 0::2] = scales_non_anchor[:, :, 1::2, 0::2]
            means_hat_split[:, :, 0::2, 1::2] = means_non_anchor[:, :, 0::2, 1::2]
            means_hat_split[:, :, 1::2, 0::2] = means_non_anchor[:, :, 1::2, 0::2]
            # entropy estimation
            _, y_slice_likelihood = self.gaussian_conditional(
                y_slice, scales_hat_split, means=means_hat_split
            )

            y_non_anchor = non_anchor_split[slice_index]
            if noisequant:
                y_non_anchor_quantilized = self.quantizer.quantize(
                    y_non_anchor, "noise"
                )
                y_non_anchor_quantilized_for_gs = self.quantizer.quantize(
                    y_non_anchor, "ste"
                )
            else:
                y_non_anchor_quantilized = (
                    self.quantizer.quantize(y_non_anchor - means_non_anchor, "ste")
                    + means_non_anchor
                )
                y_non_anchor_quantilized_for_gs = (
                    self.quantizer.quantize(y_non_anchor - means_non_anchor, "ste")
                    + means_non_anchor
                )

            y_non_anchor_quantilized[:, :, 0::2, 0::2] = 0
            y_non_anchor_quantilized[:, :, 1::2, 1::2] = 0
            y_non_anchor_quantilized_for_gs[:, :, 0::2, 0::2] = 0
            y_non_anchor_quantilized_for_gs[:, :, 1::2, 1::2] = 0

            y_hat_slice = y_anchor_quantilized + y_non_anchor_quantilized
            y_hat_slice_for_gs = (
                y_anchor_quantilized_for_gs + y_non_anchor_quantilized_for_gs
            )
            y_hat_slices.append(y_hat_slice)
            ### ste for synthesis model
            y_hat_slices_for_gs.append(y_hat_slice_for_gs)
            y_likelihood.append(y_slice_likelihood)

        y_likelihoods = torch.cat(y_likelihood, dim=1)

        """
        use STE(y) as the input of synthesizer
        """
        y_hat = torch.cat(y_hat_slices_for_gs, dim=1)
        y_hat = y_hat * self.InverseGain[quality].unsqueeze(0).unsqueeze(2).unsqueeze(3)
        recon_p_layer_features = self.g_s(y_hat)

        recon_p_layer_features = self.feature_unpadding(
            recon_p_layer_features, pad_info
        )

        return {
            "features": recon_p_layer_features,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }


class LMSFC_V2_FPN_Encoder(BaseCodec):
    def __init__(self, F=256, N=192, M=320, task=None, quality=None, **kwargs):
        super().__init__(F=F, N=N, M=M, task=task, quality=quality)
        self.groups = [0, 16, 16, 32, 64, 192]  # support depth
        self.num_slices = 5
        self.levels = 8

        self.g_a = FENet_FPN(F=self.F, N=self.N, M=self.M)
        self.h_a, self.h_s = get_hyper_enc_dec(self.N, self.M)

        self.cc_transforms = get_cc_transforms(self.groups, self.num_slices)
        self.context_prediction = get_context_prediction(self.groups, self.num_slices)
        self.ParamAggregation = get_paramAggregation(self.groups, self.num_slices)

        self.Gain = torch.nn.Parameter(
            torch.ones(size=[self.levels, M]), requires_grad=True
        )
        self.stream = None


class LMSFC_V2_FPN_Decoder(BaseCodec):
    def __init__(self, F=256, N=192, M=320, task=None, quality=None, **kwargs):
        super().__init__(F=F, N=N, M=M, task=task, quality=quality)
        self.groups = [0, 16, 16, 32, 64, 192]  # support depth
        self.num_slices = 5
        self.levels = 8

        self.g_s = DRNet_FPN(F=self.F, N=self.N, M=self.M)
        _, self.h_s = get_hyper_enc_dec(self.N, self.M)

        self.cc_transforms = get_cc_transforms(self.groups, self.num_slices)
        self.context_prediction = get_context_prediction(self.groups, self.num_slices)
        self.ParamAggregation = get_paramAggregation(self.groups, self.num_slices)

        self.InverseGain = torch.nn.Parameter(
            torch.ones(size=[self.levels, M]), requires_grad=True
        )

        self.stream = None
        self.nbframes = None
        self.height = None
        self.width = None


class LMSFC_V2_DKN_FULL(BaseCodec):
    def __init__(self, F=256, N=192, M=320, task=None, quality=None, **kwargs):
        super().__init__(F=F, N=N, M=M, task=task, quality=quality)
        self.groups = [0, 16, 16, 32, 64, 192]  # support depth
        self.num_slices = 5
        self.levels = 6
        # self.levels = 8
        self.F = 128 if "HiEve" in self.task else 256
        print(f"Load lmsfc_v2: task={task}, quality={quality}, self.F={self.F}")

        self.g_a = FENet_DKN(F=self.F, N=self.N, M=self.M)
        self.g_s = DRNet_DKN(F=self.F, N=self.N, M=self.M)
        self.h_a, self.h_s = get_hyper_enc_dec(self.N, self.M)

        self.cc_transforms = get_cc_transforms(self.groups, self.num_slices)
        self.context_prediction = get_context_prediction(self.groups, self.num_slices)
        self.ParamAggregation = get_paramAggregation(self.groups, self.num_slices)

        self.Gain = torch.nn.Parameter(
            torch.ones(size=[self.levels, M]), requires_grad=True
        )
        self.InverseGain = torch.nn.Parameter(
            torch.ones(size=[self.levels, M]), requires_grad=True
        )
        self.quantizer = Quantizer()
        self.stream = None

    def forward(
        self, features, noisequant=False, quality=None
    ):  # features: [d1, d2, d3]
        _, _, p2_h, p2_w = features[0].shape
        pad_info = self.cal_feature_padding_size((p2_h, p2_w))
        features = self.feature_padding(features, pad_info)

        y = self.g_a(features)
        y = y * self.Gain[quality].unsqueeze(0).unsqueeze(2).unsqueeze(3)

        B, C, H, W = y.size()
        z = self.h_a(y)

        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        if not noisequant:
            z_offset = self.entropy_bottleneck._get_medians()
            z_tmp = z - z_offset
            z_hat = ste_round(z_tmp) + z_offset

        latent_means, latent_scales = self.h_s(z_hat).chunk(2, 1)

        # Charm + Ckbd
        anchor = torch.zeros_like(y).to(features[0].device)
        non_anchor = torch.zeros_like(y).to(features[0].device)

        anchor[:, :, 0::2, 0::2] = y[:, :, 0::2, 0::2]
        anchor[:, :, 1::2, 1::2] = y[:, :, 1::2, 1::2]
        non_anchor[:, :, 0::2, 1::2] = y[:, :, 0::2, 1::2]
        non_anchor[:, :, 1::2, 0::2] = y[:, :, 1::2, 0::2]

        y_slices = torch.split(y, self.groups[1:], 1)

        anchor_split = torch.split(anchor, self.groups[1:], 1)
        non_anchor_split = torch.split(non_anchor, self.groups[1:], 1)
        ctx_params_anchor_split = torch.split(
            torch.zeros(B, C * 2, H, W).to(features[0].device),
            [2 * i for i in self.groups[1:]],
            1,
        )
        y_hat_slices = []
        y_hat_slices_for_gs = []
        y_likelihood = []
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
            ##support mean and scale
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
            ### checkboard process 1
            y_anchor = anchor_split[slice_index]
            (
                means_anchor,
                scales_anchor,
            ) = self.ParamAggregation[slice_index](
                torch.concat([ctx_params_anchor_split[slice_index], support], dim=1)
            ).chunk(2, 1)

            scales_hat_split = torch.zeros_like(y_anchor).to(features[0].device)
            means_hat_split = torch.zeros_like(y_anchor).to(features[0].device)

            scales_hat_split[:, :, 0::2, 0::2] = scales_anchor[:, :, 0::2, 0::2]
            scales_hat_split[:, :, 1::2, 1::2] = scales_anchor[:, :, 1::2, 1::2]
            means_hat_split[:, :, 0::2, 0::2] = means_anchor[:, :, 0::2, 0::2]
            means_hat_split[:, :, 1::2, 1::2] = means_anchor[:, :, 1::2, 1::2]
            if noisequant:
                y_anchor_quantilized = self.quantizer.quantize(y_anchor, "noise")
                y_anchor_quantilized_for_gs = self.quantizer.quantize(y_anchor, "ste")
            else:
                y_anchor_quantilized = (
                    self.quantizer.quantize(y_anchor - means_anchor, "ste")
                    + means_anchor
                )
                y_anchor_quantilized_for_gs = (
                    self.quantizer.quantize(y_anchor - means_anchor, "ste")
                    + means_anchor
                )

            y_anchor_quantilized[:, :, 0::2, 1::2] = 0
            y_anchor_quantilized[:, :, 1::2, 0::2] = 0
            y_anchor_quantilized_for_gs[:, :, 0::2, 1::2] = 0
            y_anchor_quantilized_for_gs[:, :, 1::2, 0::2] = 0

            ### checkboard process 2
            masked_context = self.context_prediction[slice_index](y_anchor_quantilized)
            means_non_anchor, scales_non_anchor = self.ParamAggregation[slice_index](
                torch.concat([masked_context, support], dim=1)
            ).chunk(2, 1)

            scales_hat_split[:, :, 0::2, 1::2] = scales_non_anchor[:, :, 0::2, 1::2]
            scales_hat_split[:, :, 1::2, 0::2] = scales_non_anchor[:, :, 1::2, 0::2]
            means_hat_split[:, :, 0::2, 1::2] = means_non_anchor[:, :, 0::2, 1::2]
            means_hat_split[:, :, 1::2, 0::2] = means_non_anchor[:, :, 1::2, 0::2]
            # entropy estimation
            _, y_slice_likelihood = self.gaussian_conditional(
                y_slice, scales_hat_split, means=means_hat_split
            )

            y_non_anchor = non_anchor_split[slice_index]
            if noisequant:
                y_non_anchor_quantilized = self.quantizer.quantize(
                    y_non_anchor, "noise"
                )
                y_non_anchor_quantilized_for_gs = self.quantizer.quantize(
                    y_non_anchor, "ste"
                )
            else:
                y_non_anchor_quantilized = (
                    self.quantizer.quantize(y_non_anchor - means_non_anchor, "ste")
                    + means_non_anchor
                )
                y_non_anchor_quantilized_for_gs = (
                    self.quantizer.quantize(y_non_anchor - means_non_anchor, "ste")
                    + means_non_anchor
                )

            y_non_anchor_quantilized[:, :, 0::2, 0::2] = 0
            y_non_anchor_quantilized[:, :, 1::2, 1::2] = 0
            y_non_anchor_quantilized_for_gs[:, :, 0::2, 0::2] = 0
            y_non_anchor_quantilized_for_gs[:, :, 1::2, 1::2] = 0

            y_hat_slice = y_anchor_quantilized + y_non_anchor_quantilized
            y_hat_slice_for_gs = (
                y_anchor_quantilized_for_gs + y_non_anchor_quantilized_for_gs
            )
            y_hat_slices.append(y_hat_slice)
            ### ste for synthesis model
            y_hat_slices_for_gs.append(y_hat_slice_for_gs)
            y_likelihood.append(y_slice_likelihood)

        y_likelihoods = torch.cat(y_likelihood, dim=1)

        """
        use STE(y) as the input of synthesizer
        """
        y_hat = torch.cat(y_hat_slices_for_gs, dim=1)
        y_hat = y_hat * self.InverseGain[quality].unsqueeze(0).unsqueeze(2).unsqueeze(3)
        recon_p_layer_features = self.g_s(y_hat)

        recon_p_layer_features = self.feature_unpadding(
            recon_p_layer_features, pad_info
        )
        return {
            "features": recon_p_layer_features,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }


class LMSFC_V2_DKN_Encoder(BaseCodec):
    def __init__(self, F=256, N=192, M=320, task=None, quality=None, **kwargs):
        super().__init__(F=F, N=N, M=M, task=task, quality=quality)
        self.groups = [0, 16, 16, 32, 64, 192]  # support depth
        self.num_slices = 5
        self.levels = 6
        self.F = 128 if "HiEve" in self.task else 256
        print(f"Load lmsfc_v2 Encoder: task={task}, quality={quality}, self.F={self.F}")

        self.g_a = FENet_DKN(F=self.F, N=self.N, M=self.M)
        self.h_a, self.h_s = get_hyper_enc_dec(self.N, self.M)

        self.cc_transforms = get_cc_transforms(self.groups, self.num_slices)
        self.context_prediction = get_context_prediction(self.groups, self.num_slices)
        self.ParamAggregation = get_paramAggregation(self.groups, self.num_slices)

        self.Gain = torch.nn.Parameter(
            torch.ones(size=[self.levels, M]), requires_grad=True
        )
        self.stream = None


class LMSFC_V2_DKN_Decoder(BaseCodec):
    def __init__(self, F=256, N=192, M=320, task=None, quality=None, **kwargs):
        super().__init__(F=F, N=N, M=M, task=task, quality=quality)
        self.groups = [0, 16, 16, 32, 64, 192]  # support depth
        self.num_slices = 5
        self.levels = 6
        self.F = 128 if "HiEve" in self.task else 256
        print(f"Load lmsfc_v2 Decoder: task={task}, quality={quality}, self.F={self.F}")

        self.g_s = DRNet_DKN(F=self.F, N=self.N, M=self.M)
        _, self.h_s = get_hyper_enc_dec(self.N, self.M)

        self.cc_transforms = get_cc_transforms(self.groups, self.num_slices)
        self.context_prediction = get_context_prediction(self.groups, self.num_slices)
        self.ParamAggregation = get_paramAggregation(self.groups, self.num_slices)

        self.InverseGain = torch.nn.Parameter(
            torch.ones(size=[self.levels, M]), requires_grad=True
        )

        self.stream = None
        self.nbframes = None
        self.height = None
        self.width = None
