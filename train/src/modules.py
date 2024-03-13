import torch
import torch.nn as nn
import torchvision
from torch import Tensor
from torch.autograd import Function


class DeformableConv2d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        group,
        kernel_size=3,
        stride=1,
        padding=1,
        dilation=1,
        bias=False,
    ):
        super(DeformableConv2d, self).__init__()

        assert type(kernel_size) == tuple or type(kernel_size) == int

        kernel_size = (
            kernel_size if type(kernel_size) == tuple else (kernel_size, kernel_size)
        )
        self.stride = stride if type(stride) == tuple else (stride, stride)
        self.padding = padding
        self.dilation = dilation

        self.offset_conv = nn.Conv2d(
            in_channels,
            2 * kernel_size[0] * kernel_size[1],
            kernel_size=kernel_size,
            stride=stride,
            padding=self.padding,
            dilation=self.dilation,
            bias=True,
        )

        nn.init.constant_(self.offset_conv.weight, 0.0)
        nn.init.constant_(self.offset_conv.bias, 0.0)

        self.modulator_conv = nn.Conv2d(
            in_channels,
            1 * kernel_size[0] * kernel_size[1],
            kernel_size=kernel_size,
            stride=stride,
            padding=self.padding,
            dilation=self.dilation,
            bias=True,
        )

        nn.init.constant_(self.modulator_conv.weight, 0.0)
        nn.init.constant_(self.modulator_conv.bias, 0.0)

        self.regular_conv = nn.Conv2d(
            in_channels=in_channels // group,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=self.padding,
            dilation=self.dilation,
            bias=bias,
        )

    def forward(self, reference, offset):
        modulator = 2.0 * torch.sigmoid(self.modulator_conv(offset))
        offset = self.offset_conv(offset)
        x = torchvision.ops.deform_conv2d(
            input=reference,
            offset=offset,
            weight=self.regular_conv.weight,
            bias=self.regular_conv.bias,
            padding=self.padding,
            mask=modulator,
            stride=self.stride,
            dilation=self.dilation,
        )
        return x


def conv(in_channels, out_channels, kernel_size=5, stride=2):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=kernel_size // 2,
    )


def deconv(in_channels, out_channels, kernel_size=5, stride=2):
    return nn.ConvTranspose2d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        output_padding=stride - 1,
        padding=kernel_size // 2,
    )


def conv3x3(in_ch: int, out_ch: int, stride: int = 1) -> nn.Module:
    """3x3 convolution with padding."""
    return nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1)


def conv1x1(in_ch: int, out_ch: int, stride: int = 1) -> nn.Module:
    """1x1 convolution."""
    return nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride)


Backward_tensorGrid = [{} for i in range(8)]
Backward_tensorGrid_cpu = {}


class LowerBound(Function):
    @staticmethod
    def forward(ctx, inputs, bound):
        b = torch.ones_like(inputs) * bound
        ctx.save_for_backward(inputs, b)
        return torch.max(inputs, b)

    @staticmethod
    def backward(ctx, grad_output):
        inputs, b = ctx.saved_tensors
        pass_through_1 = inputs >= b
        pass_through_2 = grad_output < 0

        pass_through = pass_through_1 | pass_through_2
        return pass_through.type(grad_output.dtype) * grad_output, None


class GDN(nn.Module):
    def __init__(
        self, ch, inverse=False, beta_min=1e-6, gamma_init=0.1, reparam_offset=2**-18
    ):
        super(GDN, self).__init__()
        self.inverse = inverse
        self.beta_min = beta_min
        self.gamma_init = gamma_init
        self.reparam_offset = reparam_offset

        self.build(ch)

    def build(self, ch):
        self.pedestal = self.reparam_offset**2
        self.beta_bound = (self.beta_min + self.reparam_offset**2) ** 0.5
        self.gamma_bound = self.reparam_offset

        beta = torch.sqrt(torch.ones(ch) + self.pedestal)
        self.beta = nn.Parameter(beta)

        eye = torch.eye(ch)
        g = self.gamma_init * eye
        g = g + self.pedestal
        gamma = torch.sqrt(g)

        self.gamma = nn.Parameter(gamma)
        self.pedestal = self.pedestal

    def forward(self, inputs):
        unfold = False
        if inputs.dim() == 5:
            unfold = True
            bs, ch, d, w, h = inputs.size()
            inputs = inputs.view(bs, ch, d * w, h)

        _, ch, _, _ = inputs.size()

        # Beta bound and reparam
        beta = LowerBound.apply(self.beta, self.beta_bound)
        beta = beta**2 - self.pedestal

        # Gamma bound and reparam
        gamma = LowerBound.apply(self.gamma, self.gamma_bound)
        gamma = gamma**2 - self.pedestal
        gamma = gamma.view(ch, ch, 1, 1)

        # Norm pool calc
        norm_ = nn.functional.conv2d(inputs**2, gamma, beta)
        norm_ = torch.sqrt(norm_)

        # Apply norm
        if self.inverse:
            outputs = inputs * norm_
        else:
            outputs = inputs / norm_

        if unfold:
            outputs = outputs.view(bs, ch, d, w, h)
        return outputs


class ResBlock(nn.Module):
    def __init__(self, inputchannel, outputchannel, kernel_size, stride=1):
        super(ResBlock, self).__init__()
        self.relu1 = nn.ReLU()
        self.conv1 = nn.Conv2d(
            inputchannel, outputchannel, kernel_size, stride, padding=kernel_size // 2
        )
        torch.nn.init.xavier_uniform_(self.conv1.weight.data)
        torch.nn.init.constant_(self.conv1.bias.data, 0.0)
        self.relu2 = nn.ReLU()
        self.conv2 = nn.Conv2d(
            outputchannel, outputchannel, kernel_size, stride, padding=kernel_size // 2
        )
        torch.nn.init.xavier_uniform_(self.conv2.weight.data)
        torch.nn.init.constant_(self.conv2.bias.data, 0.0)
        if inputchannel != outputchannel:
            self.adapt_conv = nn.Conv2d(inputchannel, outputchannel, 1)
            torch.nn.init.xavier_uniform_(self.adapt_conv.weight.data)
            torch.nn.init.constant_(self.adapt_conv.bias.data, 0.0)
        else:
            self.adapt_conv = None

    def forward(self, x):
        x_1 = self.relu1(x)
        firstlayer = self.conv1(x_1)
        firstlayer = self.relu2(firstlayer)
        seclayer = self.conv2(firstlayer)
        if self.adapt_conv is None:
            return x + seclayer
        else:
            return self.adapt_conv(x) + seclayer


def get_p6_decoder():
    return nn.Sequential(nn.MaxPool2d(1, stride=2))


def get_paramAggregation(groups, num_slices):
    module = nn.ModuleList(
        nn.Sequential(
            conv1x1(640 + groups[i + 1 if i > 0 else 0] * 2 + groups[i + 1] * 2, 640),
            nn.ReLU(inplace=True),
            conv1x1(640, 512),
            nn.ReLU(inplace=True),
            conv1x1(512, groups[i + 1] * 2),
        )
        for i in range(num_slices)
    )  ##from checkboard "Checkerboard Context Model for Efficient Learned Image Compression"" gep网络参数
    return module


def get_context_prediction(groups, num_slices):
    module = nn.ModuleList(
        CheckboardMaskedConv2d(
            groups[i + 1], 2 * groups[i + 1], kernel_size=5, padding=2, stride=1
        )
        for i in range(num_slices)
    )  ## from https://github.com/JiangWeibeta/Checkerboard-Context-Model-for-Efficient-Learned-Image-Compression/blob/main/version2/layers/CheckerboardContext.py
    return module


def get_cc_transforms(groups, num_slices):
    module = nn.ModuleList(
        nn.Sequential(
            conv(
                groups[min(1, i) if i > 0 else 0] + groups[i if i > 1 else 0],
                224,
                stride=1,
                kernel_size=5,
            ),
            nn.ReLU(inplace=True),
            conv(224, 128, stride=1, kernel_size=5),
            nn.ReLU(inplace=True),
            conv(128, groups[i + 1] * 2, stride=1, kernel_size=5),
        )
        for i in range(1, num_slices)
    )  ## from https://github.com/tensorflow/compression/blob/master/models/ms2020.py
    return module


def get_hyper_enc_dec(N, M):
    h_a = nn.Sequential(
        conv3x3(M, N),
        nn.ReLU(inplace=True),
        conv(N, N),
        nn.ReLU(inplace=True),
        conv(N, N),
    )

    h_s = nn.Sequential(
        deconv(N, N),
        nn.ReLU(inplace=True),
        deconv(N, N * 3 // 2),
        nn.ReLU(inplace=True),
        conv3x3(N * 3 // 2, 2 * M),
    )
    return h_a, h_s


class AttentionBlock(nn.Module):
    """Self attention block.

    Simplified variant from `"Learned Image Compression with
    Discretized Gaussian Mixture Likelihoods and Attention Modules"
    <https://arxiv.org/abs/2001.01568>`_, by Zhengxue Cheng, Heming Sun, Masaru
    Takeuchi, Jiro Katto.

    Args:
        N (int): Number of channels)
    """

    def __init__(self, N: int):
        super().__init__()

        class ResidualUnit(nn.Module):
            """Simple residual unit."""

            def __init__(self):
                super().__init__()
                self.conv = nn.Sequential(
                    conv1x1(N, N // 2),
                    nn.ReLU(inplace=True),
                    conv3x3(N // 2, N // 2),
                    nn.ReLU(inplace=True),
                    conv1x1(N // 2, N),
                )
                self.relu = nn.ReLU(inplace=True)

            def forward(self, x: Tensor) -> Tensor:
                identity = x
                out = self.conv(x)
                out += identity
                out = self.relu(out)
                return out

        self.conv_a = nn.Sequential(ResidualUnit(), ResidualUnit(), ResidualUnit())

        self.conv_b = nn.Sequential(
            ResidualUnit(),
            ResidualUnit(),
            ResidualUnit(),
            conv1x1(N, N),
        )

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        a = self.conv_a(x)
        b = self.conv_b(x)
        out = a * torch.sigmoid(b)
        out += identity
        return out


class CheckboardMaskedConv2d(nn.Conv2d):
    """
    if kernel_size == (5, 5)
    then mask:
        [[0., 1., 0., 1., 0.],
        [1., 0., 1., 0., 1.],
        [0., 1., 0., 1., 0.],
        [1., 0., 1., 0., 1.],
        [0., 1., 0., 1., 0.]]
    0: non-anchor
    1: anchor
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.register_buffer("mask", torch.zeros_like(self.weight.data))

        self.mask[:, :, 0::2, 1::2] = 1
        self.mask[:, :, 1::2, 0::2] = 1

    def forward(self, x):
        self.weight.data *= self.mask
        out = super().forward(x)

        return out


class ResidualBottleneckBlock(nn.Module):
    def __init__(self, in_ch: int):
        super().__init__()
        self.conv1 = conv1x1(in_ch, in_ch // 2)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(in_ch // 2, in_ch // 2)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = conv1x1(in_ch // 2, in_ch)

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.conv3(out)

        out = out + identity
        return out


class TripleResBlock(nn.Module):
    def __init__(self, in_ch: int):
        super().__init__()
        self.res1 = ResidualBottleneckBlock(in_ch)
        self.res2 = ResidualBottleneckBlock(in_ch)
        self.res3 = ResidualBottleneckBlock(in_ch)

    def forward(self, x: Tensor) -> Tensor:
        out = self.res1(x)
        out = self.res2(out)
        out = self.res3(out)
        return out


class Quantizer:
    def quantize(self, inputs, quantize_type="noise"):
        if quantize_type == "noise":
            half = float(0.5)
            noise = torch.empty_like(inputs).uniform_(-half, half)
            inputs = inputs + noise
            return inputs
        elif quantize_type == "ste":
            return torch.round(inputs) - inputs.detach() + inputs
        else:
            return torch.round(inputs)
