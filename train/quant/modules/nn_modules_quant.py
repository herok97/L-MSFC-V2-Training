from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


def qconv3x3(
    in_channels: int, out_channels: int, stride: int = 1, device="cpu"
) -> nn.Module:
    return QuantConv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=3,
        stride=stride,
        padding=1,
        device=device,
    )


def qconv1x1(
    in_channels: int, out_channels: int, stride: int = 1, device="cpu"
) -> nn.Module:
    return QuantConv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=1,
        stride=stride,
        device=device,
    )


def qconv(in_channels, out_channels, kernel_size=5, stride=2, device="cpu"):
    return QuantConv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=kernel_size // 2,
        device=device,
    )


def qdeconv(in_channels, out_channels, kernel_size=5, stride=2, device="cpu"):
    return QuantConvTranspose2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        output_padding=stride - 1,
        padding=kernel_size // 2,
        device=device,
    )


def calfloatbits(value: Tensor, bitdepth: int) -> Tensor:
    intv = torch.ceil(value).int() + 1
    intbits = torch.ceil(torch.log2(intv)).int()
    floatbits = bitdepth - intbits - 1
    return torch.exp2(floatbits)


def prepare_quant_modules(modules, nbit):
    for module in modules:
        if isinstance(module, nn.ModuleList):
            for sub_module in module:
                if isinstance(sub_module, nn.Sequential):
                    for layer in sub_module:
                        layer.prepare_quant(nbit)
                else:
                    sub_module.prepare_quant(nbit)
        else:
            if isinstance(module, nn.Sequential):
                for layer in module:
                    layer.prepare_quant(nbit)
            else:
                module.prepare_quant(nbit)


def quantize_modules(modules):
    for module in modules:
        if isinstance(module, nn.ModuleList):
            for sub_module in module:
                if isinstance(sub_module, nn.Sequential):
                    for layer in sub_module:
                        layer.quant_weights()
                else:
                    sub_module.quant_weights()
        else:
            if isinstance(module, nn.Sequential):
                for layer in module:
                    layer.quant_weights()
            else:
                module.quant_weights()


class QuantModule(nn.Module):
    def register_quant_v_params(self, device=None):
        self.register_buffer(
            "value_max", torch.zeros(1, device=device, requires_grad=False)
        )
        self.register_buffer(
            "value_min", torch.zeros(1, device=device, requires_grad=False)
        )
        self.register_buffer(
            "scale_v", torch.ones(1, device=device, requires_grad=False)
        )
        self.register_buffer("nbit", torch.ones(1, device=device, requires_grad=False))

    def register_quant_w_params(self, out_ch, device):
        pass

    def prepare_quant(self, nbit: int) -> None:
        self.nbit = self.nbit * nbit
        self.calibration_forward = True

    def quant_input(self, x: Tensor) -> Tensor:
        x = x.double()
        x = torch.round(x * self.scale_v).clamp(
            -(2 ** (self.nbit - 1)), 2 ** (self.nbit - 1) - 1
        )
        return x

    def compute_scale_v(self) -> None:
        v_max = torch.max(torch.abs(self.value_max), torch.abs(self.value_min))
        return calfloatbits(v_max, self.nbit)

    def compute_scale_c(self, transpose=False) -> None:
        out_dim = 0 if transpose else 1
        w_max = torch.max(self.weight, dim=out_dim, keepdim=True).values
        w_max = torch.max(w_max, dim=2, keepdim=True).values
        w_max = torch.max(w_max, dim=3, keepdim=True).values

        w_min = torch.min(self.weight, dim=out_dim, keepdim=True).values
        w_min = torch.min(w_min, dim=2, keepdim=True).values
        w_min = torch.min(w_min, dim=3, keepdim=True).values

        c_max = torch.max(
            torch.cat([torch.abs(w_max), torch.abs(w_min)], dim=3), dim=3, keepdim=True
        ).values
        return calfloatbits(c_max, self.nbit)


class QuantConv2d(nn.Conv2d, QuantModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        device = kwargs["device"] if "device" in kwargs else "cpu"
        self.register_quant_v_params(device)
        self.register_quant_w_params(kwargs["out_channels"], device)

    def register_quant_w_params(self, out_ch, device):
        self.register_buffer(
            "scale_c", torch.ones([1, out_ch, 1, 1], device=device, requires_grad=False)
        )

    def quant_weights(self):
        self.scale_v = self.compute_scale_v()
        self.scale_c = self.compute_scale_c()
        self.weight = nn.Parameter(torch.round(self.weight * self.scale_c))
        self.scale_c = self.scale_c.permute(1, 0, 2, 3)

    def quant_forward(self, x):
        self.double()
        x = self._conv_forward(x, self.weight, bias=None)  # without bias
        x = x / (self.scale_v * self.scale_c) + self.bias.unsqueeze(0).unsqueeze(
            -1
        ).unsqueeze(-1)
        return x.float()

    def forward(self, input: Tensor) -> Tensor:
        if hasattr(self, "calibration_forward"):
            self.value_max = torch.max(
                torch.cat([torch.max(input).reshape((1)), self.value_max], dim=0)
            ).reshape((1))
            self.value_min = torch.min(
                torch.cat([torch.min(input).reshape((1)), self.value_min], dim=0)
            ).reshape((1))
            return self._conv_forward(input, self.weight, self.bias)
        else:
            out = self.quant_input(input)
            return self.quant_forward(out)


class QuantConvTranspose2d(nn.ConvTranspose2d, QuantModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        device = kwargs["device"] if "device" in kwargs else "cpu"
        self.register_quant_v_params(device)
        self.register_quant_w_params(kwargs["out_channels"], device)

    def register_quant_w_params(self, out_ch, device):
        self.register_buffer(
            "scale_c", torch.ones([1, out_ch, 1, 1], device=device, requires_grad=False)
        )

    def quant_weights(self) -> None:
        self.scale_v = self.compute_scale_v()
        self.scale_c = self.compute_scale_c(transpose=True)
        self.weight = nn.Parameter(torch.round(self.weight * self.scale_c))

    def quant_forward(self, x: Tensor) -> Tensor:
        self.double()
        x = self._conv_transpose2d_forward(x, self.weight, bias=None)  # without bias
        x = x / (self.scale_v * self.scale_c) + self.bias.unsqueeze(0).unsqueeze(
            -1
        ).unsqueeze(-1)
        return x.float()

    def forward(self, input: Tensor) -> Tensor:
        if hasattr(self, "calibration_forward"):
            self.value_max = torch.max(
                torch.cat([torch.max(input).reshape((1)), self.value_max], dim=0)
            ).reshape((1))
            self.value_min = torch.min(
                torch.cat([torch.min(input).reshape((1)), self.value_min], dim=0)
            ).reshape((1))
            return self._conv_transpose2d_forward(input, self.weight, self.bias)
        else:
            out = self.quant_input(input)
            return self.quant_forward(out)

    def _conv_transpose2d_forward(
        self,
        input: Tensor,
        weight: Tensor,
        bias: Optional[Tensor],
        output_size: Optional[List[int]] = None,
    ) -> Tensor:
        if self.padding_mode != "zeros":
            raise ValueError(
                "Only `zeros` padding mode is supported for ConvTranspose2d"
            )

        assert isinstance(self.padding, tuple)
        # One cannot replace List by Tuple or Sequence in "_output_padding" because
        # TorchScript does not support `Sequence[T]` or `Tuple[T, ...]`.
        num_spatial_dims = 2
        output_padding = self._output_padding(
            input,
            output_size,
            self.stride,
            self.padding,
            self.kernel_size,  #  type: ignore[arg-type]
            num_spatial_dims,
            self.dilation,
        )  # type: ignore[arg-type]

        return F.conv_transpose2d(
            input,
            weight,
            bias,
            self.stride,
            self.padding,
            output_padding,
            self.groups,
            self.dilation,
        )


class QuantLeakyReLU(nn.LeakyReLU, QuantModule):
    def __init__(self, negative_slope: float = 1e-2, inplace: bool = False) -> None:
        super().__init__(negative_slope, inplace)
        self.register_quant_v_params()

    def quant_weights(self) -> None:
        self.scale_v = self.compute_scale_v()

    def quant_forward(self, x: Tensor) -> Tensor:
        x = torch.round(F.leaky_relu(x, self.negative_slope, self.inplace)) / (
            self.scale_v
        )
        return x.float()

    def forward(self, input: Tensor) -> Tensor:
        if hasattr(self, "calibration_forward"):
            self.value_max = torch.max(
                torch.cat([torch.max(input).reshape((1)), self.value_max], dim=0)
            ).reshape((1))
            self.value_min = torch.min(
                torch.cat([torch.min(input).reshape((1)), self.value_min], dim=0)
            ).reshape((1))
            return F.leaky_relu(input, self.negative_slope, self.inplace)
        else:
            out = self.quant_input(input)
            return self.quant_forward(out)


class QuantReLU(nn.ReLU, QuantModule):
    def __init__(self, inplace: bool = False) -> None:
        super().__init__(inplace)
        self.register_quant_v_params()

    def quant_weights(self) -> None:
        self.scale_v = self.compute_scale_v()

    def quant_forward(self, x: Tensor) -> Tensor:
        x = torch.round(F.relu(x, self.inplace)) / (self.scale_v)
        return x.float()

    def forward(self, input: Tensor) -> Tensor:
        if hasattr(self, "calibration_forward"):
            self.value_max = torch.max(
                torch.cat([torch.max(input).reshape((1)), self.value_max], dim=0)
            ).reshape((1))
            self.value_min = torch.min(
                torch.cat([torch.min(input).reshape((1)), self.value_min], dim=0)
            ).reshape((1))
            return F.relu(input, self.inplace)
        else:
            out = self.quant_input(input)
            return self.quant_forward(out)


class QuantCheckboardMaskedConv2d(QuantConv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.register_buffer("mask", torch.zeros_like(self.weight.data))

        self.mask[:, :, 0::2, 1::2] = 1
        self.mask[:, :, 1::2, 0::2] = 1

    def forward(self, x):
        self.weight.data *= self.mask
        out = super().forward(x)

        return out
