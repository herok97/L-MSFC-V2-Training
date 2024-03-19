import torch
from base import FENet_DRNet_Base
from compressai.models.priors import CompressionModel, GaussianConditional
from compressai.models.utils import update_registered_buffers
from modules import (
    get_cc_transforms,
    get_context_prediction,
    get_hyper_enc_dec,
    get_paramAggregation,
)


class FENet(FENet_DRNet_Base):
    def __init__(self, N=192, M=320, task=None, quality=None):
        super().__init__(task=task)
        self.quality = quality
        self.config = self.get_config_from_task_id(task, type="fenet")
        self.levels = self.config["levels"]

        self.g_a = self.config["net"](F=self.config["F"], N=N, M=M)
        self.Gain = torch.nn.Parameter(
            torch.ones(size=[self.levels, M]), requires_grad=True
        )


class DRNet(FENet_DRNet_Base):
    def __init__(self, N=192, M=320, task=None, quality=None):
        super().__init__(task=task)
        self.quality = quality
        self.config = self.get_config_from_task_id(task, type="drnet")
        self.levels = self.config["levels"]

        self.g_s = self.config["net"](F=self.config["F"], N=N, M=M)
        self.InverseGain = torch.nn.Parameter(
            torch.ones(size=[self.levels, M]), requires_grad=True
        )


class LMSFC_V2_CODEC(CompressionModel):
    def __init__(self, N=192, M=320, task=None, **kwargs):
        super().__init__(entropy_bottleneck_channels=N)
        self.N = int(N)
        self.M = int(M)

        self.task = task
        self.gaussian_conditional = GaussianConditional(None)
        self.groups = [0, 16, 16, 32, 64, 192]
        self.num_slices = 5  # support depth

        self.h_a, self.h_s = get_hyper_enc_dec(self.N, self.M)
        self.cc_transforms = get_cc_transforms(self.groups, self.num_slices)
        self.context_prediction = get_context_prediction(self.groups, self.num_slices)
        self.ParamAggregation = get_paramAggregation(self.groups, self.num_slices)

        self.stream = None

    def load_state_dict(self, state_dict):
        update_registered_buffers(
            self.gaussian_conditional,
            "gaussian_conditional",
            ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
            state_dict,
        )
        super().load_state_dict(state_dict)
