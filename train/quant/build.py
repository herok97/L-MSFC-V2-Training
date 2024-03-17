import glob
import torch
import cv2
import numpy as np
from omegaconf import OmegaConf

# NN-Part1
from compressai_vision.config import create_vision_model
from utils import model_arch, letterbox

# Codec
from modules.drnet import DRNet, DRNet_DKN, DRNet_FPN
from modules.fenet import FENet, FENet_DKN, FENet_FPN
from modules.inner_codec import SCCTX
from modules.inner_codec_quant import QuantSCCTX

def build_nn_part1(split_ctx, device):
    assert split_ctx in ["obj", "seg", "dn53", "alt1"]
    if split_ctx == "alt1":
        model_arch["arch"] = "jde_1088x608"
        model_arch["jde_1088x608"]["splits"] = [105, 90, 75]
    elif split_ctx == "dn53":
        model_arch["arch"] = "jde_1088x608"
        model_arch["jde_1088x608"]["splits"] = [36, 61, 74]
    elif split_ctx == "seg":
        model_arch["arch"] = "mask_rcnn_X_101_32x8d_FPN_3x"
    elif split_ctx == "obj":
        model_arch["arch"] = "faster_rcnn_X_101_32x8d_FPN_3x"
    else:
        raise NotImplementedError
    conf = OmegaConf.create(model_arch)
    nn_part1 = create_vision_model(device, conf)
    return nn_part1


def get_module_config(split_ctx, part):
    assert split_ctx in ["obj", "seg", "dn53", "alt1"]
    assert part in ["enc", "dec"]
    config_dict = {
        "obj" : {
          "F": 256,
          "levels": 8,
          "net": FENet_FPN if part == "enc" else DRNet_FPN
        },
        "seg" : {
          "F": 256,
          "levels": 8,
          "net": FENet_FPN if part == "enc" else DRNet_FPN
        },
        "alt1" : {
          "F": 128,
          "levels": 6,
          "net": FENet_DKN if part == "enc" else DRNet_DKN
        },
        "dn53" : {
          "F": 256,
          "levels": 6,
          "net": FENet_DKN if part == "enc" else DRNet_DKN
        },
    }
    return config_dict[split_ctx]

def build_fenet(split_ctx, weights, device):
    config = get_module_config(split_ctx, "enc")
    fenet = FENet(N=192, M=320, split_ctx=split_ctx, quality=config["levels"], config=config).to(device)
    state_dict = torch.load(weights, map_location=device)
    fenet.load_state_dict(state_dict, strict=True)
    fenet.eval()
    return fenet

def build_drnet(split_ctx, weights, device):
    config = get_module_config(split_ctx, "dec")
    drnet = DRNet(N=192, M=320, split_ctx=split_ctx, quality=config["levels"], config=config).to(device)
    state_dict = torch.load(weights, map_location=device)
    drnet.load_state_dict(state_dict, strict=True)
    drnet.eval()
    return drnet

def build_inner_codec(split_ctx, weights, device, quant=False):
    state_dict = torch.load(weights, map_location=device)

    if quant:
        inner_codec = QuantSCCTX(N=192, M=320, task=split_ctx).to(device)
        inner_codec.load_state_dict(state_dict, strict=False)
    else:
        inner_codec = SCCTX(N=192, M=320, task=split_ctx).to(device)
        inner_codec.load_state_dict(state_dict)
    inner_codec.eval()
    return inner_codec

def build_feature_list(nn_part1, split_ctx, path, device):
    assert split_ctx in ["obj", "seg", "dn53", "alt1"]

    path_list = sorted(glob.glob(path + "/*.jpg"))
    tensor_list = []

    for path in path_list:
        x = cv2.imread(path)
        if split_ctx in ["obj", "seg"]:
            x = torch.as_tensor(x.astype("float32").transpose(2, 0, 1), device=device)
            features = nn_part1.input_to_features([{"image": x.squeeze(0)}])["data"]
            features = [features[f"p{i}"] for i in range(2, 6)]
        else:
            x, _, _, _, _ = letterbox(x)
            x = x[:, :, ::-1].transpose(2, 0, 1)
            x = torch.Tensor(np.ascontiguousarray(x, dtype=np.float32) / 255.0, device='cpu')
            features = nn_part1.input_to_features([{"image": x.squeeze(0)}])["data"]
            features = [features[i].to(device) for i in nn_part1.split_layer_list]
            
        tensor_list.append(features)

    return tensor_list