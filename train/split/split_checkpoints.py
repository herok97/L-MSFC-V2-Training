import torch
from structures import FENet, DRNet, LMSFC_V2_CODEC
from pathlib import Path
import argparse, os

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--task', type=str, required=True)
    parser.add_argument('--result_path', type=str, required=True)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    split = {
        "obj": "obj",
        "seg": "seg",
        "hieve": "alt1",
        "tvd": "dn53",
    }[args.task]
    
    path = Path(args.checkpoint)
    ckpt = torch.load(path)
    if 'state_dict' in list(ckpt.keys()):
        ckpt = ckpt['state_dict']

    fenet_dict = FENet(task=split).state_dict()
    drnet_dict = DRNet(task=split).state_dict()
    inner_dict = LMSFC_V2_CODEC(task=split).state_dict()

    fenet_new_dict = {}
    drnet_new_dict = {}
    inner_new_dict = {}

    for k, v in ckpt.items():
        if k in fenet_dict:
            # print(f'{k} saved in fenet_dict')
            fenet_dict[k] = v
        elif k in drnet_dict:
            # print(f'{k} saved in drnet_dict')
            drnet_dict[k] = v
        elif k in inner_dict:
            # print(f'{k} saved in inner_dict')
            inner_dict[k] = v
        else:
            NotImplementedError
    
    os.makedirs(args.result_path, exist_ok=True)
    torch.save(fenet_dict, f"{args.result_path}/fenet_{split}.pth")
    torch.save(drnet_dict, f"{args.result_path}/drnet_{split}.pth")
    torch.save(inner_dict, f"{args.result_path}/inner_codec_{split}.pth")