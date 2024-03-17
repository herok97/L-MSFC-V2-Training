import argparse, random, sys, os, glob
import torch
import matplotlib.pyplot as plt
from pathlib import Path
import torch.nn.functional as F
from tqdm import tqdm
from build import (
    build_nn_part1,
    build_fenet,
    build_drnet,
    build_inner_codec,
    build_feature_list,
)

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

class AverageMeter:
    """Compute running average."""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
def parse_args(argv):
    parser = argparse.ArgumentParser(description="Arguments for an experiment")
    # ---------------------- Main Arguments ---------------------------------|
    parser.add_argument("--split_ctx", type=str, help="dn53 (tvd), alt1 (hieve), obj, seg")
    parser.add_argument("--dataset", type=str, help="Path to the dataset")
    parser.add_argument("--fenet_weight", type=str, help="Path to the fenet checkpoint")
    parser.add_argument("--drnet_weight", type=str, help="Path to the drnet checkpoint")
    parser.add_argument("--inner_codec_weight", type=str, help="Path to the inner_codec checkpoint")
    parser.add_argument("--bin_path", type=str, help="Path to the binary file")
    parser.add_argument("--exp_name", type=str, help="Experiment name")
    parser.add_argument("--enc_device", type=str, help="device for encoding")
    parser.add_argument("--dec_device", type=str, help="device for decoding")
    parser.add_argument("--encode", type=str2bool, nargs='?', const=True, default=False, help="Activate encoding")
    parser.add_argument("--decode", type=str2bool, nargs='?', const=True, default=False, help="Activate encoding")
    parser.add_argument("--quant", type=str2bool, nargs='?', const=True, default=True, help="quant")
    args = parser.parse_args()
    return args

def save_features(filename, titles, recon_features, features, mae_list):
    fig, axes = plt.subplots(2, len(recon_features), figsize=(20, 8), dpi=300)

    for i, (title, recon_feature, feature) in enumerate(zip(titles, recon_features, features)):
        ax_gt = axes[0, i]
        ax_recon = axes[1, i]
        
        b, c, h, w = feature.shape

        feature = torch.sum(feature.squeeze(0), dim=0) / c
        recon_feature = torch.sum(recon_feature.squeeze(0), dim=0) / c
        im_gt = ax_gt.imshow(feature.detach().cpu().numpy(), cmap=plt.cm.jet)
        im_recon = ax_recon.imshow(recon_feature.detach().cpu().numpy(), cmap=plt.cm.jet)
        ax_gt.set_title(title + f" (MAE: {mae_list[i]})")
        
        shrink_scale = 1.0
        aspect = feature.shape[0] / float(feature.shape[1])
        if aspect < 1.0:
            shrink_scale = aspect
        fig.colorbar(im_gt, ax=ax_gt, shrink=shrink_scale)
        fig.colorbar(im_recon, ax=ax_recon, shrink=shrink_scale)

    plt.savefig(filename)
    

def encode(args):
    print("Build NN-Part1, FENet, Inner Codec")
    nn_part1 = build_nn_part1(args.split_ctx, args.enc_device)
    fenet = build_fenet(args.split_ctx, args.fenet_weight, args.enc_device)
    inner_codec = build_inner_codec(args.split_ctx, args.inner_codec_weight, args.enc_device, args.quant)
    feature_list = build_feature_list(nn_part1, args.split_ctx, args.dataset, args.enc_device)
    
    bin_path = Path(args.bin_path) / args.exp_name / "bin"
    os.makedirs(bin_path, exist_ok=True)

    print("========= Encoding ==========")
    for i, features in tqdm(enumerate(feature_list)):
        out_fenet = fenet(features)
        inner_codec.encode(1, out_fenet["y"], bin_path / f"{i+1}.bin", out_fenet["maxChShape"])
    
    
def decode(args):
    print("DRNet, Inner Codec")
    drnet = build_drnet(args.split_ctx, args.drnet_weight, args.dec_device)
    inner_codec = build_inner_codec(args.split_ctx, args.inner_codec_weight, args.dec_device, args.quant)
    
    bin_path = Path(args.bin_path) / args.exp_name / "bin"
    bin_list = sorted(glob.glob(str(bin_path) + "/*.bin"))
    recon_feature_list = []

    print("========= Decoding ==========")
    for i, bin in tqdm(enumerate(bin_list)):
        y, maxChShape = inner_codec.decode(bin)
        features = drnet(y, maxChShape)
        recon_feature_list.append(features)
    return recon_feature_list

def evaluate(recon_feature_list, args):
    nn_part1 = build_nn_part1(args.split_ctx, args.enc_device)
    feature_list = build_feature_list(nn_part1, args.split_ctx, args.dataset, args.dec_device)
    mae = AverageMeter()
    titles = {
        'obj': ['p2', 'p3', 'p4', 'p5'],
        'seg': ['p2', 'p3', 'p4', 'p5'],
        'dn53': ['f1', 'f2', 'f3'],
        'alt1': ['f1', 'f2', 'f3'],
    }    
    print("========= Evaluate ==========")
    path = Path(args.bin_path) / args.exp_name / "images"
    os.makedirs(path, exist_ok=True)
    for i, (recon_features, features) in tqdm(enumerate(zip(recon_feature_list, feature_list))):
        mae_list = []
        for rf, f in zip(recon_features, features):
            mae_ = F.l1_loss(rf.cuda(), f.cuda()).item()
            mae.update(mae_)
            mae_list.append(round(mae_, 3))
        save_features(path / f"{i+1}.png", titles[args.split_ctx], recon_features, features, mae_list)
    print(mae.avg)
    
def main(argv):
    args = parse_args(argv)
    torch.manual_seed(123)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    random.seed(123)

    if args.encode:
        encode(args)

    if args.decode:
        recon_feature_list = decode(args)
        evaluate(recon_feature_list, args)


if __name__ == "__main__":
    main(sys.argv[1:])
