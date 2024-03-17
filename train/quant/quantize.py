import argparse, random, sys, os, glob
import torch
from pathlib import Path
import torch.nn.functional as F
from tqdm import tqdm
from build import (
    build_nn_part1,
    build_fenet,
    build_inner_codec,
    build_feature_list,
)

        
def parse_args(argv):
    parser = argparse.ArgumentParser(description="Arguments for an experiment")
    # ---------------------- Main Arguments ---------------------------------|
    parser.add_argument("--split_ctx", type=str, help="dn53 (tvd), alt1 (hieve), obj, seg")
    parser.add_argument("--dataset", type=str, help="Path to the dataset")
    parser.add_argument("--fenet_weight", type=str, help="Path to the fenet checkpoint")
    parser.add_argument("--inner_codec_weight", type=str, help="Path to the inner_codec checkpoint")
    parser.add_argument("--inner_codec_quant_weight", type=str, help="Path to the inner_codec_quant checkpoint")
    parser.add_argument("--device", type=str, help="device for encoding")
    args = parser.parse_args()
    return args
    
def main(argv):
    args = parse_args(argv)
    torch.manual_seed(123)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    random.seed(123)

    print("Build NN-Part1, FENet, Inner Codec")
    nn_part1 = build_nn_part1(args.split_ctx, args.device)
    fenet = build_fenet(args.split_ctx, args.fenet_weight, args.device)
    inner_codec = build_inner_codec(args.split_ctx, args.inner_codec_weight, args.device,  quant=True)
    feature_list = build_feature_list(nn_part1, args.split_ctx, args.dataset, args.device)

    print("== Preparing Quantization ==")
    inner_codec.prepare_quant_modules(nbit=16)
    
    print("========= Encoding ==========")
    for i, features in tqdm(enumerate(feature_list)):
        out_fenet = fenet(features)
        inner_codec.encode(1, out_fenet["y"], "tmp.bin", out_fenet["maxChShape"])
    

    print("== Quantization ==")
    inner_codec.quantize_model()

    # Save quantized model
    os.remove("tmp.bin")
    torch.save(inner_codec.state_dict(), args.inner_codec_quant_weight)    

if __name__ == "__main__":
    main(sys.argv[1:])
