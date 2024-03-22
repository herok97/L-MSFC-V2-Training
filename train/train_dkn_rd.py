import argparse
import math
import os
import random
import sys

from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from data.config import model_arch

# Codec
from data.dataset import OpenImagePexelsDKN
from omegaconf import OmegaConf
from src.lmsfcv2 import LMSFC_V2_DKN_FULL
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from torchvision.transforms import transforms as T
from tqdm import tqdm
import numpy as np
# Detectron2
from compressai_vision.config import create_vision_model

thisdir = Path(__file__).parent
config_path = str(thisdir.joinpath("../cfgs").resolve())


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Example training script.")
    # ---------------------- Main Arguments ---------------------------------|
    parser.add_argument("-t", "--task", type=str, help="TVD or HiEve")
    parser.add_argument("-rc", "--random_crop", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=1,
    )
    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        default="/data/dataset/",
        help="Dataset path, Note that 'openImage/train' and 'openImage/val' directories should exist",
    )
    parser.add_argument(
        "-s", "--safe_load", type=int, default=0, help="1 for safe load"
    )
    parser.add_argument(
        "-o", "--only_model", type=int, default=0, help="Load only model weights"
    )

    parser.add_argument(
        "-savedir", "--savedir", type=str, default="save/", help="save_dir"
    )
    parser.add_argument("-logdir", "--logdir", type=str, default="log/", help="log_dir")
    parser.add_argument(
        "-lr",
        "--learning-rate",
        default=1e-4,
        type=float,
        help="Learning rate (default: %(default)s)",
    )
    parser.add_argument(
        "-n",
        "--num-workers",
        type=int,
        default=4,
        help="Dataloaders threads (default: %(default)s)",
    )

    parser.add_argument(
        "--aux-learning-rate",
        default=1e-3,
        help="Auxiliary loss learning rate (default: %(default)s)",
    )
    parser.add_argument(
        "--clip_max_norm",
        default=1.0,
        type=float,
        help="gradient clipping max norm default: %(default)s",
    )
    parser.add_argument("--checkpoint", type=str, help="Path to a checkpoint")
    args = parser.parse_args(argv)
    return args


class RateDistortionLoss(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""

    def __init__(self, task):
        super().__init__()
        if task == "TVD":
            self.lmbda = {
                # m65715
                0: 0.0300,
                1: 0.1500,
                2: 0.4500,
                3: 1.2500,
                4: 3.0000,
                5: 7.5000,
            }
        elif task == "HiEve":
            self.lmbda = {
                # m65715
                0: 0.5000,
                1: 1.0000,
                2: 2.0000,
                3: 3.0000,
                4: 7.5000,
                5: 15.0000,
            }

        self.mse = nn.MSELoss()
        self.weights = [0.33, 0.33, 0.33]

    def forward(self, output, target, shape, q):
        N, _, H, W = shape
        out = {}
        num_pixels = N * H * W

        out["bpp_loss"] = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output["likelihoods"].values()
        )

        out["mse_loss"] = sum(
            [
                self.mse(recon_and_gt[0], recon_and_gt[1]) * self.weights[i]
                for i, recon_and_gt in enumerate(zip(output["features"], target))
            ]
        )
        out["loss"] = self.lmbda[q] * out["mse_loss"] + out["bpp_loss"]
        return out


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


def configure_optimizers(net, args):
    """Separate parameters for the main optimizer and the auxiliary optimizer.
    Return two optimizers"""

    parameters = {
        n
        for n, p in net.named_parameters()
        if not n.endswith(".quantiles") and p.requires_grad
    }
    aux_parameters = {  # hyperprior의
        n
        for n, p in net.named_parameters()
        if n.endswith(".quantiles") and p.requires_grad
    }

    # Make sure we don't have an intersection of parameters
    params_dict = dict(net.named_parameters())
    inter_params = parameters & aux_parameters
    union_params = parameters | aux_parameters

    assert len(inter_params) == 0
    assert len(union_params) - len(params_dict.keys()) == 0

    optimizer = optim.Adam(
        (params_dict[n] for n in sorted(parameters)),
        lr=args.learning_rate,
    )
    aux_optimizer = optim.Adam(
        (params_dict[n] for n in sorted(aux_parameters)),
        lr=args.aux_learning_rate,
    )
    return optimizer, aux_optimizer


def test(epoch, test_dataloader, compressor, task_model, criterion, logger, lr):
    compressor.eval()
    device = next(compressor.parameters()).device
    loss = AverageMeter()
    bpp_loss = AverageMeter()
    mse_loss = AverageMeter()
    aux_loss = AverageMeter()
    psnr = AverageMeter()

    with torch.no_grad():
        for q in tqdm(range(6), leave=True, desc="Validation\t"):
            for i, x in tqdm(enumerate(test_dataloader), leave=False):
                x = x.to(device)
                features = task_model.input_to_features([{"image": x.squeeze(0)}])[
                    "data"
                ]
                features = [features[i] for i in features]

                out_net = compressor(features, quality=q)
                out_criterion = criterion(out_net, features, x.shape, q)

                aux_loss.update(compressor.aux_loss())
                bpp_loss.update(out_criterion["bpp_loss"])
                loss.update(out_criterion["loss"])
                mse_loss.update(out_criterion["mse_loss"])
                psnr.update(
                    10 * (torch.log(1 * 1 / out_criterion["mse_loss"]) / math.log(10))
                )

    print(
        f"\tTest Loss: {loss.avg:.3f} |"
        f"\tTest MSE loss: {mse_loss.avg:.3f} |"
        f"\tTest PSNR: {psnr.avg:.3f} |"
        f"\tTest Bpp loss: {bpp_loss.avg:.2f} |"
        f"\tTest Aux loss: {aux_loss.avg:.2f}\n"
    )

    logger.add_scalar("Test Loss", loss.avg, epoch)
    logger.add_scalar("Test MSE loss", mse_loss.avg, epoch)
    logger.add_scalar("Test PSNR", psnr.avg, epoch)
    logger.add_scalar("Test Bpp loss", bpp_loss.avg, epoch)
    logger.add_scalar("Test Aux loss", aux_loss.avg, epoch)
    logger.add_scalar("lr", lr, epoch)


def train(
    compressor,
    task_model,
    criterion,
    train_dataloader,
    test_dataloader,
    optimizer,
    aux_optimizer,
    args,
    logger,
):
    device = next(compressor.parameters()).device

    # Validation Sanity Check
    print("Sanity Check")
    test(
        -1,
        test_dataloader,
        compressor,
        task_model,
        criterion,
        logger,
        optimizer.param_groups[0]["lr"],
    )
    compressor.train()
    epoch = 0
    for epoch in tqdm(range(22), leave=True):
        # 10    ->  15      -> 18   -> 20   -> 22
        # 1e-4  ->  5e-5    -> 1e-5 -> 5e-6 -> 1e-6
        if epoch < 10:
            optimizer.param_groups[0]["lr"] = 1e-4
        elif epoch < 15:
            optimizer.param_groups[0]["lr"] = 5e-5
        elif epoch < 18:
            optimizer.param_groups[0]["lr"] = 1e-5
        elif epoch < 20:
            optimizer.param_groups[0]["lr"] = 5e-6
        elif epoch < 22:
            optimizer.param_groups[0]["lr"] = 1e-6

        global_step = epoch * len(train_dataloader)
        with tqdm(train_dataloader, leave=False) as tepoch:
            for i, x in enumerate(tepoch):
                global_step += 1
                x = x.to(device)

                with torch.no_grad():
                    features = task_model.input_to_features([{"image": x.squeeze(0)}])[
                        "data"
                    ]
                    features = [features[i] for i in features]

                optimizer.zero_grad()
                aux_optimizer.zero_grad()

                q = global_step % 6
                out_net = compressor(features, quality=q)
                out_criterion = criterion(out_net, features, x.shape, q)

                out_criterion["loss"].backward()
                if args.clip_max_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        compressor.parameters(), args.clip_max_norm
                    )
                optimizer.step()

                aux_loss = compressor.aux_loss()
                aux_loss.backward()
                aux_optimizer.step()
                psnr = 10 * (
                    torch.log(1 * 1 / out_criterion["mse_loss"]) / math.log(10)
                )

                # Training log
                if global_step % 50 == 0:
                    logger.add_scalar("Loss", out_criterion["loss"].item(), global_step)
                    logger.add_scalar(
                        "MSE", out_criterion["mse_loss"].item(), global_step
                    )
                    logger.add_scalar("PSNR", psnr.item(), global_step)
                    logger.add_scalar(
                        "Bpp", out_criterion["bpp_loss"].item(), global_step
                    )
                    logger.add_scalar("Aux loss", aux_loss.item(), global_step)

                tepoch.set_postfix(
                    loss=out_criterion["loss"].item(),
                    mse=out_criterion["mse_loss"].item(),
                    bpp=out_criterion["bpp_loss"].item(),
                    psnr=psnr.item(),
                )

            test(
                epoch,
                test_dataloader,
                compressor,
                task_model,
                criterion,
                logger,
                optimizer.param_groups[0]["lr"],
            )
            compressor.train()

            os.makedirs(args.savedir, exist_ok=True)
            torch.save(
                {
                    "epoch": epoch,
                    "state_dict": compressor.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "aux_optimizer": aux_optimizer.state_dict(),
                },
                f"{args.savedir}/epoch={epoch}_checkpoint.pth",
            )
            if os.path.exists(
                f"{args.savedir}/epoch={epoch-1}_checkpoint.pth"
            ) and os.path.exists(f"{args.savedir}/epoch={epoch}_checkpoint.pth"):
                os.remove(f"{args.savedir}/epoch={epoch-1}_checkpoint.pth")


def build_darknet(task, device):
    assert task in ["TVD", "HiEve"]
    if task == "HiEve":
        model_arch["arch"] = "jde_1088x608"
        model_arch["jde_1088x608"]["splits"] = [105, 90, 75]

    else:
        model_arch["arch"] = "jde_1088x608"
        model_arch["jde_1088x608"]["splits"] = [36, 61, 74]

    conf = OmegaConf.create(model_arch)
    task_model = create_vision_model(device, conf)
    return task_model


def build_dataset(args, seed):
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(seed)
    
    train_transforms = T.Compose(
        [
            T.ToTensor(),
            T.RandomHorizontalFlip(p=0.5),
        ]
    )
    train_dataset = OpenImagePexelsDKN(
        args.dataset, transform=train_transforms, split="train", random_crop_mode=args.random_crop
    )
    test_dataset = OpenImagePexelsDKN(args.dataset, split="val")

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        worker_init_fn=seed_worker,
        generator=g,
        shuffle=True,
        pin_memory=True,
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.test_batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=True,
    )
    return train_dataloader, test_dataloader


def manual_seeds(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    
    
def main(argv):
    manual_seeds(seed=971231)
    args = parse_args(argv)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 50)
    print(f"Dataset load:\t{args.dataset}")
    train_dataloader, test_dataloader = build_dataset(args, seed=971231)

    print(f"Model load:\t{LMSFC_V2_DKN_FULL.__name__}")
    print(f"Task:\t\t{args.task}")
    compressor = LMSFC_V2_DKN_FULL(task=args.task)
    compressor = compressor.to(device)
    print("=" * 50)

    task_model = build_darknet(args.task, device)

    optimizer, aux_optimizer = configure_optimizers(compressor, args)
    criterion = RateDistortionLoss(task=args.task)

    if args.checkpoint:
        print("Loading", args.checkpoint)
        checkpoint = torch.load(args.checkpoint, map_location=device)
        if args.safe_load:
            safe_load_state_dict(compressor, checkpoint["state_dict"])
        else:
            compressor.load_state_dict(checkpoint["state_dict"])

        if not args.only_model:
            optimizer.load_state_dict(checkpoint["optimizer"])
            aux_optimizer.load_state_dict(checkpoint["aux_optimizer"])

    logger = SummaryWriter(args.logdir)

    train(
        compressor,
        task_model,
        criterion,
        train_dataloader,
        test_dataloader,
        optimizer,
        aux_optimizer,
        args,
        logger,
    )


def safe_load_state_dict(model, pretrained_dict):
    new_model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in new_model_dict}
    new_model_dict.update(pretrained_dict)
    model.load_state_dict(new_model_dict)


if __name__ == "__main__":
    main(sys.argv[1:])
