import argparse, math, random, sys, os

sys.path.append(
    os.path.dirname(
        os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
    )
)

from tensorboardX import SummaryWriter
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path
from omegaconf import OmegaConf

# Codec
from data.dataset import OpenImagePexelsDKN
from data.config import model_arch
from src.lmsfcv2 import LMSFC_V2_DKN_FULL

# Detectron2
from compressai_vision.config import create_vision_model
from torchvision.transforms import transforms as T

thisdir = Path(__file__).parent
config_path = str(thisdir.joinpath("../cfgs").resolve())


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Example training script.")
    # ---------------------- Main Arguments ---------------------------------|
    parser.add_argument("-t", "--task", type=str, help="TVD or HiEve")
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
        default="/data/data/openImage/",
        help="Dataset path, Note that 'openImage/train' and 'openImage/val' directories should exist",
    )
    parser.add_argument(
        "-s", "--safe_load", type=int, default=0, help="1 for safe load"
    )
    parser.add_argument(
        "-p", "--patience", type=int, default=50000, help="patience of scheduler"
    )
    parser.add_argument(
        "-o", "--only_model", type=int, default=0, help="Load only model weights"
    )

    parser.add_argument(
        "-savedir", "--savedir", type=str, default="save/", help="save_dir"
    )
    parser.add_argument("-logdir", "--logdir", type=str, default="log/", help="log_dir")
    parser.add_argument(
        "-total_step",
        "--total_step",
        default=5000000,
        type=int,
        help="total_step (default: %(default)s)",
    )
    parser.add_argument(
        "-test_step",
        "--test_step",
        default=5000,
        type=int,
        help="test_step (default: %(default)s)",
    )
    parser.add_argument(
        "-acc_step",
        "--acc_step",
        default=1,
        type=int,
        help="accumulation_step (default: %(default)s)",
    )
    parser.add_argument(
        "-save_step",
        "--save_step",
        default=50000,
        type=int,
        help="save_step (default: %(default)s)",
    )
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
        "--save", action="store_true", default=True, help="Save model to disk"
    )
    parser.add_argument(
        "--seed", type=float, default=123, help="Set random seed for reproducibility"
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


def letterCrop(
    input_image_shape, features, height=608, width=1088
):  # resize a rectangular image to a padded rectangular
    def get_downsize_multi(x, t):
        for _ in range(t):
            x = int((x+1) // 2) 
        return x
    shape = (input_image_shape[0].item(), input_image_shape[1].item())   # shape = [height, width]
    ratio = min(float(height) / shape[0], float(width) / shape[1])
    new_shape = (
        round(shape[1] * ratio),
        round(shape[0] * ratio),
    )  # new_shape = [width, height]
    # print(f"shape: {shape}\tnew_shape: {new_shape}")
    dw = (width - new_shape[0]) // 2  # width padding
    dh = (height - new_shape[1]) // 2  # height padding

    for i in range(len(features)):
        l = r = int(-get_downsize_multi(dw, i+3))
        t = b = int(-get_downsize_multi(dh, i+3))
        features[i] = torch.nn.functional.pad(
            features[i],
            (l, r, t, b),
        )
    return features

class RateDistortionLoss(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""

    def __init__(self, task):
        super().__init__()
        if task == "TVD":
            self.lmbda = {
                # For 2024-1 CTTC
                0: 0.1500,
                1: 0.4500,
                2: 1.2500,
                3: 3.0000,
                4: 7.5000,
                5: 15.0000,
                
                # m65715
                0: 0.0300,
                # 1: 0.1500,
                # 2: 0.4500,
                # 3: 1.2500,
                # 4: 3.0000,
                # 5: 7.5000,
            }
        elif task == "HiEve":
            self.lmbda = {
                # For 2024-1 CTTC
                # 0: 1.0000,
                # 1: 4.0000,
                # 2: 10.0000,
                # 3: 25.0000,
                # 4: 65.0000,
                # 5: 130.0000,
                
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


def save_feature(title, feature):
    import matplotlib.pyplot as plt
    import os
    feature = feature[0]
    feature = feature[5]
    plt.imshow(feature.detach().cpu().numpy(), cmap=plt.cm.jet)

    shrink_scale = 1.0
    aspect = feature.shape[0] / float(feature.shape[1])
    if aspect < 1.0:
        shrink_scale = aspect
    plt.colorbar(shrink=shrink_scale)
    plt.clim(-8, 8)
    plt.tight_layout()
    plt.title(title)
    plt.show()
    os.makedirs('./image_result/', exist_ok=True)
    plt.savefig(f"./image_result/{title}.png")
    plt.close()


def test(
    global_step, test_dataloader, compressor, task_model, criterion, logger, lr, args
):
    compressor.eval()
    device = next(compressor.parameters()).device
    loss = AverageMeter()
    bpp_loss = AverageMeter()
    mse_loss = AverageMeter()
    aux_loss = AverageMeter()
    psnr = AverageMeter()

    with torch.no_grad():
        for q in range(6):
            for i, x in tqdm(enumerate(test_dataloader)):
                x = x['img'].to(device)
                x = x.to(device)
                features = task_model.input_to_features([{"image": x.squeeze(0)}])[
                    "data"
                ]
                features = [features[i] for i in task_model.split_layer_list]

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
        f"Test global_step {global_step}: Average losses:"
        f"\tTest Loss: {loss.avg:.3f} |"
        f"\tTest MSE loss: {mse_loss.avg:.3f} |"
        f"\tTest PSNR: {psnr.avg:.3f} |"
        f"\tTest Bpp loss: {bpp_loss.avg:.2f} |"
        f"\tTest Aux loss: {aux_loss.avg:.2f}\n"
    )

    logger.add_scalar("Test Loss", loss.avg, global_step)
    logger.add_scalar("Test MSE loss", mse_loss.avg, global_step)
    logger.add_scalar("Test PSNR", psnr.avg, global_step)
    logger.add_scalar("Test Bpp loss", bpp_loss.avg, global_step)
    logger.add_scalar("Test Aux loss", aux_loss.avg, global_step)
    logger.add_scalar("lr", lr, global_step)
    return loss.avg


def train(
    compressor,
    task_model,
    criterion,
    train_dataloader,
    test_dataloader,
    optimizer,
    aux_optimizer,
    lr_scheduler,
    global_step,
    args,
    logger,
):
    compressor.train()
    device = next(compressor.parameters()).device
    best_loss = float("inf")

    for loop in range(100000):  # infinite loop
        for i, x in enumerate(tqdm(train_dataloader)):
            global_step += 1
            x = x['img'].to(device)
            x = x.to(device)

            # Feature extraction
            with torch.no_grad():
                features = task_model.input_to_features([{"image": x.squeeze(0)}])[
                    "data"
                ]
                features = [features[i] for i in task_model.split_layer_list]

            optimizer.zero_grad()
            aux_optimizer.zero_grad()

            q = random.randint(0, 5)
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
            psnr = 10 * (torch.log(1 * 1 / out_criterion["mse_loss"]) / math.log(10))
            # Training log
            if global_step % 50 == 0:
                tqdm.write(
                    f"Train step \t{global_step}: \t["
                    f"{i * len(x)}/{len(train_dataloader.dataset)}"
                    f" ({50. * i / len(train_dataloader):.0f}%)]"
                    f'\tLoss: {out_criterion["loss"].item():.3f} |'
                    f'\tMSE loss: {out_criterion["mse_loss"].item():.3f} |'
                    f"\tPSNR: {psnr.item():.3f} |"
                    f'\tBpp loss: {out_criterion["bpp_loss"].item():.2f} |'
                    f"\tAux loss: {aux_loss.item():.2f}"
                )

                logger.add_scalar("Loss", out_criterion["loss"].item(), global_step)
                logger.add_scalar(
                    "MSE loss", out_criterion["mse_loss"].item(), global_step
                )
                logger.add_scalar("PSNR", psnr.item(), global_step)
                logger.add_scalar(
                    "Bpp loss", out_criterion["bpp_loss"].item(), global_step
                )
                logger.add_scalar("Aux loss", aux_loss.item(), global_step)

            # validation
            if global_step % args.test_step == 0:
                loss = test(
                    global_step,
                    test_dataloader,
                    compressor,
                    task_model,
                    criterion,
                    logger,
                    optimizer.param_groups[0]["lr"],
                    args,
                )
                compressor.train()

                lr_scheduler.step(loss)

                is_best = loss < best_loss
                if is_best:
                    print("!!!!!!!!!!!BEST!!!!!!!!!!!!!")
                best_loss = min(loss, best_loss)

                if global_step % args.save_step == 0:
                    os.makedirs(args.savedir, exist_ok=True)
                    save_checkpoint(
                        {
                            "global_step": global_step,
                            "state_dict": compressor.state_dict(),
                            "loss": loss,
                            "optimizer": optimizer.state_dict(),
                            "aux_optimizer": aux_optimizer.state_dict(),
                            "lr_scheduler": lr_scheduler.state_dict(),
                        },
                        is_best,
                        filename=f"{args.savedir}/{global_step}_checkpoint.pth",
                    )

                if (
                    optimizer.param_groups[0]["lr"] <= 5e-6
                    or args.total_step == global_step
                ):
                    os.makedirs(args.savedir, exist_ok=True)
                    print(
                        f'Finished. \tcurrent lr:{optimizer.param_groups[0]["lr"]} \tglobal step:{global_step}'
                    )
                    save_checkpoint(
                        {
                            "global_step": global_step,
                            "state_dict": compressor.state_dict(),
                            "loss": loss,
                            "optimizer": optimizer.state_dict(),
                            "aux_optimizer": aux_optimizer.state_dict(),
                            "lr_scheduler": lr_scheduler.state_dict(),
                        },
                        is_best,
                        filename=f"{args.savedir}/{global_step}_checkpoint.pth",
                    )
                    exit(0)


def save_checkpoint(state, is_best, filename="checkpoint.pth"):
    torch.save(state, filename)


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


def build_dataset(args):
    # Detectron transform (relies on cv2 instead of Pillow)
    train_transforms = T.Compose(
        [
            T.ToTensor(),
            T.RandomHorizontalFlip(p=0.5),
        ]
    )
    train_dataset = OpenImagePexelsDKN(
        args.dataset, transform=train_transforms, split="train"
    )
    test_dataset = OpenImagePexelsDKN(args.dataset, split="val")

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
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


def main(argv):
    args = parse_args(argv)

    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)

    print("Dataset load")
    train_dataloader, test_dataloader = build_dataset(args)
    logger = SummaryWriter(args.logdir)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Model load")
    compressor = LMSFC_V2_DKN_FULL(task=args.task)
    compressor = compressor.to(device)

    task_model = build_darknet(args.task, device)
    global_step = 0

    optimizer, aux_optimizer = configure_optimizers(compressor, args)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=round(args.patience / args.test_step) - 1,
    )

    criterion = RateDistortionLoss(task=args.task)

    if args.checkpoint:  # load from previous checkpoint
        print("Loading", args.checkpoint)
        checkpoint = torch.load(args.checkpoint, map_location=device)
        if args.safe_load:
            safe_load_state_dict(compressor, checkpoint["state_dict"])
        else:
            compressor.load_state_dict(checkpoint["state_dict"])

        if not args.only_model:
            global_step = checkpoint["global_step"]
            optimizer.load_state_dict(checkpoint["optimizer"])
            aux_optimizer.load_state_dict(checkpoint["aux_optimizer"])
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])

    print(f"Learning rate: {optimizer.param_groups[0]['lr']}")

    train(
        compressor,
        task_model,
        criterion,
        train_dataloader,
        test_dataloader,
        optimizer,
        aux_optimizer,
        lr_scheduler,
        global_step,
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
