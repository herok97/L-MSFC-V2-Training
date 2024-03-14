from pathlib import Path
from torch.utils.data import Dataset
from PIL import Image
import torch
import cv2
import numpy as np
import detectron2.data.transforms as T
import random

def get_downsampled_shape(height, width, p):
    new_h = (height + p - 1) // p * p
    new_w = (width + p - 1) // p * p
    return int(new_h / p + 0.5), int(new_w / p + 0.5)

def letterbox(
    img, height=608, width=1088, color=(127.5, 127.5, 127.5)
):  # resize a rectangular image to a padded rectangular
    shape = img.shape[:2]  # shape = [height, width]
    ratio = min(float(height) / shape[0], float(width) / shape[1])
    new_shape = (
        round(shape[1] * ratio),
        round(shape[0] * ratio),
    )  # new_shape = [width, height]
    dw = (width - new_shape[0]) / 2  # width padding
    dh = (height - new_shape[1]) / 2  # height padding
    top, bottom = round(dh - 0.1), round(dh + 0.1)
    left, right = round(dw - 0.1), round(dw + 0.1)
    img = cv2.resize(img, new_shape, interpolation=cv2.INTER_AREA)  # resized, no border
    img = cv2.copyMakeBorder(
        img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )  # padded rectangular
    return img, ratio, dw, dh, shape


class OpenImageDatasetFPN(Dataset):
    def __init__(self, root, transform=None, split="train"):
        splitdir = Path(root) / split

        if not splitdir.is_dir():
            raise RuntimeError(f'Invalid directory "{root}"')

        self.samples = sorted([f for f in splitdir.iterdir() if f.is_file()])
        self.transform = transform
        self.mode = split

    def __getitem__(self, index):
        img = cv2.imread(str(self.samples[index]))
        if self.transform is not None:
            img, _ = T.apply_transform_gens(self.transform, img)
        img = torch.as_tensor(img.astype("float32").transpose(2, 0, 1))
        return img

    def __len__(self):
        return len(self.samples)
    
    
class OpenImagePexelsDKN(Dataset):
    def __init__(self, root, transform=None, split="train"):
        splitdir1 = Path(root) / "openImage" / split
        splitdir2 = Path(root) / "pexels" / split

        if not splitdir1.is_dir() or not splitdir2.is_dir():
            raise RuntimeError(f'Invalid directory "{splitdir1}" and "{splitdir2}"')

        if split == "train":
            self.samples1 = [f for f in splitdir1.iterdir() if f.is_file()]
            self.samples2 = [f for f in splitdir2.iterdir() if f.is_file()]
            self.samples = self.samples1 + self.samples2
        elif split == "val":
            self.samples = [f for f in splitdir1.iterdir() if f.is_file()] + [
                f for f in splitdir2.iterdir() if f.is_file()
            ]

        self.transform = transform
        self.mode = split

    def random_crop(self, img):
        height, width, _ = img.shape

        new_height = int(height / 2)
        new_width = int(width / 2)

        start_x = random.randint(0, width - new_width)
        start_y = random.randint(0, height - new_height)

        cropped_image = img[
            start_y : start_y + new_height, start_x : start_x + new_width
        ]

        return cropped_image

    def __getitem__(self, index):
        img0 = cv2.imread(str(self.samples[index]))
        if self.mode == "train" and index > len(self.samples1):
            img0 = self.random_crop(img0)

        img, _, _, _, shape = letterbox(img0)
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img, dtype=np.float32)
        img /= 255.0

        if self.transform is not None:
            self.transform(img)

        return {
            'img': img,
            'ori_img_shape': shape,
        }
            

    def __len__(self):
        return len(self.samples)


class PexelsDKN(Dataset):
    def __init__(self, root, transform=None, split="train"):
        splitdir = Path(root) / split

        if not splitdir.is_dir():
            raise RuntimeError(f'Invalid directory "{root}"')

        self.samples = [f for f in splitdir.iterdir() if f.is_file()]
        self.transform = transform
        self.mode = split

    def random_crop(self, img):
        height, width, _ = img.shape

        new_height = int(height / 2)
        new_width = int(width / 2)

        start_x = random.randint(0, width - new_width)
        start_y = random.randint(0, height - new_height)

        cropped_image = img[
            start_y : start_y + new_height, start_x : start_x + new_width
        ]

        return cropped_image

    def __getitem__(self, index):
        img0 = cv2.imread(str(self.samples[index]))
        # random crop with 2/3 resolution
        # img0 = self.random_crop(img0)
        img, _, _, _, shape = letterbox(img0)
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img, dtype=np.float32)
        img /= 255.0

        if self.transform is not None:
            self.transform(img)
        return {
            'img': img,
            'ori_img_shape': shape,
        }

    def __len__(self):
        return len(self.samples)


def toImgPIL(imgOpenCV):
    return Image.fromarray(cv2.cvtColor(imgOpenCV, cv2.COLOR_BGR2RGB))


def toImgOpenCV(imgPIL):  # Conver imgPIL to imgOpenCV
    i = np.array(imgPIL)  # After mapping from PIL to numpy : [R,G,B,A]
    # numpy Image Channel system: [B,G,R,A]
    red = i[:, :, 0].copy()
    i[:, :, 0] = i[:, :, 2].copy()
    i[:, :, 2] = red
    return i
