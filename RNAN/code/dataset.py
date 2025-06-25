import os

import bcolz
import numpy as np
import torch
from torch.utils.data import Dataset


class ContextualizedDecompressionDataset(Dataset):
    """
    PyTorch Dataset for contextualized decompression.
    Loads low-res and high-res images and BPG sizes from bcolz and npz files.
    """

    def __init__(self, mode, path, name):
        super().__init__()
        path = os.path.join(path, "{}/{}".format(name, mode))
        self.lr_data = bcolz.open(rootdir=path + "_lr", mode="r")
        self.hr_data = bcolz.open(rootdir=path + "_hr", mode="r")
        data = np.load(path + ".npz")
        self.bpg_size = data["bpg_size"]

    def __len__(self):
        return len(self.bpg_size)

    def get_image(self, img):
        # Normalize image to [-1, 1]
        return img.astype(np.float32) / 127.5 - 1

    def __getitem__(self, index):
        # Return normalized low-res, high-res, and BPG size for a sample
        return (
            self.get_image(self.lr_data[index]),
            self.get_image(self.hr_data[index]),
            self.bpg_size[index],
        )


class SeqDataLoader:
    """
    Sequential data loader for batching dataset samples.
    Supports finite and infinite iteration over batches.
    """

    def __init__(self, dataset, batch_size, drop_last, finite=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.size = (
            int(np.ceil(len(dataset) / batch_size))
            if not drop_last
            else (len(dataset) // batch_size)
        )
        self.counter = 0
        self.finite = finite

    def __iter__(self):
        self.counter = 0
        return self

    def get_image(self, imgs):
        # Normalize images to [-1, 1]
        return imgs.astype(np.float32) / 127.5 - 1

    def __next__(self):
        self.counter += 1
        if self.counter > self.size:
            if self.finite:
                raise StopIteration
            else:
                self.counter = 1
        start_index = (self.counter - 1) * self.batch_size
        end_index = self.counter * self.batch_size
        # Return batch of normalized low-res, high-res, and BPG sizes as tensors
        return (
            torch.from_numpy(
                self.get_image(self.dataset.lr_data[start_index:end_index])
            ),
            torch.from_numpy(
                self.get_image(self.dataset.hr_data[start_index:end_index])
            ),
            torch.from_numpy(self.dataset.bpg_size[start_index:end_index]),
        )


class ContextualizedDecompressionLoader:
    """
    Loader for contextualized decompression datasets.
    Provides train and validation loaders and reference BPSP values.
    """

    bpsp_dict = dict(
        tiny=dict(
            FLIF=3.86,
        ),
        cifar=dict(
            SparseTransformer=2.80,
            PixelCNN=3.03,  # 30%
            IDF=3.34,  # 23% improvement
            GLOW=3.35,
            FLIF=4.37,
        ),
        imagenet_32=dict(
            ImageTransformer=3.77,
            PixelCNN=3.83,
            MultiScalePixelCNN=3.95,
            IDF=4.18,
            L3C=4.76,
            FLIF=5.09,
        ),
        imagenet_64=dict(
            SparseTransformer=3.44,
            PixelCNN=3.57,
            MultiScalePixelCNN=3.70,
            GLOW=3.81,
            IDF=3.90,
            L3C=4.42,
            FLIF=4.55,
        ),
    )

    def __init__(self, name, batch_size=10, path="../../prepared_data/"):
        assert name in {
            "tiny",
            "imagenet_32",
            "imagenet_64",
            "cifarQ40",
            "cifarQ49",
            "cifar",
        }
        self.name = name
        self.dataset_train = ContextualizedDecompressionDataset("train", path, name)
        self.dataset_val = ContextualizedDecompressionDataset("val", path, name)
        self.loader_train = SeqDataLoader(
            self.dataset_train, batch_size=batch_size, drop_last=True, finite=False
        )
        self.loader_val = SeqDataLoader(
            self.dataset_val, batch_size=batch_size, drop_last=False, finite=True
        )

    def get_bpsp(self):
        # Return reference bits-per-subpixel values for the dataset
        return self.bpsp_dict[self.name]
