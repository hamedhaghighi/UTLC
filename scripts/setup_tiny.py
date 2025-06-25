import os
from glob import glob

import numpy as np
from common import get_bpg, get_carray
from skimage import io
from skimage.color import grey2rgb
from tqdm import trange


def process(path, mode):
    if mode == "train":
        image_addresses = "train/*/images/*.JPEG"
    else:
        image_addresses = "{}/images/*.JPEG".format(mode)
    image_addresses = glob(os.path.join(path, image_addresses), recursive=True)
    if mode != "test":
        np.random.shuffle(image_addresses)
    expected_len = len(image_addresses)
    data_path = "../prepared_data/tiny/{}".format(mode)
    lr_data = get_carray(data_path, expected_len, 64, is_lr=True)
    hr_data = get_carray(data_path, expected_len, 64, is_lr=False)
    bpg_sizes = np.zeros((expected_len,), dtype=np.uint32)
    flif_sizes = np.zeros((expected_len,), dtype=np.uint32)
    for i in trange(expected_len, desc=mode, dynamic_ncols=True):
        hr_img = io.imread(image_addresses[i])
        if hr_img.ndim == 2:
            hr_img = grey2rgb(hr_img)
        bpg_size, bpg_lr = get_bpg(hr_img, i, remove_img=True)
        lr_data.append(bpg_lr)
        hr_data.append(hr_img.transpose((2, 0, 1)))
        bpg_sizes[i] = bpg_size
    bpg_sizes = bpg_sizes * 8.0 / 64 / 64 / 3  # bpsp
    np.savez(data_path + ".npz", bpg_size=bpg_sizes)
    lr_data.flush()
    hr_data.flush()


def main(path):
    path = os.path.expanduser(path)
    if not os.path.exists(path):
        raise ValueError("please download and place tiny-imagenet in {}".format(path))
    os.makedirs("../prepared_data/tiny/", exist_ok=True)
    process(path, "test")
    process(path, "val")
    process(path, "train")


if __name__ == "__main__":
    main("~/.torch/data/tiny-imagenet-200/")
