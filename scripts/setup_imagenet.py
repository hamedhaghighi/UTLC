import os
import pickle

import numpy as np
from common import get_bpg, get_carray
from tqdm import trange


# NOTE the pickle files are already shuffled and there is no need to reshuffle them
def process(
    mode, img_size, last_dataset_index=0, last_dataset_offset=0, is_train=False
):
    expected_len = dict(train=1281167 - 20000, val=20000, test=50000)[mode]
    path = "/hdd/Danial/Compression_Dataset/prepared_imagenet64/{}".format(mode)
    lr_data = get_carray(path, expected_len, img_size, is_lr=True)
    hr_data = get_carray(path, expected_len, img_size, is_lr=False)
    bpg_sizes = np.zeros((expected_len,), dtype=np.uint32)
    if mode == "test":
        with open(
            os.path.expanduser("~/.torch/data/imagenet_{}/val_data".format(img_size)),
            "rb",
        ) as f:
            dataset = pickle.load(f)["data"].reshape((-1, 3, img_size, img_size))
    else:
        with open(
            os.path.expanduser(
                "/hdd/Danial/Compression_Dataset/Imagenet64_train/train_data_batch_{}".format(
                    1 + last_dataset_index
                )
            ),
            "rb",
        ) as f:
            dataset = pickle.load(f)["data"][last_dataset_offset:].reshape(
                (-1, 3, img_size, img_size)
            )
    current_offset = 0
    for i in trange(expected_len, desc=mode, dynamic_ncols=True):
        hr_img = dataset[i - current_offset].transpose(1, 2, 0)
        bpg_size, bpg_lr = get_bpg(hr_img, i, remove_img=True)
        lr_data.append(bpg_lr)
        hr_data.append(dataset[i - current_offset])
        bpg_sizes[i] = bpg_size
        if i - current_offset + 1 == len(dataset):
            current_offset = i + 1
            last_dataset_index += 1
            with open(
                os.path.expanduser(
                    "/hdd/Danial/Compression_Dataset/Imagenet{}_train/train_data_batch_{}".format(
                        img_size, 1 + last_dataset_index
                    )
                ),
                "rb",
            ) as f:
                dataset = pickle.load(f)["data"].reshape((-1, 3, img_size, img_size))
        last_dataset_offset = i - current_offset + 1
    bpg_sizes = bpg_sizes * 8.0 / img_size / img_size / 3  # bpsp
    np.savez("{}.npz".format(path), bpg_size=bpg_sizes)
    lr_data.flush()
    hr_data.flush()
    return last_dataset_index, last_dataset_offset


def main(img_size):
    path = "/hdd/Danial/Compression_Dataset/Imagenet64_train"
    if not os.path.exists(os.path.expanduser(path)):
        raise ValueError(
            "please download and place imagenet_{} in {}".format(img_size, path)
        )
    os.makedirs("/hdd/Danial/Compression_Dataset/prepared_imagenet64/", exist_ok=True)
    # _, _ = process('test', img_size) # I have already created the test and val on my computer
    ldi, ldo = process("val", img_size)
    print("\n\nldi: {}, ldo: {}\n\n".format(ldi, ldo))
    # ldi, ldo = 0, 20000
    _, _ = process("train", img_size, ldi, ldo, True)


if __name__ == "__main__":
    main(64)
