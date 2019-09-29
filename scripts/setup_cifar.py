import os
import numpy as np
from tqdm import trange
from torchvision.datasets import CIFAR10

from common import get_carray, get_bpg


def prepare_set(mode, indices, given_set=None):
    expected_len = len(indices)
    data_path = '../prepared_data/cifar/{}'.format(mode)
    lr_data = get_carray(data_path, expected_len, 32, is_lr=True)
    hr_data = get_carray(data_path, expected_len, 32, is_lr=False)
    bpg_sizes = np.zeros((expected_len,), dtype=np.uint32)
    this_set = CIFAR10(root='~/.torch/data', train=mode != 'test', download=True) if given_set is None else given_set
    for i in trange(expected_len, desc=mode, dynamic_ncols=True):
        hr_img = this_set.data[indices[i]]
        bpg_size, bpg_lr = get_bpg(hr_img, i, remove_img=True)
        lr_data.append(bpg_lr)
        hr_data.append(hr_img.transpose((2, 0, 1)))
        bpg_sizes[i] = bpg_size
    bpg_sizes = bpg_sizes * 8.0 / 32 / 32 / 3  # bpsp
    np.savez(data_path + '.npz', bpg_size=bpg_sizes)
    lr_data.flush()
    hr_data.flush()
    return this_set


def main():
    os.makedirs('../prepared_data/cifar/', exist_ok=True)
    test_set = prepare_set('test', np.arange(10000))
    del test_set
    all_indices = np.arange(50000)
    np.random.shuffle(all_indices)
    val_indices, train_indices = all_indices[:10000], all_indices[10000:]
    train_set = prepare_set('val', val_indices)
    train_set = prepare_set('train', train_indices, train_set)
    del train_set


if __name__ == '__main__':
    main()
