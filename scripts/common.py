import os

import bcolz
import imageio as io
import numpy as np


def get_size_of_file(path):  # in bytes
    """Return the size of a file in bytes."""
    return os.stat(path).st_size


def get_bpg(img, index, remove_img=False):
    """
    Encode an image to BPG format, decode it back, and return the file size and reconstructed image.
    Optionally removes intermediate files.
    """
    img_file_name = "orig_{}.png".format(index)
    bpg_file_name = "orig_{}.bpg".format(index)
    dec_file_name = "orig_{}_dec.png".format(index)
    if not os.path.exists(img_file_name):
        io.imsave(img_file_name, img)
    # NOTE needs libpng-dev, yasm, libjpeg-dev, libsdl-image1.2-dev, cmake
    # NOTE that it doesn't work on 8x8 or 16x16 pictures
    os.system("../libbpg/bpgenc {} -o {}".format(img_file_name, bpg_file_name))
    file_size = get_size_of_file(bpg_file_name)
    os.system("../libbpg/bpgdec {} -o {}".format(bpg_file_name, dec_file_name))
    recon = io.imread(dec_file_name).transpose((2, 0, 1))
    os.system("rm {}".format(dec_file_name))
    os.system("rm {}".format(bpg_file_name))
    if remove_img:
        os.system("rm {}".format(img_file_name))
    return file_size, recon


def get_carray(name, expected_len, img_size, is_lr):
    """
    Create a bcolz carray for storing images with the given parameters.
    """
    return bcolz.carray(
        np.zeros((0, 3, img_size, img_size), dtype=np.uint8),
        expectedlen=expected_len,
        mode="w",
        rootdir="{}_{}".format(name, "lr" if is_lr else "hr"),
    )
