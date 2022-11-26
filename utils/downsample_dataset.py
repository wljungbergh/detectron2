import os
from PIL import Image

import rawpy
import numpy as np
from tqdm.contrib.concurrent import process_map
from tqdm import tqdm
import cv2

SCALE = 5
DOWNSAMPLED_WIDTH = 1206
DOWNSAMPLED_HEIGHT = 802


ROOT = "/workspaces/s0001387/raw_od/pascal_raw/pascal_raw"


def downsample_and_store_jpeg(image_path: str):
    image = Image.open(image_path)
    gray_img = image.convert("L")

    for grey in [True, False]:
        folder = os.path.dirname(image_path).replace("full", "fifth")
        if grey:
            folder = folder.replace("jpg", "grey")
        filename = os.path.basename(image_path)
        new_filename = os.path.join(folder, filename)
        if os.path.exists(new_filename):
            continue

        os.makedirs(folder, exist_ok=True)
        os.chmod(folder, 0o777)
        if grey:
            image_ = gray_img
        else:
            image_ = image
        image_resized = image_.resize(
            (DOWNSAMPLED_WIDTH, DOWNSAMPLED_HEIGHT), Image.BILINEAR
        )
        image_resized.save(new_filename)


def _downsample(kernel: np.ndarray) -> np.ndarray:
    assert kernel.shape[0] == kernel.shape[1]
    kernel_size = kernel.shape[0]
    half_size = kernel_size // 2
    quater_size = kernel_size // 4

    r = kernel[0:half_size:2, 0:half_size:2].mean()
    b = kernel[half_size::2, half_size::2].mean()
    g1 = (
        kernel[0:half_size:2, half_size::2].sum()
        + kernel[quater_size, half_size + quater_size]
    ) / ((quater_size + 1) ** 2 + 1)
    g2 = (
        kernel[half_size::2, 0:half_size:2].sum()
        + kernel[half_size + quater_size, quater_size]
    ) / ((quater_size + 1) ** 2 + 1)

    return np.array([[r, g1], [g2, b]]).astype(np.uint16)


def downsample_and_store_raw(image_path: str):
    folder = os.path.dirname(image_path).replace("full", "fifth")
    filename = os.path.basename(image_path).replace(".nef", ".npy")
    new_filename = os.path.join(folder, filename)

    if os.path.exists(new_filename):
        return

    with rawpy.imread(image_path) as raw:
        # make sure that it has the same size as jpg without removing any partial bayer patterns
        raw_image = raw.raw_image_visible

        raw_image_downsampled = downsample_raw_image(raw_image, scale=SCALE)
        # store as numpy array

        os.makedirs(folder, exist_ok=True)
        os.chmod(folder, 0o777)
        np.save(new_filename, raw_image_downsampled)


def downsample_raw_image(image: np.ndarray, scale: int = 5):
    # split image into 6x6 blocks
    # downsample each block
    # store in new image
    new_image = np.zeros(
        (image.shape[0] // scale, image.shape[1] // scale), dtype=np.uint16
    )
    kernel_size = 2 * scale
    for ii, i in enumerate(range(0, image.shape[0] - kernel_size, kernel_size)):
        for jj, j in enumerate(range(0, image.shape[1] - kernel_size, kernel_size)):
            kernel = image[i : i + kernel_size, j : j + kernel_size]
            new_image[2 * ii : 2 * (ii + 1), 2 * jj : 2 * (jj + 1)] = _downsample(
                kernel
            )
    return new_image


def main():
    jpg_folder = os.path.join(ROOT, "full", "jpg")
    raw_folder = os.path.join(ROOT, "full", "raw")

    jpg_files = [
        os.path.join(jpg_folder, f)
        for f in os.listdir(jpg_folder)
        if f.endswith(".jpg") and not f.startswith(".")
    ]
    raw_files = [
        os.path.join(raw_folder, f)
        for f in os.listdir(raw_folder)
        if f.endswith(".nef") and not f.startswith(".")
    ]

    process_map(downsample_and_store_jpeg, jpg_files, max_workers=None)
    process_map(downsample_and_store_raw, raw_files, max_workers=None)


if __name__ == "__main__":

    main()
