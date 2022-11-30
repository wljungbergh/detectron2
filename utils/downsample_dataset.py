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

    # if os.path.exists(new_filename):
    #    return

    with rawpy.imread(image_path) as raw:
        # make sure that it has the same size as jpg without removing any partial bayer patterns
        raw_image = raw.raw_image_visible

        raw_image_downsampled = downsample_raw_image(raw_image, scale=SCALE)

        # store as numpy array

        os.makedirs(folder, exist_ok=True)
        os.chmod(folder, 0o777)
        np.save(new_filename, raw_image_downsampled)


def postproccess_raw_to_jpg(image_path: str):
    downsample_folder = os.path.dirname(image_path).replace("full", "fifth")
    downsample_filename = os.path.basename(image_path).replace(".nef", ".npy")
    downsample_filename = os.path.join(downsample_folder, downsample_filename)
    downsampled_raw = np.load(downsample_filename)

    new_folder = (
        os.path.dirname(image_path)
        .replace("full", "fifth")
        .replace("fifth/raw", "fifth/jpg_downsampled")
    )
    new_filename = os.path.basename(image_path).replace(".nef", ".jpg")
    new_filename = os.path.join(new_folder, new_filename)

    if os.path.exists(new_filename):
        return

    with rawpy.imread(image_path) as raw:
        # make sure that it has the same size as jpg without removing any partial bayer patterns
        for h in range(DOWNSAMPLED_HEIGHT):
            for w in range(DOWNSAMPLED_WIDTH):
                raw.raw_image_visible[h, w] = downsampled_raw[h, w]

        rgb = raw.postprocess()[:DOWNSAMPLED_HEIGHT, :DOWNSAMPLED_WIDTH, :]

    os.makedirs(new_folder, exist_ok=True)
    os.chmod(new_folder, 0o777)

    Image.fromarray(rgb).save(new_filename)


def downsample_raw_image(image: np.ndarray, scale: int = 5):
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


def get_linear_raw(image_path: str):

    folder = (
        os.path.dirname(image_path)
        .replace("full", "fifth")
        .replace("fifth/raw", "fifth/linear_raw")
    )
    filename = os.path.basename(image_path).replace(".nef", ".npy")
    new_filename = os.path.join(folder, filename)

    if os.path.exists(new_filename):
        return

    os.makedirs(folder, exist_ok=True)
    os.chmod(folder, 0o777)

    with rawpy.imread(image_path) as raw:
        # make sure that it has the same size as jpg without removing any partial bayer patterns
        rgb = raw.postprocess(gamma=(1, 1), no_auto_bright=True, output_bps=16)
        new_image = np.zeros((rgb.shape[0], rgb.shape[1]), dtype=np.uint16)
        new_image[::2, ::2] = rgb[::2, ::2, 0]  # red
        new_image[1::2, 1::2] = rgb[1::2, 1::2, 2]  # blue
        new_image[::2, 1::2] = rgb[::2, 1::2, 1]  # green
        new_image[1::2, ::2] = rgb[1::2, ::2, 1]  # green

        downsampled_linear = downsample_raw_image(new_image, scale=SCALE)
        np.save(new_filename, downsampled_linear)


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
    process_map(postproccess_raw_to_jpg, raw_files, max_workers=None)
    process_map(get_linear_raw, raw_files, max_workers=None)


if __name__ == "__main__":

    main()
