import os
from PIL import Image

import rawpy
import numpy as np
from tqdm.contrib.concurrent import process_map
from tqdm import tqdm

DOWNSAMPLED_WIDTH = 1282  # 3848 // 3
DOWNSAMPLED_HEIGHT = 722  # 2168 // 3

NIKON_HEIGHT = 880  # 2640 // 3 = 880
NIKON_WIDTH = 1322  # 3968 // 3 = 1322
NIKON_SCALING = 3
NIKON_EXT = ".NEF"

SONY_HEIGHT = 734  # 3672 // 5 = 734
SONY_WIDTH = 1099  # 5496 // 5 = 1099
SONY_SCALING = 5
SONY_EXT = ".ARW"

NOD_RAW_ROOT = "/workspaces/s0001387/raw_od/nod_raw"


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


def process_frame(
    input_raw_path: str,
    input_jpg_path: str,
    output_raw_path: str,
    output_jpg_path: str,
    scale: int,
):

    if os.path.exists(output_raw_path) and os.path.exists(output_jpg_path):
        return

    if scale == SONY_SCALING:
        H = SONY_HEIGHT * 5
        W = SONY_WIDTH * 5
    elif scale == NIKON_SCALING:
        H = NIKON_HEIGHT * 3
        W = NIKON_WIDTH * 3

    with rawpy.imread(input_raw_path) as raw:
        raw_image = raw.raw_image_visible[:H, :W].copy()

    downsampled_raw = downsample_raw_image(raw_image, scale=scale)

    np.save(output_raw_path, downsampled_raw)

    original_jpg = Image.open(input_jpg_path)
    downsampled_jpg = original_jpg.resize(
        (downsampled_raw.shape[1], downsampled_raw.shape[0])
    )
    downsampled_jpg.save(output_jpg_path)


def main():

    nikon_jpg_folder = os.path.join(NOD_RAW_ROOT, "nikon", "full", "jpg")
    sony_jpg_folder = os.path.join(NOD_RAW_ROOT, "sony", "full", "jpg")

    nikon_jpg_files = [
        os.path.join(nikon_jpg_folder, f) for f in os.listdir(nikon_jpg_folder)
    ]
    nikon_raw_files = [
        f.replace("/jpg/", "/raw/").replace(".JPG", ".NEF") for f in nikon_jpg_files
    ]
    sony_jpg_files = [
        os.path.join(sony_jpg_folder, f) for f in os.listdir(sony_jpg_folder)
    ]
    sony_raw_files = [
        f.replace("/jpg/", "/raw/").replace(".JPG", ".ARW") for f in sony_jpg_files
    ]

    nikon_output_raw_folder = os.path.join(NOD_RAW_ROOT, "nikon", "third", "raw")
    nikon_output_jpg_folder = os.path.join(NOD_RAW_ROOT, "nikon", "third", "jpg")
    sony_output_raw_folder = os.path.join(NOD_RAW_ROOT, "sony", "fifth", "raw")
    sony_output_jpg_folder = os.path.join(NOD_RAW_ROOT, "sony", "fifth", "jpg")

    os.makedirs(nikon_output_raw_folder, exist_ok=True)
    os.makedirs(nikon_output_jpg_folder, exist_ok=True)
    os.makedirs(sony_output_raw_folder, exist_ok=True)
    os.makedirs(sony_output_jpg_folder, exist_ok=True)

    nikon_output_raw_files = [
        os.path.join(
            nikon_output_raw_folder, os.path.basename(f).replace(NIKON_EXT, ".npy")
        )
        for f in nikon_raw_files
    ]
    nikon_output_jpg_files = [
        os.path.join(
            nikon_output_jpg_folder, os.path.basename(f).replace(".JPG", ".jpg")
        )
        for f in nikon_jpg_files
    ]
    sony_output_raw_files = [
        os.path.join(
            sony_output_raw_folder, os.path.basename(f).replace(SONY_EXT, ".npy")
        )
        for f in sony_raw_files
    ]
    sony_output_jpg_files = [
        os.path.join(
            sony_output_jpg_folder, os.path.basename(f).replace(".JPG", ".jpg")
        )
        for f in sony_jpg_files
    ]

    input_raw_paths = nikon_raw_files + sony_raw_files
    input_jpg_paths = nikon_jpg_files + sony_jpg_files
    output_raw_paths = nikon_output_raw_files + sony_output_raw_files
    output_jpg_paths = nikon_output_jpg_files + sony_output_jpg_files
    scales = [NIKON_SCALING] * len(nikon_raw_files) + [SONY_SCALING] * len(
        sony_raw_files
    )

    process_map(
        process_frame,
        input_raw_paths[:5],
        input_jpg_paths[:5],
        output_raw_paths[:5],
        output_jpg_paths[:5],
        scales[:5],
        chunksize=1,
        max_workers=1,
    )

    for irp, ijp, orp, ojp, s in tqdm(
        zip(
            input_raw_paths, input_jpg_paths, output_raw_paths, output_jpg_paths, scales
        )
    ):
        process_frame(irp, ijp, orp, ojp, s)


if __name__ == "__main__":
    main()
