import os
from PIL import Image

import numpy as np
from tqdm.contrib.concurrent import process_map

SCALE = 3
DOWNSAMPLED_WIDTH = 1282  # 3848 // 3
DOWNSAMPLED_HEIGHT = 722  # 2168 // 3

ZOD_RAW_ROOT = "/workspaces/s0001387/raw_od/zod_raw"


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


def process_frame(frame_id: str):

    orignal_raw_path = os.path.join(ZOD_RAW_ROOT, "full", "raw", frame_id + ".npy")
    output_raw_path = os.path.join(ZOD_RAW_ROOT, "thrid", "raw", frame_id + ".npy")

    original_jpg_folder = f"/staging/dataset_donation/round_2/single_frames/{frame_id}/camera_front_original"
    fn = [f for f in os.listdir(original_jpg_folder) if f.endswith(".png")][0]
    original_jpg_path = os.path.join(original_jpg_folder, fn)
    output_jpg_path = os.path.join(ZOD_RAW_ROOT, "thrid", "jpg", frame_id + ".jpg")

    if os.path.exists(output_raw_path) and os.path.exists(output_jpg_path):
        return

    original_raw = np.load(orignal_raw_path)
    downsampled_raw = downsample_raw_image(original_raw, scale=3)

    np.save(output_raw_path, downsampled_raw)

    original_jpg = Image.open(original_jpg_path)
    downsampled_jpg = original_jpg.resize(
        (downsampled_raw.shape[1], downsampled_raw.shape[0])
    )
    downsampled_jpg.save(output_jpg_path)


def main():

    frame_ids = [str(i).zfill(6) for i in range(0, 100_000)]
    process_map(process_frame, frame_ids, chunksize=25)


if __name__ == "__main__":
    main()
