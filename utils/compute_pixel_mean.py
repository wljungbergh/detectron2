"""Script to compute the mean and std dev pixel value of a dataset."""
import os
from PIL import Image
import numpy as np
import tqdm

ROOT = "/workspaces/s0001387/raw_od/pascal_raw/pascal_raw/fifth"
JPG_ROOT = os.path.join(ROOT, "jpg")
RAW_ROOT = os.path.join(ROOT, "raw")
GREY_ROOT = os.path.join(ROOT, "grey")
N_PIXELS_PER_IMAGE = 1206 * 802
N_IMAGES = 4259


def main():
    r_channel_sum = g_channel_sum = b_channel_sum = 0

    raw_channel_sum = 0
    grey_channel_sum = 0
    with tqdm.tqdm(total=N_IMAGES * 3 * 2) as pbar:
        for root, _, files in os.walk(JPG_ROOT):
            for file in files:
                image_path = os.path.join(root, file)
                image = Image.open(image_path)
                r, g, b = image.split()
                r_channel_sum += np.sum(np.array(r))
                g_channel_sum += np.sum(np.array(g))
                b_channel_sum += np.sum(np.array(b))
                pbar.update(1)

        for root, _, files in os.walk(RAW_ROOT):
            for file in files:
                raw_path = os.path.join(root, file)
                image = np.load(raw_path)
                raw_channel_sum += np.sum(np.array(image))
                pbar.update(1)

        for root, _, files in os.walk(GREY_ROOT):
            for file in files:
                grey_path = os.path.join(root, file)
                image = Image.open(grey_path)
                grey_channel_sum += np.sum(np.array(image))
                pbar.update(1)

        # compute the means
        r_mean = r_channel_sum / (N_PIXELS_PER_IMAGE * N_IMAGES)
        g_mean = g_channel_sum / (N_PIXELS_PER_IMAGE * N_IMAGES)
        b_mean = b_channel_sum / (N_PIXELS_PER_IMAGE * N_IMAGES)
        raw_mean = raw_channel_sum / (N_PIXELS_PER_IMAGE * N_IMAGES)
        grey_mean = grey_channel_sum / (N_PIXELS_PER_IMAGE * N_IMAGES)

        # compute the standard deviations
        r_channel_sum = g_channel_sum = b_channel_sum = 0
        raw_channel_sum = 0
        grey_channel_sum = 0
        for root, _, files in os.walk(JPG_ROOT):
            for file in files:
                image_path = os.path.join(root, file)
                image = Image.open(image_path)
                r, g, b = image.split()
                r_channel_sum += np.sum((np.array(r) - r_mean) ** 2)
                g_channel_sum += np.sum((np.array(g) - g_mean) ** 2)
                b_channel_sum += np.sum((np.array(b) - b_mean) ** 2)
                pbar.update(1)

        for root, _, files in os.walk(RAW_ROOT):
            for file in files:
                raw_path = os.path.join(root, file)
                image = np.load(raw_path)
                raw_channel_sum += np.sum((np.array(image) - raw_mean) ** 2)
                pbar.update(1)

        for root, _, files in os.walk(GREY_ROOT):
            for file in files:
                grey_path = os.path.join(root, file)
                image = Image.open(grey_path)
                grey_channel_sum += np.sum((np.array(image) - grey_mean) ** 2)
                pbar.update(1)

    r_std = np.sqrt(r_channel_sum / (N_PIXELS_PER_IMAGE * N_IMAGES))
    g_std = np.sqrt(g_channel_sum / (N_PIXELS_PER_IMAGE * N_IMAGES))
    b_std = np.sqrt(b_channel_sum / (N_PIXELS_PER_IMAGE * N_IMAGES))
    raw_std = np.sqrt(raw_channel_sum / (N_PIXELS_PER_IMAGE * N_IMAGES))
    grey_std = np.sqrt(grey_channel_sum / (N_PIXELS_PER_IMAGE * N_IMAGES))

    print("r_mean: ", r_mean)
    print("g_mean: ", g_mean)
    print("b_mean: ", b_mean)
    print("raw_mean: ", raw_mean)
    print("grey_mean: ", grey_mean)
    print("r_std: ", r_std)
    print("g_std: ", g_std)
    print("b_std: ", b_std)
    print("raw_std: ", raw_std)
    print("grey_std: ", grey_std)


if __name__ == "__main__":
    main()
