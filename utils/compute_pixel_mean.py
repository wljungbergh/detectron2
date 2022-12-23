"""Script to compute the mean and std dev pixel value of a dataset."""
import os
from PIL import Image
import numpy as np
import tqdm
import matplotlib.pyplot as plt

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


def raw_per_channel():
    r_sum = g1_sum = g2_sum = b_sum = 0
    # create a histogram bins for each channel
    r_bins = np.zeros(4095)
    g1_bins = np.zeros(4095)
    g2_bins = np.zeros(4095)
    b_bins = np.zeros(4095)

    with tqdm.tqdm(total=N_PIXELS_PER_IMAGE) as pbar:
        for root, _, files in os.walk(RAW_ROOT):
            for file in files:
                raw_path = os.path.join(root, file)
                image = np.load(raw_path)
                # split based on rgbg pattern
                r = image[0::2, 0::2]
                b = image[1::2, 1::2]
                g1 = image[0::2, 1::2]
                g2 = image[1::2, 0::2]
                r_sum += np.sum(r)
                g1_sum += np.sum(g1)
                g2_sum += np.sum(g2)
                b_sum += np.sum(b)

                # update the histogram bins
                r_bins += np.histogram(r, bins=4095, range=(0, 4095))[0]
                g1_bins += np.histogram(g1, bins=4095, range=(0, 4095))[0]
                g2_bins += np.histogram(g2, bins=4095, range=(0, 4095))[0]
                b_bins += np.histogram(b, bins=4095, range=(0, 4095))[0]

                pbar.update(1)

    r_mean = r_sum / (N_PIXELS_PER_IMAGE / 4 * N_IMAGES)
    g1_mean = g1_sum / (N_PIXELS_PER_IMAGE / 4 * N_IMAGES)
    g2_mean = g2_sum / (N_PIXELS_PER_IMAGE / 4 * N_IMAGES)
    b_mean = b_sum / (N_PIXELS_PER_IMAGE / 4 * N_IMAGES)

    r_sum = g1_sum = g2_sum = b_sum = 0
    with tqdm.tqdm(total=N_PIXELS_PER_IMAGE) as pbar:
        for root, _, files in os.walk(RAW_ROOT):
            for file in files:
                raw_path = os.path.join(root, file)
                image = np.load(raw_path)
                # split based on rgbg pattern
                r = image[0::2, 0::2]
                b = image[1::2, 1::2]
                g1 = image[0::2, 1::2]
                g2 = image[1::2, 0::2]
                r_sum += np.sum((r - r_mean) ** 2)
                g1_sum += np.sum((g1 - g1_mean) ** 2)
                g2_sum += np.sum((g2 - g2_mean) ** 2)
                b_sum += np.sum((b - b_mean) ** 2)
                pbar.update(1)

    r_std = np.sqrt(r_sum / (N_PIXELS_PER_IMAGE / 4 * N_IMAGES))
    g1_std = np.sqrt(g1_sum / (N_PIXELS_PER_IMAGE / 4 * N_IMAGES))
    g2_std = np.sqrt(g2_sum / (N_PIXELS_PER_IMAGE / 4 * N_IMAGES))
    b_std = np.sqrt(b_sum / (N_PIXELS_PER_IMAGE / 4 * N_IMAGES))

    print("r_mean: ", r_mean)
    print("g1_mean: ", g1_mean)
    print("g2_mean: ", g2_mean)
    print("b_mean: ", b_mean)
    print("r_std: ", r_std)
    print("g1_std: ", g1_std)
    print("g2_std: ", g2_std)
    print("b_std: ", b_std)


def hist_per_channel():

    # create a histogram bins for each channel
    raw_r_bins = np.zeros(4095)
    raw_g_bins = np.zeros(4095)
    raw_b_bins = np.zeros(4095)

    jpg_r_bins = np.zeros(255)
    jpg_g_bins = np.zeros(255)
    jpg_b_bins = np.zeros(255)

    with tqdm.tqdm(total=N_IMAGES) as pbar:
        for root, _, files in os.walk(RAW_ROOT):
            for file in files:
                raw_path = os.path.join(root, file)
                jpg_path = os.path.join(JPG_ROOT, file.replace(".npy", ".jpg"))

                image = np.load(raw_path)
                # split based on rgbg pattern
                r = image[0::2, 0::2]
                b = image[1::2, 1::2]
                g1 = image[0::2, 1::2]
                g2 = image[1::2, 0::2]

                # update the histogram bins
                raw_r_bins += np.histogram(r, bins=4095, range=(0, 4095))[0]
                raw_g_bins += np.histogram(g1, bins=4095, range=(0, 4095))[0]
                raw_g_bins += np.histogram(g2, bins=4095, range=(0, 4095))[0]
                raw_b_bins += np.histogram(b, bins=4095, range=(0, 4095))[0]

                image = Image.open(jpg_path)
                image = np.array(image)

                jpg_r_bins += np.histogram(image[:, :, 0], bins=255, range=(0, 255))[0]
                jpg_g_bins += np.histogram(image[:, :, 1], bins=255, range=(0, 255))[0]
                jpg_b_bins += np.histogram(image[:, :, 2], bins=255, range=(0, 255))[0]

                pbar.update(1)

    # save the histogram bins
    np.save("raw_r_bins.npy", raw_r_bins)
    np.save("raw_g_bins.npy", raw_g_bins)
    np.save("raw_b_bins.npy", raw_b_bins)

    np.save("jpg_r_bins.npy", jpg_r_bins)
    np.save("jpg_g_bins.npy", jpg_g_bins)
    np.save("jpg_b_bins.npy", jpg_b_bins)

    # plot the histogram bins
    plt.figure()
    plt.plot(raw_r_bins)
    plt.title("raw_r_bins")
    plt.savefig("raw_r_bins.png")

    plt.figure()
    plt.plot(raw_g_bins)
    plt.title("raw_g_bins")
    plt.savefig("raw_g_bins.png")

    plt.figure()
    plt.plot(raw_b_bins)
    plt.title("raw_b_bins")
    plt.savefig("raw_b_bins.png")

    plt.figure()
    plt.plot(jpg_r_bins)
    plt.title("jpg_r_bins")
    plt.savefig("jpg_r_bins.png")

    plt.figure()

    plt.plot(jpg_g_bins)
    plt.title("jpg_g_bins")
    plt.savefig("jpg_g_bins.png")

    plt.figure()
    plt.plot(jpg_b_bins)
    plt.title("jpg_b_bins")
    plt.savefig("jpg_b_bins.png")


if __name__ == "__main__":
    hist_per_channel()
