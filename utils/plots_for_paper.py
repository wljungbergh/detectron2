import os
import json
import pickle
import torch

from pycocotools import cocoeval, coco
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from matplotlib.lines import Line2D


DROOT = "/workspaces/s0001387/raw_od/detectron2/output/learnable_isp_00/"
JPG_00_RESULTS_PATH = os.path.join(DROOT, "jpg_00/instances_predictions.pth")
JPG_00_COCO_RESULTS_PATH = os.path.join(DROOT, "jpg_00/coco_instances_results.json")

JPG_01_RESULTS_PATH = os.path.join(DROOT, "jpg_01/instances_predictions.pth")
JPG_02_RESULTS_PATH = os.path.join(DROOT, "jpg_02/instances_predictions.pth")

RAW_PLAIN_00_RESULTS_PATH = os.path.join(
    DROOT, "raw_plain_00/instances_predictions.pth"
)
RAW_PLAIN_00_COCO_RESULTS_PATH = os.path.join(
    DROOT, "raw_plain_00/coco_instances_results.json"
)

RAW_PLAIN_01_RESULTS_PATH = os.path.join(
    DROOT, "raw_plain_01/instances_predictions.pth"
)
RAW_PLAIN_02_RESULTS_PATH = os.path.join(
    DROOT, "raw_plain_02/instances_predictions.pth"
)

IMAGE_ROOT = "/workspaces/s0001387/raw_od/pascal_raw/pascal_raw/fifth/downsampled_jpg"
RAW_ROOT = "/workspaces/s0001387/raw_od/pascal_raw/pascal_raw/fifth/raw/"
VAL_ANNOTATION_PATH = (
    "/workspaces/s0001387/raw_od/pascal_raw/pascal_raw/trainval/fifth_val_jpg.json"
)

ID_TO_CLASS = {
    1: "bicyle",
    2: "car",
    3: "person",
}

CLASS_TO_COLOR = {
    "bicyle": "green",
    "car": "blue",
    "person": "red",
}

SCORE_THRESHOLD = 0.75


def load_results(path):
    """Load the results from a json file."""
    with open(path, "rb") as f:
        results = torch.load(f)
    return results


def load_all_results():
    """Load all the results."""
    jpg_00_results = load_results(JPG_00_RESULTS_PATH)
    jpg_01_results = load_results(JPG_01_RESULTS_PATH)
    jpg_02_results = load_results(JPG_02_RESULTS_PATH)

    raw_plain_00_results = load_results(RAW_PLAIN_00_RESULTS_PATH)
    raw_plain_01_results = load_results(RAW_PLAIN_01_RESULTS_PATH)
    raw_plain_02_results = load_results(RAW_PLAIN_02_RESULTS_PATH)

    return (
        jpg_00_results,
        jpg_01_results,
        jpg_02_results,
        raw_plain_00_results,
        raw_plain_01_results,
        raw_plain_02_results,
    )


def plot(raw_plain_00_det, jpg_00_det, cocoGt):
    """Plot the results."""

    NICE_IMG_IDS = (
        "2014_000111",
        "2014_000155",
        "2014_000219",
        "2014_000295",
        "2014_000325",
        "2014_000369",
        "2014_000379",
        "2014_000397",
        "2014_000421",
        "2014_000441",
        "2014_000533",
        "2014_000681",
        "2014_000711",
    )
    for img_id in cocoGt.getImgIds():
        if img_id not in NICE_IMG_IDS:
            continue

        raw_det = raw_plain_00_det.imgToAnns[img_id]
        jpg_det = jpg_00_det.imgToAnns[img_id]
        ann = cocoGt.imgToAnns[img_id]

        fig, ax = plt.subplots(3, 1, figsize=(6, 12))
        img = plt.imread(os.path.join(IMAGE_ROOT, cocoGt.imgs[img_id]["file_name"]))
        for i in range(3):
            ax[i].imshow(img)

        # add the annotations
        for i, items in enumerate([ann, jpg_det, raw_det]):
            for a in items:
                if "score" in a and a["score"] < SCORE_THRESHOLD:
                    continue
                bbox = a["bbox"]
                rect = plt.Rectangle(
                    (bbox[0], bbox[1]),
                    bbox[2],
                    bbox[3],
                    fill=False,
                    edgecolor=CLASS_TO_COLOR[ID_TO_CLASS[a["category_id"]]],
                    linewidth=5,
                )
                ax[i].add_patch(rect)

        ax[0].set_title(f"Ground Truth")
        ax[1].set_title(f"RGB Baseline")
        ax[2].set_title(f"Learnable Yeo-Johnson")
        # increase the font size of the titles
        for i in range(3):
            ax[i].title.set_fontsize(25)

        # create legend
        legend_elements = [
            Line2D(
                [0],
                [0],
                color="g",
                label="bicyle",
            ),
            Line2D(
                [0],
                [0],
                color="b",
                label="car",
            ),
            Line2D(
                [0],
                [0],
                color="r",
                label="person",
            ),
        ]

        locs_map = {
            "2014_000369": "upper right",
            "2014_000441": "upper left",
            "2014_000681": "upper right",
        }
        if img_id == "2014_000681":
            ax[0].legend(
                handles=legend_elements,
                loc=locs_map.get(img_id, "lower right"),
                fontsize=25,
            )
        for i in range(3):
            # remove the axis
            ax[i].axis("off")

        # plt.show()
        fig.tight_layout()
        # plt.show()
        plt.savefig(
            f"/home/s0001387/Documents/phd/projects/figures/results_{img_id}.png"
        )
        plt.close()


def plot_parameter_evolutions():
    name = "raw_plain_00"
    os.path.join(DROOT, name)
    iterations = [5000 * i + 4999 for i in range(30)]
    filenames = [f"model_{str(i).zfill(7)}.pth" for i in iterations]
    paths = [os.path.join(DROOT, name, f) for f in filenames]
    if not os.path.exists(f"{name}_lambdas.npy"):
        lambdas = [0.35]
        gammas = [1.0]

        for p in tqdm(paths):
            weights = torch.load(p)
            l = weights["model"]["backbone.bottom_up.stem.learnable_isp.lambda_param"]
            g = weights["model"]["backbone.bottom_up.stem.learnable_isp.gamma_param"]
            lambdas.append(l.detach().cpu().numpy())
            gammas.append(g.detach().cpu().numpy())
            np.save(f"{name}_lambdas.npy", lambdas)
            np.save(f"{name}_gammas.npy", gammas)
    else:
        lambdas = np.load(f"{name}_lambdas.npy")
        gammas = np.load(f"{name}_gammas.npy")

    fig, ax = plt.subplots(2, 1, figsize=(12, 12))
    ax[0].plot([0] + iterations, lambdas)
    ax[0].set_title("Lambda evolution")
    # set the axis labels
    ax[0].set_xlabel("Iterations")
    ax[0].set_ylabel("Lambda")
    ax[1].plot([0] + iterations, gammas)
    ax[1].set_title("Gamma evolution")
    # set the axis labels
    ax[1].set_xlabel("Iterations")
    ax[1].set_ylabel("Gamma")
    plt.show()


def main():
    """Load the COCO files."""
    cocoGt = coco.COCO(VAL_ANNOTATION_PATH)
    jpg_00_det = cocoGt.loadRes(JPG_00_COCO_RESULTS_PATH)
    raw_plain_00_det = cocoGt.loadRes(RAW_PLAIN_00_COCO_RESULTS_PATH)

    jpg_coco_eval = cocoeval.COCOeval(cocoGt, jpg_00_det, "bbox")
    jpg_coco_eval.evaluate()
    jpg_coco_eval.accumulate()

    raw_plain_coco_eval = cocoeval.COCOeval(cocoGt, raw_plain_00_det, "bbox")
    raw_plain_coco_eval.evaluate()
    raw_plain_coco_eval.accumulate()

    plot(jpg_00_det=jpg_00_det, raw_plain_00_det=raw_plain_00_det, cocoGt=cocoGt)
    # plot_parameter_evolutions()


if __name__ == "__main__":
    # main()

    # plot the functional form of the Yeo-Johnson transformation
    # x = np.linspace(0, 4095, 1000)
    # y_init = (np.power((x + 1), 0.35) - 1) / 0.35
    # y_final = (np.power((x + 1), 0.1) - 1) / 1.0

    # fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    # ax.plot(
    #     x,
    #     y_init,
    #     label=r"Initial  $\lambda = 0.35$",
    #     linestyle="--",
    #     linewidth=2.0,
    #     color="blue",
    # )
    # ax.plot(x, y_final, label=r"Final $\lambda = 0.11$", linewidth=2.0, color="blue")

    # ax.grid()
    # ax.set_xlabel("Pixel value", fontsize=25)
    # ax.set_ylabel("Output activation value", fontsize=25)
    # ax.tick_params(axis="both", which="major", labelsize=20)
    # ax.legend(fontsize=25)

    # fig.tight_layout()
    # plt.savefig("/home/s0001387/Documents/phd/projects/figures/yeo_johnson.pdf")
    with open("metrics_grouped_by_name.pkl", "rb") as f:
        metrics_grouped_by_name = pickle.load(f)

    # plot the values of lambda for LearnableYeoJohnson (choose the first one)
    fig, ax = plt.subplots(figsize=(10, 10))

    def _plot(data, label, init_value):
        data = np.concatenate([[init_value], data])
        x_axis_data = np.arange(0, 150_000 + 20, 20)
        ax.plot(x_axis_data, data, label=label, linewidth=2.0)

    # ax.set_xlabel("Iteration")
    # ax.set_ylabel(r"$\lambda$")
    # ax.legend()
    # set
    # plt.show()
    _plot(
        metrics_grouped_by_name["raw_plain_LearnableYeoJohnson"]["params/lambda"][0],
        r"$\lambda$",
        0.35,
    )
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Parameter value")
    ax.legend(fontsize=25)

    # show the grid
    ax.grid()

    # set the font size of the x and y labels
    ax.xaxis.label.set_fontsize(25)
    ax.yaxis.label.set_fontsize(25)

    # set the font size of the tick labels
    ax.tick_params(axis="both", which="major", labelsize=20)
    # set the ticks be every 30k
    ax.set_xticks(np.arange(0, 150_000 + 30_000, 30_000))

    fig.tight_layout()
    # save the figure as a pdf
    # plt.show()
    fig.savefig("/home/s0001387/Documents/phd/projects/figures/params_only_yeo.pdf")
    print(
        metrics_grouped_by_name["raw_plain_LearnableYeoJohnson"]["params/lambda"][0][-1]
    )
