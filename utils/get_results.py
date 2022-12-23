from collections import defaultdict
import os
import json
import pickle
import numpy as np
import matplotlib.pyplot as plt
import re
from tbparser.summary_reader import SummaryReader

ROOT = "/workspaces/s0001387/raw_od/detectron2_copy/output/nod_learnable_isp_00"
ROOT = "/workspaces/s0001387/raw_od/detectron2_copy/output/learnable_isp_02_zprod"

EXPERIMENTS = [
    "downsampled_jpg_NoOp",
    "raw_plain_NoOp",
    "raw_plain_LearnableGamma",
    "raw_plain_LearnableYeoJohnson",
    "raw_plain_NormalizedLearnableErf",
]

TABLE_ROW = "& {AP:.1f} $\pm$ {AP_STD:.1f} {SPACING} & {AP_50:.1f} $\pm$ {AP_50_STD:.1f} {SPACING} & {AP_75:.1f} $\pm$ {AP_75_STD:.1f} {SPACING} & {AP_CAR:.1f} $\pm$ {AP_CAR_STD:.1f} {SPACING} & {AP_PED:.1f} $\pm$ {AP_PED_STD:.1f} {SPACING} & {AP_BIC:.1f} $\pm$ {AP_BIC_STD:.1f}"


def main_old():
    folders = os.listdir(ROOT)

    metrics_per_folder = {}
    for folder in folders:
        metrics = defaultdict(list)
        path = os.path.join(ROOT, folder, "log.txt")
        # read the log file and parse all lines that look like this
        # [12/09 12:57:43] d2.evaluation.testing INFO: copypaste: AP,AP50,AP75,APs,APm,APl
        # [12/09 12:57:43] d2.evaluation.testing INFO: copypaste: 51.4040,86.1611,55.2984,nan,11.8755,54.0352

        with open(path, "r") as f:
            lines = f.readlines()
            for line in lines:
                if "copypaste" in line:
                    if "bbox" in line:
                        continue
                    if "AP,AP50,AP75,APs,APm,APl" in line:
                        continue

                    # get the AP values
                    aps = [
                        float(f) if f != "nan" else 0
                        for f in line.split("copypaste: ")[1].strip().split(",")
                    ]
                    metric_names = ["AP", "AP50", "AP75", "APs", "APm", "APl"]
                    for metric_name, ap in zip(metric_names, aps):
                        metrics[metric_name].append(ap)
                # or like this
                # | bicycle    | 41.912 | car        | 62.015 | person     | 50.284 |

                if "|" in line:
                    if "bicycle" in line and "." in line:
                        try:
                            aps = [float(f) for f in line.split("|")[2::2]]
                        except:
                            continue
                        metric_names = ["AP_bicycle", "AP_car", "AP_person"]
                        for metric_name, ap in zip(metric_names, aps):
                            metrics[metric_name].append(ap)

        metrics_per_folder[folder] = metrics

    # the metrics are now stored in a dict with the folder name as key, where the digit at the end is the run number
    # we want to plot the mean and std of the APs for each metric
    # so group by name without the run number
    metrics_per_name = defaultdict(list)
    for folder, metrics in metrics_per_folder.items():
        name = re.sub(r"\d+$", "", folder)
        metrics_per_name[name].append(metrics)

    mean_std_per_name = defaultdict(dict)
    for experiment in metrics_per_name:
        for m in [
            "AP",
            "AP50",
            "AP75",
            "APs",
            "APm",
            "APl",
            "AP_bicycle",
            "AP_car",
            "AP_person",
        ]:
            metrics = [d[m] for d in metrics_per_name[experiment]]
            metrics = np.array(metrics)
            mean = np.mean(metrics, axis=0)
            std = np.std(metrics, axis=0)
            mean_std_per_name[experiment][m] = (mean, std)

    # plot the mean and std for AP and AP50 in seperate figures for all experiments
    for m in [
        "AP",
        "AP50",
        "AP75",
        "AP_bicycle",
        "AP_car",
        "AP_person",
    ]:
        plt.figure()
        for experiment in mean_std_per_name:
            mean, std = mean_std_per_name[experiment][m]
            plt.plot(mean, label=experiment)
            plt.fill_between(
                range(len(mean)),
                mean - std,
                mean + std,
                alpha=0.2,
            )
        plt.legend()
        plt.title(m)
        plt.xlabel("epoch")
        plt.ylabel("AP")

    # plt.show()

    # print the mean and std for all metrics for all experiments and format accoring to
    # \begin{tabular}{lcccccc}
    #        \hline
    #        Components             & $AP$                       & $AP_{50}$                  & $AP_{75}$                  & $AP_{car}$                 & $AP_{ped}$                 & $AP_{bic}$    \\
    #        \hline
    #        RGB Baseline           & 0.0 $\pm$ 0.2 {SPACING} & 0.0 $\pm$ 0.2 {SPACING} & 0.0 $\pm$ 0.2 {SPACING} & 0.0 $\pm$ 0.2 {SPACING} & 0.0 $\pm$ 0.2 {SPACING} & 0.0 $\pm$ 0.2 \\
    #        NoOp                   & 0.0 $\pm$ 0.2 {SPACING} & 0.0 $\pm$ 0.2 {SPACING} & 0.0 $\pm$ 0.2 {SPACING} & 0.0 $\pm$ 0.2 {SPACING} & 0.0 $\pm$ 0.2 {SPACING} & 0.0 $\pm$ 0.2 \\
    #        Learnable Decompanding & 0.0 $\pm$ 0.2 {SPACING} & 0.0 $\pm$ 0.2 {SPACING} & 0.0 $\pm$ 0.2 {SPACING} & 0.0 $\pm$ 0.2 {SPACING} & 0.0 $\pm$ 0.2 {SPACING} & 0.0 $\pm$ 0.2 \\
    #        Learnable Tonemapping  & 0.0 $\pm$ 0.2 {SPACING} & 0.0 $\pm$ 0.2 {SPACING} & 0.0 $\pm$ 0.2 {SPACING} & 0.0 $\pm$ 0.2 {SPACING} & 0.0 $\pm$ 0.2 {SPACING} & 0.0 $\pm$ 0.2 \\
    #        \hline
    #    \end{tabular}

    # name_map = {
    #     "jpg_": "RGB Baseline",
    #     "raw_plain_": "NoOp",
    #     "with_decompanding_raw_plain_": "Learnable Decompanding",
    #     "with_tonemap_raw_plain_": "Learnable Tonemapping",
    # }

    name_map = {
        "nod_jpg_NoOp_": "RGB Baseline",
        "nod_raw_plain_NoOp_": "NoOp",
        "nod_raw_plain_LearnableDecompanding_": "Learnable Decompanding",
        "nod_raw_plain_LearnableTonemapping_": "Learnable Tonemapping",
    }

    for experiment in name_map.keys():
        print(name_map[experiment], end="")
        for m in [
            "AP",
            "AP50",
            "AP75",
            "AP_bicycle",
            "AP_car",
            "AP_person",
        ]:
            mean, std = mean_std_per_name[experiment][m]
            mean_ = mean[-1]
            std_ = std[-1]

            print(f" & {mean_:.3f} $\pm$ {std_:.2f}", end="")

            if m != "AP_person":
                print(" {SPACING}", end="")
            else:
                print(r"\\", end="")

        print("")


def get_data():
    metrics_per_experiment = defaultdict(lambda: defaultdict(list))
    for experiment in EXPERIMENTS:
        for i in range(3):
            folder_name = f"{experiment}_{str(i).zfill(2)}"
            reader = SummaryReader(
                os.path.join(ROOT, folder_name),
                tag_filter=(
                    "AP",
                    "AP50",
                    "AP75",
                    "AP-bicycle",
                    "AP-car",
                    "AP-person",
                    "params/lambda",
                    "params/gamma",
                    "params/mu",
                    "params/sigma",
                ),
            )
            for item in reader:
                metrics_per_experiment[folder_name][item.tag].append(item.value)

    # convert to numpy arrays
    for experiment in metrics_per_experiment:
        for tag in metrics_per_experiment[experiment]:
            metrics_per_experiment[experiment][tag] = np.array(
                metrics_per_experiment[experiment][tag]
            )

    metrics_grouped_by_name = defaultdict(lambda: defaultdict(list))
    for experiment in metrics_per_experiment:
        name = "_".join(experiment.split("_")[:-1])
        for key in metrics_per_experiment[experiment]:
            metrics_grouped_by_name[name][key].append(
                metrics_per_experiment[experiment][key]
            )

    # convert to numpy arrays
    for name in metrics_grouped_by_name:
        for tag in metrics_grouped_by_name[name]:
            metrics_grouped_by_name[name][tag] = np.vstack(
                metrics_grouped_by_name[name][tag]
            )

    # compute mean and std
    mean_std_per_name = defaultdict(lambda: defaultdict(list))
    for name in metrics_grouped_by_name:
        for tag in metrics_grouped_by_name[name]:
            mean_std_per_name[name][tag] = (
                np.mean(metrics_grouped_by_name[name][tag], axis=0),
                np.std(metrics_grouped_by_name[name][tag], axis=0),
            )

    return metrics_grouped_by_name, mean_std_per_name


def main():

    if not os.path.exists("metrics_grouped_by_name.pkl"):
        metrics_grouped_by_name, mean_std_per_name = get_data()
        # convert from defaultdict to dict
        for name in metrics_grouped_by_name:
            metrics_grouped_by_name[name] = dict(metrics_grouped_by_name[name])
        for name in mean_std_per_name:
            mean_std_per_name[name] = dict(mean_std_per_name[name])

        metrics_grouped_by_name = dict(metrics_grouped_by_name)
        mean_std_per_name = dict(mean_std_per_name)

    else:
        with open("metrics_grouped_by_name.pkl", "rb") as f:
            metrics_grouped_by_name = pickle.load(f)
        with open("mean_std_per_name.pkl", "rb") as f:
            mean_std_per_name = pickle.load(f)

    # pickle these objects
    with open("metrics_grouped_by_name.pkl", "wb") as f:
        pickle.dump(metrics_grouped_by_name, f)
    with open("mean_std_per_name.pkl", "wb") as f:
        pickle.dump(mean_std_per_name, f)

    # plot the values of lambda for LearnableYeoJohnson (choose the first one)
    fig, ax = plt.subplots(figsize=(10, 5))

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
    _plot(
        metrics_grouped_by_name["raw_plain_LearnableGamma"]["params/gamma"][0],
        r"$\gamma$",
        1.0,
    )
    _plot(
        metrics_grouped_by_name["raw_plain_NormalizedLearnableErf"]["params/mu"][0],
        r"$\mu$",
        1.0,
    )
    _plot(
        metrics_grouped_by_name["raw_plain_NormalizedLearnableErf"]["params/sigma"][0],
        r"$\sigma$",
        1.0,
    )
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Parameter value")
    ax.legend()
    # show the grid
    ax.grid()

    # set the font size of the x and y labels
    ax.xaxis.label.set_fontsize(18)
    ax.yaxis.label.set_fontsize(18)

    # set the font size of the tick labels
    ax.tick_params(axis="both", which="major", labelsize=15)

    fig.tight_layout()
    # save the figure as a pdf
    fig.savefig("/home/s0001387/Documents/phd/projects/figures/params.pdf")

    # print the mean and std for AP, AP50, AP75, AP-bicycle, AP-car, AP-person for each experiment
    for experiment in mean_std_per_name:
        print(experiment)
        s = TABLE_ROW.format(
            AP=mean_std_per_name[experiment]["AP"][0][-1],
            AP_STD=mean_std_per_name[experiment]["AP"][1][-1],
            AP_50=mean_std_per_name[experiment]["AP50"][0][-1],
            AP_50_STD=mean_std_per_name[experiment]["AP50"][1][-1],
            AP_75=mean_std_per_name[experiment]["AP75"][0][-1],
            AP_75_STD=mean_std_per_name[experiment]["AP75"][1][-1],
            AP_BIC=mean_std_per_name[experiment]["AP-bicycle"][0][-1],
            AP_BIC_STD=mean_std_per_name[experiment]["AP-bicycle"][1][-1],
            AP_CAR=mean_std_per_name[experiment]["AP-car"][0][-1],
            AP_CAR_STD=mean_std_per_name[experiment]["AP-car"][1][-1],
            AP_PED=mean_std_per_name[experiment]["AP-person"][0][-1],
            AP_PED_STD=mean_std_per_name[experiment]["AP-person"][1][-1],
            SPACING=r"\hspace{5pt}",
        )
        print(s)


if __name__ == "__main__":
    main()
