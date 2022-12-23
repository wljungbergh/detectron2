import time
import subprocess

SLURM_SCRIPT = r"""#!/bin/bash
#SBATCH --nodes 1
#SBATCH --gres=gpu:1
#SBATCH -c 32
#SBATCH --mem=100G
#SBATCH --time 3-00:00:00
#SBATCH --output /workspaces/%u/slurm/{name}_%j.out
#SBATCH -p zprod
#SBATCH --no-requeue

singularity exec --nv\
  -B /workspaces:/workspaces\
  -B /workspaces/s0001387/raw_od/detectron2_copy:/home/appuser/detectron2_repo\
  -B /workspaces/s0001387/raw_od/pascal_raw/pascal_raw:/datasets/pascal_raw/\
  -B /workspaces/s0001387/raw_od/nod_raw:/datasets/nod_raw/\
  --pwd /home/appuser/detectron2_repo\
  --env PYTHONPATH=/home/appuser/detectron2_repo\
  /workspaces/s0001387/raw_od/detectron2.sif\
  python3 projects/RawOD/train.py\
  --config-file projects/RawOD/configs/faster_rcnn_R_50_FPN_1x.yaml\
  --experiment_name {name}\
  --format {format}\
  {flags}\
  {nod_dataset}
"""


N_EXPERIMENTS_PER_FORMAT = 1
FORMATS = [
    "raw_plain",
    # "raw_rgb",
    # "raw_offset",
    # "raw_reshape",
    "downsampled_jpg",
    # "linear_raw_plain",
    # "linear_raw_rgb",
    # "linear_raw_offset",
    # "linear_raw_reshape",
]

COMBINATIONS = {
    "NoOpWithMaxNorm": "--normalize_with_max",
    "LearnableYeoJohnsonWithMaxNorm": "--use_yeojohnson --normalize_with_max",
    "NormalizedLearnableYeoJohnssonWithMaxNorm": "--use_yeojohnson --use_normalize --normalize_with_max",
    "NoOp": "",
    "LearnableYeoJohnson": "--use_yeojohnson",
    "LearnableGamma": "--use_gamma",
    "LearnableErf": "--use_erf",
    "NormalizedLearnableYeoJohnssonWith": "--use_yeojohnson --use_normalize",
    "NormalizedLearnableGamma": "--use_gamma --use_normalize",
    "NormalizedLearnableErf": "--use_erf --use_normalize",
}


def main():
    nod_dataset = False
    for format in FORMATS:
        if format == "raw_plain":
            combos = COMBINATIONS
        elif format == "downsampled_jpg":
            combos = {"NoOp": ""}
        for i in range(N_EXPERIMENTS_PER_FORMAT):
            for combo_name, flags in combos.items():
                name = "nod_" if nod_dataset else ""
                name += f"{format}_{combo_name}_{str(i).zfill(2)}"

                script = SLURM_SCRIPT.format(
                    name=name,
                    format=format,
                    nod_dataset="--nod_dataset\\" if nod_dataset else "",
                    flags=flags,
                )
                with open(f"projects/RawOD/slurm/{name}.sh", "w") as f:
                    f.write(script)
                subprocess.run(["sbatch", f"projects/RawOD/slurm/{name}.sh"])
                time.sleep(0.5)


if __name__ == "__main__":
    main()
