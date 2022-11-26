import time
import subprocess

SLURM_SCRIPT = r"""#!/bin/bash
#SBATCH --nodes 1
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --time 1-00:00:00
#SBATCH --output /workspaces/%u/slurm/{name}_%j.out
#SBATCH -p zprodlow
#SBATCH --no-requeue
#SBATCH --exclude=hal-gn09

singularity exec --nv\
  -B /workspaces:/workspaces\
  -B /workspaces/s0001387/raw_od/detectron2:/home/appuser/detectron2_repo\
  -B /workspaces/s0001387/raw_od/pascal_raw:/datasets\
  --pwd /home/appuser/detectron2_repo\
  --env PYTHONPATH=/home/appuser/detectron2_repo\
  /workspaces/s0001387/raw_od/detectron2.sif\
  python3 projects/RawOD/train.py\
  --config-file projects/RawOD/configs/faster_rcnn_R_50_FPN_1x.yaml\
  --experiment_name {name}\
  --format {format}\
  {pretrained}
"""


N_EXPERIMENTS_PER_FORMAT = 3
FORMATS = ["raw", "grey", "jpg"]


def main():
    for format in FORMATS:
        for pretrained in [True, False]:
            if pretrained and format != "jpg":
                continue

            for i in range(N_EXPERIMENTS_PER_FORMAT):
                name = (
                    f"{format}_{str(i).zfill(2)}"
                )

                if pretrained:
                    name += "_pretrained"

                script = SLURM_SCRIPT.format(
                    name=name,
                    format=format,
                    pretrained="--pretrained" if pretrained else "",
                )
                with open(f"projects/RawOD/slurm/{name}.sh", "w") as f:
                    f.write(script)
                subprocess.run(["sbatch", f"projects/RawOD/slurm/{name}.sh"])
                time.sleep(0.5)


if __name__ == "__main__":
    main()
