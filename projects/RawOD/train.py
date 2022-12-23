#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
"""
Detectron2 training script with a plain training loop.

This script reads a given config file and runs the training or evaluation.
It is an entry point that is able to train standard models in detectron2.

In order to let one script support training of many models,
this script contains logic that are specific to these built-in models and therefore
may not be suitable for your own project.
For example, your research project perhaps only needs a single "evaluator".

Therefore, we recommend you to use detectron2 as a library and take
this file as an example of how to use the library.
You may want to write your own script with your datasets and other customizations.

Compared to "train_net.py", this script supports fewer default features.
It also includes fewer abstraction, therefore is easier to add custom logic.
"""

import logging
import os
from argparse import ArgumentParser
from collections import OrderedDict
from dataclasses import dataclass

import torch
from torch.nn.parallel import DistributedDataParallel

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer, PeriodicCheckpointer
from detectron2.config import get_cfg
from detectron2.data import (
    build_detection_test_loader,
    build_detection_train_loader,
)
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import (
    default_argument_parser,
    default_setup,
    default_writers,
    launch,
)
from detectron2.evaluation import (
    COCOEvaluator,
    inference_on_dataset,
    print_csv_format,
)
from detectron2.modeling import build_model
from detectron2.solver import build_lr_scheduler, build_optimizer
from detectron2.utils.events import EventStorage

logger = logging.getLogger("detectron2")
PASCAL_DATASET_NAME = "pascal_raw_{train_or_val}_{format}_{resolution}"
NOD_DATASET_NAME = "nod_raw_{camera}_{train_or_val}_{raw_or_jpg}"


@dataclass
class RawODConfig:
    experiment_name: str
    format: str = "jpg"
    resolution: str = "fifth"
    max_iter: int = 150_000
    batch_size: int = 16
    lr: float = 3e-4
    eval_period: int = 5000
    pretrained: bool = False
    nod_dataset: bool = False
    use_normalize: bool = False
    use_yeojohnson: bool = False
    use_gamma: bool = False
    use_erf: bool = False
    normalize_with_max: bool = False


def add_args(argparser: ArgumentParser) -> ArgumentParser:
    # add format and resolution
    argparser.add_argument("--format", type=str)
    # add resolution
    argparser.add_argument("--resolution", type=str)
    # add max_iter
    argparser.add_argument("--max_iter", type=int)
    # add batch_size
    argparser.add_argument("--batch_size", type=int)
    # add lr
    argparser.add_argument("--lr", type=float)
    # add eval_period
    argparser.add_argument("--eval_period", type=int)
    # add experiment_name
    argparser.add_argument("--experiment_name", type=str, required=True)
    # add pretrained
    argparser.add_argument("--pretrained", action="store_true")
    # add nod_dataset
    argparser.add_argument("--nod_dataset", action="store_true")
    # add use_normalize
    argparser.add_argument("--use_normalize", action="store_true")
    # add use_yeojohnson
    argparser.add_argument("--use_yeojohnson", action="store_true")
    # add use_gamma
    argparser.add_argument("--use_gamma", action="store_true")
    # add use_erf
    argparser.add_argument("--use_erf", action="store_true")
    # add normalize_with_max
    argparser.add_argument("--normalize_with_max", action="store_true")

    return argparser


def register_datasets(format: str, resolution: str):
    if "raw" in format:
        linear = "linear_" if "linear" in format else ""
        json_path = "/datasets/pascal_raw/trainval/{resolution}_{train_or_val}_raw.json"
        image_root = f"/datasets/pascal_raw/{resolution}/{linear}raw"
    elif "jpg" in format:
        downsampled = "downsampled_" if "downsampled" in format else ""
        json_path = "/datasets/pascal_raw/trainval/{resolution}_{train_or_val}_jpg.json"
        image_root = f"/datasets/pascal_raw/{resolution}/{downsampled}jpg"
    elif "grey" in format:
        json_path = "/datasets/pascal_raw/trainval/{resolution}_{train_or_val}_jpg.json"
        image_root = f"/datasets/pascal_raw/{resolution}/{format}"
    else:
        raise ValueError(f"Unknown format: {format}")

    for train_or_val in ["train", "val"]:
        register_coco_instances(
            PASCAL_DATASET_NAME.format(
                train_or_val=train_or_val, format=format, resolution=resolution
            ),
            {},
            json_path.format(resolution=resolution, train_or_val=train_or_val),
            image_root,
        )


def register_nod_datasets(format: str):
    nod_raw_root = "/datasets/nod_raw"

    if "raw" in format:
        raw_or_jpg = "raw"
    elif "jpg" in format:
        raw_or_jpg = "jpg"

    for camera, resolution in [("nikon", "third"), ("sony", "fifth")]:
        for train_or_val in ["train", "val"]:
            register_coco_instances(
                name=NOD_DATASET_NAME.format(
                    camera=camera, train_or_val=train_or_val, raw_or_jpg=raw_or_jpg
                ),
                metadata={},
                json_file=os.path.join(
                    nod_raw_root,
                    camera,
                    "annotations",
                    f"{camera}_{train_or_val}_{raw_or_jpg}.json",
                ),
                image_root=os.path.join(nod_raw_root, camera, resolution, raw_or_jpg),
            )


def do_test(cfg, model, storage=None):
    results = OrderedDict()
    for dataset_name in cfg.DATASETS.TEST:
        data_loader = build_detection_test_loader(cfg, dataset_name)

        evaluator = COCOEvaluator(dataset_name, output_dir=cfg.OUTPUT_DIR)

        results_i = inference_on_dataset(model, data_loader, evaluator)
        results[dataset_name] = results_i
        if comm.is_main_process():
            logger.info("Evaluation results for {} in csv format:".format(dataset_name))
            print_csv_format(results_i)

        if storage is not None:
            for k, v in results_i["bbox"].items():
                if k in ("APs", "APm", "APl"):
                    continue
                storage.put_scalar(k, v, smoothing_hint=False)

    if len(results) == 1:
        results = list(results.values())[0]
    return results


def do_train(cfg, model, resume=False):
    model.train()
    optimizer = build_optimizer(cfg, model)
    scheduler = build_lr_scheduler(cfg, optimizer)

    checkpointer = DetectionCheckpointer(
        model, cfg.OUTPUT_DIR, optimizer=optimizer, scheduler=scheduler
    )
    start_iter = (
        checkpointer.resume_or_load(cfg.MODEL.WEIGHTS, resume=resume).get(
            "iteration", -1
        )
        + 1
    )
    max_iter = cfg.SOLVER.MAX_ITER

    periodic_checkpointer = PeriodicCheckpointer(
        checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD, max_iter=max_iter
    )

    writers = (
        default_writers(cfg.OUTPUT_DIR, max_iter) if comm.is_main_process() else []
    )

    # compared to "train_net.py", we do not support accurate timing and
    # precise BN here, because they are not trivial to implement in a small training loop
    data_loader = build_detection_train_loader(cfg)
    logger.info("Starting training from iteration {}".format(start_iter))
    with EventStorage(start_iter) as storage:
        for data, iteration in zip(data_loader, range(start_iter, max_iter)):
            storage.iter = iteration
            loss_dict = model(data)
            losses = sum(loss_dict.values())
            assert torch.isfinite(losses).all(), loss_dict

            loss_dict_reduced = {
                k: v.item() for k, v in comm.reduce_dict(loss_dict).items()
            }
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())
            if comm.is_main_process():
                storage.put_scalars(total_loss=losses_reduced, **loss_dict_reduced)

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            storage.put_scalar(
                "params/lambda",
                model.backbone.bottom_up.stem.learnable_isp.lambda_param.item(),
                smoothing_hint=False,
            )
            storage.put_scalar(
                "params/gamma",
                model.backbone.bottom_up.stem.learnable_isp.gamma_param.item(),
                smoothing_hint=False,
            )
            storage.put_scalar(
                "params/mu",
                model.backbone.bottom_up.stem.learnable_isp.mu_param.item(),
                smoothing_hint=False,
            )
            storage.put_scalar(
                "params/sigma",
                model.backbone.bottom_up.stem.learnable_isp.sigma_param.item(),
                smoothing_hint=False,
            )

            scheduler.step()

            if (
                cfg.TEST.EVAL_PERIOD > 0
                and (iteration + 1) % cfg.TEST.EVAL_PERIOD == 0
                and iteration != max_iter - 1
            ):
                do_test(cfg, model, storage)
                # Compared to "train_net.py", the test results are not dumped to EventStorage
                comm.synchronize()

            if iteration - start_iter > 5 and (
                (iteration + 1) % 20 == 0 or iteration == max_iter - 1
            ):
                for writer in writers:
                    writer.write()
            periodic_checkpointer.step(iteration)


def set_datasets(cfg, run_config: RawODConfig):
    if run_config.nod_dataset:
        cfg.DATASETS.TRAIN = (
            NOD_DATASET_NAME.format(
                train_or_val="train",
                camera="nikon",
                raw_or_jpg="raw" if "raw" in run_config.format else "jpg",
            ),
            NOD_DATASET_NAME.format(
                train_or_val="train",
                camera="sony",
                raw_or_jpg="raw" if "raw" in run_config.format else "jpg",
            ),
        )
        cfg.DATASETS.TEST = (
            NOD_DATASET_NAME.format(
                train_or_val="val",
                camera="nikon",
                raw_or_jpg="raw" if "raw" in run_config.format else "jpg",
            ),
            NOD_DATASET_NAME.format(
                train_or_val="val",
                camera="sony",
                raw_or_jpg="raw" if "raw" in run_config.format else "jpg",
            ),
        )
    else:
        cfg.DATASETS.TRAIN = (
            PASCAL_DATASET_NAME.format(
                train_or_val="train",
                format=run_config.format,
                resolution=run_config.resolution,
            ),
        )
        cfg.DATASETS.TEST = (
            PASCAL_DATASET_NAME.format(
                train_or_val="val",
                format=run_config.format,
                resolution=run_config.resolution,
            ),
        )


def setup(args, run_config: RawODConfig):
    """
    Create configs and perform basic setups.
    """
    # jpg = rgb, grey=grayscale, raw=np.uint16
    register_datasets(format=run_config.format, resolution=run_config.resolution)
    register_nod_datasets(run_config.format)

    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)

    set_datasets(cfg, run_config)

    # Basic setup
    cfg.SOLVER.IMS_PER_BATCH = run_config.batch_size  # batch_size
    # set the warmup
    cfg.SOLVER.WARMUP_FACTOR = 0.001
    cfg.SOLVER.WARMUP_ITERS = 5000
    cfg.SOLVER.BASE_LR = run_config.lr
    cfg.SOLVER.MAX_ITER = run_config.max_iter
    # set the validation period to 5000 iterations
    cfg.TEST.EVAL_PERIOD = run_config.eval_period
    cfg.DATALOADER.NUM_WORKERS = 16
    cfg.SOLVER.STEPS = [
        int(0.7 * run_config.max_iter),
    ]
    cfg.SOLVER.GAMMA = 0.1
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3
    cfg.MODEL.RETINANET.NUM_CLASSES = 3

    # settings that are importatnt for the raw data, the rgb should have the same settings
    # ps. trying out these new settings to see if augmentations helps even though the do not make
    # sense for raw data
    cfg.INPUT.MIN_SIZE_TRAIN = (802,) if not run_config.nod_dataset else (880,)
    cfg.INPUT.MIN_SIZE_TEST = (802,) if not run_config.nod_dataset else (880,)
    # remove horizontal flip
    cfg.INPUT.RANDOM_FLIP = "none"

    # set the output directory
    cfg.OUTPUT_DIR = os.path.join(
        cfg.OUTPUT_DIR,
        f"{'nod_' if run_config.nod_dataset else ''}learnable_isp_02_zprod",
        run_config.experiment_name,
    )
    if "raw" in run_config.format:
        cfg.MODEL.RESNETS.USE_NORMALIZE = run_config.use_normalize
        cfg.MODEL.RESNETS.USE_YEOJOHNSON = run_config.use_yeojohnson
        cfg.MODEL.RESNETS.USE_GAMMA = run_config.use_gamma
        cfg.MODEL.RESNETS.USE_ERF = run_config.use_erf
        cfg.MODEL.RESNETS.NORMALIZE_WITH_MAX = run_config.normalize_with_max
    else:
        cfg.MODEL.RESNETS.USE_NORMALIZE = False
        cfg.MODEL.RESNETS.USE_YEOJOHNSON = False
        cfg.MODEL.RESNETS.USE_GAMMA = False
        cfg.MODEL.RESNETS.USE_ERF = False
        cfg.MODEL.RESNETS.NORMALIZE_WITH_MAX = False

    # train from scratch
    if run_config.pretrained:
        cfg.MODEL.WEIGHTS = (
            "projects/RawvOD/weights/model_final_b275ba_no_first_layer.pkl"
        )
    else:
        cfg.MODEL.WEIGHTS = ""
        cfg.MODEL.BACKBONE.FREEZE_AT = 0

    if "raw" in run_config.format:
        # these are removed prior to training. Note that the length of this list vector is used to initialize the
        # first layer of the model, hence we need different depending on what tpye is used.
        if "reshape" in run_config.format:
            cfg.MODEL.PIXEL_MEAN = [0.0] * 4
            cfg.MODEL.PIXEL_STD = [1.0] * 4
        elif "plain" in run_config.format:
            cfg.MODEL.PIXEL_MEAN = [0.0] * 3
            cfg.MODEL.PIXEL_STD = [1.0] * 3
        elif "offset" in run_config.format:
            cfg.MODEL.PIXEL_MEAN = [0.0] * 4
            cfg.MODEL.PIXEL_STD = [1.0] * 4
        elif "rgb" in run_config.format:
            cfg.MODEL.PIXEL_MEAN = [0.0] * 3
            cfg.MODEL.PIXEL_STD = [1.0] * 3
        else:
            raise NotImplementedError("Unknown format: {}".format(run_config.format))

        cfg.INPUT.FORMAT = run_config.format
    elif run_config.format == "grey":
        cfg.MODEL.PIXEL_MEAN = [110.794]
        cfg.MODEL.PIXEL_STD = [63.426]
        cfg.INPUT.FORMAT = run_config.format
    elif "jpg" in run_config.format:
        cfg.MODEL.PIXEL_MEAN = [126.398, 108.167, 83.467]
        cfg.MODEL.PIXEL_STD = [68.696, 62.867, 60.831]
        cfg.INPUT.FORMAT = run_config.format
    else:
        raise ValueError(f"Unknown format {run_config.format}")

    cfg.freeze()
    default_setup(
        cfg, args
    )  # if you don't like any of the default setup, write your own setup code
    return cfg


def main(args, run_config: RawODConfig):
    cfg = setup(args, run_config)

    model = build_model(cfg)
    logger.info("Model:\n{}".format(model))
    if args.eval_only:
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        return do_test(cfg, model)

    distributed = comm.get_world_size() > 1
    if distributed:
        model = DistributedDataParallel(
            model, device_ids=[comm.get_local_rank()], broadcast_buffers=False
        )

    do_train(cfg, model, resume=args.resume)
    return do_test(cfg, model)


if __name__ == "__main__":
    parser = default_argument_parser()
    parser = add_args(parser)
    args = parser.parse_args()
    # create the run cofnig for the arument names that exists in RawODConfig
    run_config = RawODConfig(
        **{
            k: v
            for k, v in vars(args).items()
            if k in RawODConfig.__dataclass_fields__ and v is not None
        }
    )

    print("Command Line Args:", args)
    main(args, run_config)
