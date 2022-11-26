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
    MetadataCatalog,
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
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator,
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    PascalVOCDetectionEvaluator,
    SemSegEvaluator,
    inference_on_dataset,
    print_csv_format,
)
from detectron2.modeling import build_model
from detectron2.solver import build_lr_scheduler, build_optimizer
from detectron2.utils.events import EventStorage

logger = logging.getLogger("detectron2")
DATASET_NAME = "pascal_raw_{train_or_val}_{format}_{resolution}"


@dataclass
class RawODConfig:
    experiment_name: str
    format: str = "jpg"
    resolution: str = "fifth"
    max_iter: int = 5000
    batch_size: int = 16
    lr: float = 0.00025
    eval_period: int = 1500
    pretrained: bool = False

    def __post_init__(self):
        assert self.format in ["jpg", "grey", "raw"], f"Invalid format: {self.format}"
        assert self.resolution in ["fifth", "full"]


def add_args(argparser: ArgumentParser) -> ArgumentParser:
    # add format and resolution
    argparser.add_argument("--format", type=str, default="jpg")
    # add resolution
    argparser.add_argument("--resolution", type=str, default="fifth")
    # add max_iter
    argparser.add_argument("--max_iter", type=int, default=3000)
    # add batch_size
    argparser.add_argument("--batch_size", type=int, default=32)
    # add lr
    argparser.add_argument("--lr", type=float, default=0.00025)
    # add eval_period
    argparser.add_argument("--eval_period", type=int, default=1000)
    # add experiment_name
    argparser.add_argument("--experiment_name", type=str, required=True)
    # add pretrained
    argparser.add_argument("--pretrained", action="store_true")


    return argparser


def register_datasets(format: str, resolution: str):
    assert format in ("raw", "jpg", "grey")

    image_root = f"/datasets/pascal_raw/{resolution}/{format}"

    format_ = "jpg" if format == "grey" else format

    for train_or_val in ["train", "val"]:
        register_coco_instances(
            DATASET_NAME.format(
                train_or_val=train_or_val, format=format, resolution=resolution
            ),
            {},
            f"/datasets/pascal_raw/trainval/{resolution}_{train_or_val}_{format_}.json",
            image_root,
        )


def do_test(cfg, model):
    results = OrderedDict()
    for dataset_name in cfg.DATASETS.TEST:
        data_loader = build_detection_test_loader(cfg, dataset_name)

        evaluator = COCOEvaluator(dataset_name, output_dir=cfg.OUTPUT_DIR)

        results_i = inference_on_dataset(model, data_loader, evaluator)
        results[dataset_name] = results_i
        if comm.is_main_process():
            logger.info("Evaluation results for {} in csv format:".format(dataset_name))
            print_csv_format(results_i)
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
                "lr", optimizer.param_groups[0]["lr"], smoothing_hint=False
            )
            scheduler.step()

            if (
                    cfg.TEST.EVAL_PERIOD > 0
                    and (iteration + 1) % cfg.TEST.EVAL_PERIOD == 0
                    and iteration != max_iter - 1
            ):
                do_test(cfg, model)
                # Compared to "train_net.py", the test results are not dumped to EventStorage
                comm.synchronize()

            if iteration - start_iter > 5 and (
                    (iteration + 1) % 20 == 0 or iteration == max_iter - 1
            ):
                for writer in writers:
                    writer.write()
            periodic_checkpointer.step(iteration)


def setup(args, run_config: RawODConfig):
    """
    Create configs and perform basic setups.
    """
    # jpg = rgb, grey=grayscale, raw=np.uint16
    register_datasets(format=run_config.format, resolution=run_config.resolution)

    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.DATASETS.TRAIN = (
        DATASET_NAME.format(
            train_or_val="train",
            format=run_config.format,
            resolution=run_config.resolution,
        ),
    )
    cfg.DATASETS.TEST = (
        DATASET_NAME.format(
            train_or_val="val",
            format=run_config.format,
            resolution=run_config.resolution,
        ),
    )

    # set the output directory
    cfg.OUTPUT_DIR = os.path.join(cfg.OUTPUT_DIR, run_config.experiment_name)

    # train from scratch
    if run_config.pretrained and run_config.format == "jpg":
        cfg.MODEL.WEIGHTS = "projects/RawOD/weights/model_final_b275ba.pkl"
    else:
        cfg.MODEL.WEIGHTS = ""

    if run_config.format == "raw":
        cfg.MODEL.PIXEL_MEAN = [336.4967]
        cfg.MODEL.PIXEL_STD = [441.862]
        cfg.INPUT.FORMAT = "RAW"
    elif run_config.format == "grey":
        cfg.MODEL.PIXEL_MEAN = [110.794]
        cfg.MODEL.PIXEL_STD = [63.426]
        cfg.INPUT.FORMAT = "GREY"
    elif run_config.format == "jpg":
        cfg.MODEL.PIXEL_MEAN = [126.398, 108.167, 83.467]
        cfg.MODEL.PIXEL_STD = [68.696, 62.867, 60.831]
        cfg.INPUT.FORMAT = "RGB"
    else:
        raise ValueError(f"Unknown format {run_config.format}")

    cfg.SOLVER.IMS_PER_BATCH = run_config.batch_size  # batch_size

    cfg.SOLVER.BASE_LR = run_config.lr
    cfg.SOLVER.MAX_ITER = run_config.max_iter
    # set the validation period to 5000 iterations
    cfg.TEST.EVAL_PERIOD = run_config.eval_period

    cfg.DATALOADER.NUM_WORKERS = 2

    cfg.SOLVER.STEPS = [int(0.8 * run_config.max_iter)]
    cfg.SOLVER.GAMMA = 0.5
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3
    cfg.INPUT.MIN_SIZE_TRAIN = cfg.INPUT.MIN_SIZE_TEST = (802,)

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

    run_config = RawODConfig(
        experiment_name=args.experiment_name,
        format=args.format,
        resolution=args.resolution,
        max_iter=args.max_iter,
        batch_size=args.batch_size,
        lr=args.lr,
        eval_period=args.eval_period,
    )

    print("Command Line Args:", args)
    main(args, run_config)
