import os
import json
import numpy as np

ROOT = "/workspaces/s0001387/raw_od/nod_raw"
SONY_ANNOTATIONS = os.path.join(ROOT, "sony", "annotations")
NIKON_ANNOTATIONS = os.path.join(ROOT, "nikon", "annotations")
SONY_RAW_TRAIN_FILE = os.path.join(SONY_ANNOTATIONS, "raw_new_Sony_RX100m7_train.json")
SONY_RAW_VAL_FILE = os.path.join(SONY_ANNOTATIONS, "raw_new_Sony_RX100m7_val.json")
SONY_JPG_TRAIN_FILE = os.path.join(
    SONY_ANNOTATIONS, "rawpy_new_Sony_RX100m7_train.json"
)
SONY_JPG_VAL_FILE = os.path.join(SONY_ANNOTATIONS, "rawpy_new_Sony_RX100m7_val.json")

NIKON_JPG_TRAIN_FILE = os.path.join(NIKON_ANNOTATIONS, "rawpy_new_Nikon750_train.json")
NIKON_JPG_VAL_FILE = os.path.join(NIKON_ANNOTATIONS, "rawpy_new_Nikon750_val.json")
NIKON_RAW_TRAIN_FILE = os.path.join(NIKON_ANNOTATIONS, "raw_new_Nikon750_train.json")
NIKON_RAW_VAL_FILE = os.path.join(NIKON_ANNOTATIONS, "raw_new_Nikon750_val.json")

NIKON_OUTPUT_FILENAME = os.path.join(
    NIKON_ANNOTATIONS, "nikon_{train_or_val}_{raw_or_jpg}.json"
)
SONY_OUTPUT_FILENAME = os.path.join(
    SONY_ANNOTATIONS, "sony_{train_or_val}_{raw_or_jpg}.json"
)


NIKON_HEIGHT = 880  # 2640 // 3 = 880
NIKON_WIDTH = 1322  # 3968 // 3 = 1322
NIKON_SCALING = 3
NIKON_EXT = ".NEF"

SONY_HEIGHT = 734  # 3672 // 5 = 734
SONY_WIDTH = 1099  # 5496 // 5 = 1099
SONY_SCALING = 5
SONY_EXT = ".ARW"


def load_annotation_files():
    # load sony annotations
    with open(SONY_RAW_TRAIN_FILE, "r") as f:
        sony_raw_train = json.load(f)
    with open(SONY_RAW_VAL_FILE, "r") as f:
        sony_raw_val = json.load(f)
    with open(SONY_JPG_TRAIN_FILE, "r") as f:
        sony_jpg_train = json.load(f)
    with open(SONY_JPG_VAL_FILE, "r") as f:
        sony_jpg_val = json.load(f)

    # load nikon annotations
    with open(NIKON_RAW_TRAIN_FILE, "r") as f:
        nikon_raw_train = json.load(f)
    with open(NIKON_RAW_VAL_FILE, "r") as f:
        nikon_raw_val = json.load(f)
    with open(NIKON_JPG_TRAIN_FILE, "r") as f:
        nikon_jpg_train = json.load(f)
    with open(NIKON_JPG_VAL_FILE, "r") as f:
        nikon_jpg_val = json.load(f)

    return (
        sony_raw_train,
        sony_raw_val,
        sony_jpg_train,
        sony_jpg_val,
        nikon_raw_train,
        nikon_raw_val,
        nikon_jpg_train,
        nikon_jpg_val,
    )


def recreate_annotations(annotation: dict, scale: int):
    """Recreate the annotations in coco format.
    The annotations are scaled down along with the image height, width and area.

    """
    for image in annotation["images"]:
        if scale == SONY_SCALING:
            image["file_name"] = image["file_name"].replace(".ARW", ".npy")
            image["height"] = SONY_HEIGHT
            image["width"] = SONY_WIDTH
        elif scale == NIKON_SCALING:
            image["file_name"] = image["file_name"].replace(".NEF", ".npy")
            image["height"] = NIKON_HEIGHT
            image["width"] = NIKON_WIDTH

        image["file_name"] = image["file_name"].replace(".JPG", ".jpg")

    for a in annotation["annotations"]:
        a["bbox"] = [
            a["bbox"][0] // scale,
            a["bbox"][1] // scale,
            a["bbox"][2] // scale,
            a["bbox"][3] // scale,
        ]
        a["area"] //= scale**2

    return annotation


def main():

    annotation_files = load_annotation_files()
    sony_annotation_files = annotation_files[:4]
    nikon_annotation_files = annotation_files[4:]

    # recreate sony annotations
    sony_raw_train = recreate_annotations(sony_annotation_files[0], SONY_SCALING)
    sony_raw_val = recreate_annotations(sony_annotation_files[1], SONY_SCALING)
    sony_jpg_train = recreate_annotations(sony_annotation_files[2], SONY_SCALING)
    sony_jpg_val = recreate_annotations(sony_annotation_files[3], SONY_SCALING)

    # recreate nikon annotations
    nikon_raw_train = recreate_annotations(nikon_annotation_files[0], NIKON_SCALING)
    nikon_raw_val = recreate_annotations(nikon_annotation_files[1], NIKON_SCALING)
    nikon_jpg_train = recreate_annotations(nikon_annotation_files[2], NIKON_SCALING)
    nikon_jpg_val = recreate_annotations(nikon_annotation_files[3], NIKON_SCALING)

    # save annotations
    with open(
        SONY_OUTPUT_FILENAME.format(train_or_val="train", raw_or_jpg="raw"), "w"
    ) as f:
        json.dump(sony_raw_train, f)
    with open(
        SONY_OUTPUT_FILENAME.format(train_or_val="val", raw_or_jpg="raw"), "w"
    ) as f:
        json.dump(sony_raw_val, f)
    with open(
        SONY_OUTPUT_FILENAME.format(train_or_val="train", raw_or_jpg="jpg"), "w"
    ) as f:
        json.dump(sony_jpg_train, f)
    with open(
        SONY_OUTPUT_FILENAME.format(train_or_val="val", raw_or_jpg="jpg"), "w"
    ) as f:
        json.dump(sony_jpg_val, f)

    with open(
        NIKON_OUTPUT_FILENAME.format(train_or_val="train", raw_or_jpg="raw"), "w"
    ) as f:
        json.dump(nikon_raw_train, f)
    with open(
        NIKON_OUTPUT_FILENAME.format(train_or_val="val", raw_or_jpg="raw"), "w"
    ) as f:
        json.dump(nikon_raw_val, f)
    with open(
        NIKON_OUTPUT_FILENAME.format(train_or_val="train", raw_or_jpg="jpg"), "w"
    ) as f:
        json.dump(nikon_jpg_train, f)
    with open(
        NIKON_OUTPUT_FILENAME.format(train_or_val="val", raw_or_jpg="jpg"), "w"
    ) as f:
        json.dump(nikon_jpg_val, f)


if __name__ == "__main__":
    main()
