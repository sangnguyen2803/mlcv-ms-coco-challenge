from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

DATASET_FOLDER: Path = PROJECT_ROOT / "ms-coco"
PRETRAINED_MODELS_FOLDER: Path = PROJECT_ROOT / "pre-trained_models"
TRAINED_MODELS_FOLDER: Path = PROJECT_ROOT / "trained_models"

IMAGES_DIR: Path = DATASET_FOLDER / "images"
LABELS_DIR: Path = DATASET_FOLDER / "labels"

TRAIN_IMAGES_DIR: Path = IMAGES_DIR / "train-resized"
TEST_IMAGES_DIR: Path = IMAGES_DIR / "test-resized"
TRAIN_LABELS_DIR: Path = LABELS_DIR / "train"

NUM_CLASSES: int = 80

# Keep tested model names nearby; last one is active.
MODEL_NAME = "vit_b_16"
MODEL_NAME = "swin_v2_t"
MODEL_NAME = "regnet_y_800mf"
MODEL_NAME = "convnext_small"
MODEL_NAME = "convnext_tiny"
MODEL_NAME = "mobilenet_v3_large"

BEST_MODEL_PATH: Path = TRAINED_MODELS_FOLDER / "best_model.pt"
FREEZE_BACKBONE: bool = True

CLASSES: tuple[str, ...] = (
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
)
