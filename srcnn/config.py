import os
import random
import warnings

import numpy as np
import torch


CFG = {
    "MODEL_NAME": "rrdb_refiner",
    "MODEL_FEATURES": 64,
    "MODEL_BLOCKS": 16,
    "MODEL_GROWTH_CHANNELS": 32,
    "UPSCALE_FACTOR": 4,
    "TRAIN_PATCH_SIZE": 256,
    "INFER_TILE_SIZE": 512,
    "INFER_TILE_OVERLAP": 64,
    "EPOCHS": 30,
    "LEARNING_RATE": 1e-4,
    "BATCH_SIZE": 4,
    "VAL_BATCH_SIZE": 2,
    "VAL_RATIO": 0.1,
    "VAL_EVERY_N_EPOCHS": 1,
    "VAL_TILE_SIZE": 512,
    "VAL_TILE_OVERLAP": 64,
    "TEST_BATCH_SIZE": 2,
    "SEED": 41,
    "USE_WANDB": True,
    "WANDB_PROJECT": "srcnn",
    "WANDB_ENTITY": "",
    "WANDB_RUN_NAME": "srcnn-patch-tile",
    "WANDB_MODE": "online",
    "WANDB_LOG_IMAGES": True,
    "WANDB_MAX_LOG_IMAGES": 4,
    "WANDB_TRAIN_IMAGE_EVERY_N_EPOCHS": 1,
}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def get_device() -> torch.device:
    # Some environments emit a warning on CUDA probe when driver/runtime is incompatible.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"CUDA initialization: The NVIDIA driver on your system is too old.*",
            category=UserWarning,
        )
        use_cuda = torch.cuda.is_available()

    return torch.device("cuda" if use_cuda else "cpu")


DEVICE = get_device()
