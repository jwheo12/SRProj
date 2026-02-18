import os
import zipfile

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm.auto import tqdm

from srcnn.config import CFG, DEVICE, seed_everything
from srcnn.dataset import create_dataloaders
from srcnn.engine import inference, train
from srcnn.model import build_model


def init_wandb(cfg):
    if not cfg.get("USE_WANDB", False):
        return None, None

    try:
        import wandb
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "wandb is not installed. Run `pip install wandb` and retry."
        ) from exc

    run = wandb.init(
        project=cfg["WANDB_PROJECT"],
        entity=cfg["WANDB_ENTITY"] or None,
        name=cfg["WANDB_RUN_NAME"] or None,
        mode=cfg.get("WANDB_MODE", "online"),
        config=cfg,
    )
    return wandb, run


def _tensor_to_rgb_image(tensor):
    image = tensor.detach().cpu().numpy().transpose(1, 2, 0)
    image = np.clip(image * 255.0, 0, 255).astype("uint8")
    # OpenCV loaded images are BGR; convert for W&B visualization.
    return image[:, :, ::-1]


def build_train_image_logger(wandb, run, max_images):
    def _log_train_images(epoch, lr_batch, pred_batch, gt_batch):
        lr_images = []
        hr_images = []
        gt_images = []
        sample_count = min(max_images, lr_batch.size(0))
        for idx in range(sample_count):
            lr_img = _tensor_to_rgb_image(lr_batch[idx])
            pred_img = _tensor_to_rgb_image(pred_batch[idx])
            gt_img = _tensor_to_rgb_image(gt_batch[idx])
            lr_images.append(wandb.Image(lr_img, caption=f"epoch={epoch} idx={idx}"))
            hr_images.append(wandb.Image(pred_img, caption=f"epoch={epoch} idx={idx}"))
            gt_images.append(wandb.Image(gt_img, caption=f"epoch={epoch} idx={idx}"))

        if lr_images:
            run.log(
                {
                    "epoch": epoch,
                    "train/LR": lr_images,
                    "train/HR": hr_images,
                    "train/GT": gt_images,
                }
            )

    return _log_train_images


def log_inference_images(wandb, run, pred_name_list, pred_img_list, max_images):
    images = []
    sample_count = min(max_images, len(pred_img_list))
    for idx in range(sample_count):
        image = pred_img_list[idx][:, :, ::-1]
        images.append(wandb.Image(image, caption=pred_name_list[idx]))

    if images:
        run.log({"inference/samples": images})


def save_submission(pred_name_list, pred_img_list, output_dir="./submission", zip_path="./submission.zip"):
    os.makedirs(output_dir, exist_ok=True)
    sub_imgs = []

    for name, pred_img in tqdm(zip(pred_name_list, pred_img_list), total=len(pred_name_list)):
        out_path = os.path.join(output_dir, name)
        cv2.imwrite(out_path, pred_img)
        sub_imgs.append(out_path)

    with zipfile.ZipFile(zip_path, "w") as submission:
        for path in sub_imgs:
            submission.write(path, arcname=os.path.basename(path))

    print("Done.")
    return zip_path


def main():
    seed_everything(CFG["SEED"])
    wandb, wandb_run = init_wandb(CFG)
    train_image_logger = None
    if wandb_run is not None and CFG.get("WANDB_LOG_IMAGES", False):
        train_image_logger = build_train_image_logger(
            wandb=wandb,
            run=wandb_run,
            max_images=CFG["WANDB_MAX_LOG_IMAGES"],
        )

    try:
        train_df = pd.read_csv("./train.csv")
        test_df = pd.read_csv("./test.csv")

        train_loader, test_loader = create_dataloaders(
            train_df=train_df,
            test_df=test_df,
            batch_size=CFG["BATCH_SIZE"],
            test_batch_size=CFG["TEST_BATCH_SIZE"],
            num_workers=6,
        )

        model = nn.DataParallel(build_model(CFG))
        num_params = sum(param.numel() for param in model.parameters())
        print(f"Model: {CFG['MODEL_NAME']} | Params: {num_params:,}")
        if wandb_run is not None:
            wandb_run.log({"model/num_parameters": num_params})
        optimizer = torch.optim.Adam(params=model.parameters(), lr=CFG["LEARNING_RATE"])
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

        infer_model = train(
            model=model,
            optimizer=optimizer,
            train_loader=train_loader,
            scheduler=scheduler,
            device=DEVICE,
            epochs=CFG["EPOCHS"],
            log_fn=wandb_run.log if wandb_run is not None else None,
            image_log_fn=train_image_logger,
            image_log_every=CFG["WANDB_TRAIN_IMAGE_EVERY_N_EPOCHS"],
            num_image_samples=CFG["WANDB_MAX_LOG_IMAGES"],
        )

        pred_img_list, pred_name_list = inference(
            infer_model,
            test_loader,
            DEVICE,
            tile_size=CFG["INFER_TILE_SIZE"],
            overlap=CFG["INFER_TILE_OVERLAP"],
        )
        zip_path = save_submission(pred_name_list, pred_img_list)

        if wandb_run is not None:
            wandb_run.log({"inference/num_images": len(pred_img_list)})
            if CFG.get("WANDB_LOG_IMAGES", False):
                log_inference_images(
                    wandb=wandb,
                    run=wandb_run,
                    pred_name_list=pred_name_list,
                    pred_img_list=pred_img_list,
                    max_images=CFG["WANDB_MAX_LOG_IMAGES"],
                )
            artifact = wandb.Artifact("submission", type="inference")
            artifact.add_file(zip_path)
            wandb_run.log_artifact(artifact)
    finally:
        if wandb_run is not None:
            wandb_run.finish()


if __name__ == "__main__":
    main()
