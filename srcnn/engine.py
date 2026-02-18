from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm


def train(
    model,
    optimizer,
    train_loader,
    scheduler,
    device,
    epochs,
    log_fn=None,
    image_log_fn=None,
    image_log_every=1,
    num_image_samples=4,
):
    model.to(device)
    criterion = nn.MSELoss().to(device)
    best_state_dict = None
    best_loss = float("inf")
    fixed_image_batch = None

    for epoch in range(1, epochs + 1):
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        model.train()
        train_loss = []

        for lr_img, hr_img in tqdm(iter(train_loader)):
            lr_img = lr_img.float().to(device)
            hr_img = hr_img.float().to(device)

            optimizer.zero_grad()
            pred_hr_img = model(lr_img)
            loss = criterion(pred_hr_img, hr_img)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())

            if image_log_fn is not None and fixed_image_batch is None:
                sample_count = min(num_image_samples, lr_img.size(0))
                if sample_count > 0:
                    fixed_image_batch = (
                        lr_img[:sample_count].detach().cpu(),
                        hr_img[:sample_count].detach().cpu(),
                    )

        if scheduler is not None:
            scheduler.step()

        epoch_loss = float(np.mean(train_loss))
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_state_dict = deepcopy(model.state_dict())

        print(f"Epoch : [{epoch}] Train Loss : [{epoch_loss:.5f}]")
        if log_fn is not None:
            metrics = {
                "epoch": epoch,
                "train/loss": epoch_loss,
                "train/best_loss": best_loss,
                "train/lr": optimizer.param_groups[0]["lr"],
            }
            if device.type == "cuda":
                metrics["train/peak_vram_mb"] = (
                    torch.cuda.max_memory_allocated(device) / 1024 / 1024
                )
            log_fn(metrics)

        should_log_images = (
            image_log_fn is not None
            and fixed_image_batch is not None
            and image_log_every > 0
            and epoch % image_log_every == 0
        )
        if should_log_images:
            model.eval()
            with torch.no_grad():
                fixed_lr = fixed_image_batch[0].to(device)
                fixed_pred = model(fixed_lr).detach().cpu()
            model.train()
            image_log_fn(epoch, fixed_image_batch[0], fixed_pred, fixed_image_batch[1])

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
    return model


def _tile_starts(length, tile_size, overlap):
    if tile_size >= length:
        return [0]

    stride = tile_size - overlap
    if stride <= 0:
        raise ValueError("INFER_TILE_OVERLAP must be smaller than INFER_TILE_SIZE")

    starts = list(range(0, length - tile_size + 1, stride))
    last = length - tile_size
    if starts[-1] != last:
        starts.append(last)
    return starts


def _forward_tiled(model, image, tile_size, overlap):
    _, channels, height, width = image.shape
    if tile_size >= height and tile_size >= width:
        return model(image)

    output = torch.zeros((1, channels, height, width), device=image.device)
    weight = torch.zeros((1, 1, height, width), device=image.device)

    y_starts = _tile_starts(height, tile_size, overlap)
    x_starts = _tile_starts(width, tile_size, overlap)

    for y in y_starts:
        for x in x_starts:
            tile = image[:, :, y : y + tile_size, x : x + tile_size]
            pred = model(tile)
            tile_h, tile_w = pred.shape[2], pred.shape[3]
            output[:, :, y : y + tile_h, x : x + tile_w] += pred
            weight[:, :, y : y + tile_h, x : x + tile_w] += 1.0

    return output / weight.clamp_min(1e-6)


def inference(model, test_loader, device, tile_size, overlap):
    model.to(device)
    model.eval()
    pred_img_list = []
    name_list = []

    with torch.no_grad():
        for lr_img_batch, file_name_batch in tqdm(iter(test_loader)):
            for idx, name in enumerate(file_name_batch):
                lr_img = lr_img_batch[idx : idx + 1].float().to(device)
                pred_hr_img = _forward_tiled(model, lr_img, tile_size, overlap)
                pred = pred_hr_img[0].cpu().clone().detach().numpy()
                pred = pred.transpose(1, 2, 0)
                pred = np.clip(pred * 255.0, 0, 255)
                pred_img_list.append(pred.astype("uint8"))
                name_list.append(name)

    return pred_img_list, name_list
