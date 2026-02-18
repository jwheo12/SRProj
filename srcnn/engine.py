from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm


def _compute_psnr(pred, target):
    mse = F.mse_loss(pred, target, reduction="none").mean(dim=(1, 2, 3))
    mse = torch.clamp(mse, min=1e-12)
    return 10.0 * torch.log10(1.0 / mse)


def _ssim_per_image(pred, target, window_size=11, sigma=1.5):
    channels = pred.size(1)
    dtype = pred.dtype
    device = pred.device

    coords = torch.arange(window_size, dtype=dtype, device=device) - (window_size // 2)
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g = g / g.sum()
    kernel_2d = torch.outer(g, g)
    kernel = kernel_2d.expand(channels, 1, window_size, window_size).contiguous()

    padding = window_size // 2
    mu_pred = F.conv2d(pred, kernel, padding=padding, groups=channels)
    mu_target = F.conv2d(target, kernel, padding=padding, groups=channels)
    mu_pred_sq = mu_pred * mu_pred
    mu_target_sq = mu_target * mu_target
    mu_pred_target = mu_pred * mu_target

    sigma_pred_sq = F.conv2d(pred * pred, kernel, padding=padding, groups=channels) - mu_pred_sq
    sigma_target_sq = (
        F.conv2d(target * target, kernel, padding=padding, groups=channels) - mu_target_sq
    )
    sigma_pred_target = (
        F.conv2d(pred * target, kernel, padding=padding, groups=channels) - mu_pred_target
    )

    c1 = (0.01**2)
    c2 = (0.03**2)
    numerator = (2 * mu_pred_target + c1) * (2 * sigma_pred_target + c2)
    denominator = (mu_pred_sq + mu_target_sq + c1) * (sigma_pred_sq + sigma_target_sq + c2)
    ssim_map = numerator / torch.clamp(denominator, min=1e-12)
    return ssim_map.mean(dim=(1, 2, 3))


def _predict_batch(model, lr_img, tile_size=None, overlap=0):
    if tile_size is None:
        return model(lr_img)

    preds = []
    for idx in range(lr_img.size(0)):
        preds.append(_forward_tiled(model, lr_img[idx : idx + 1], tile_size, overlap))
    return torch.cat(preds, dim=0)


def evaluate(model, val_loader, device, tile_size=None, overlap=0):
    model.eval()
    val_loss = 0.0
    val_psnr = 0.0
    val_ssim = 0.0
    num_samples = 0

    with torch.no_grad():
        for lr_img, hr_img in tqdm(iter(val_loader), leave=False):
            lr_img = lr_img.float().to(device)
            hr_img = hr_img.float().to(device)
            pred_hr = _predict_batch(model, lr_img, tile_size=tile_size, overlap=overlap)

            batch_size = lr_img.size(0)
            num_samples += batch_size
            val_loss += F.mse_loss(pred_hr, hr_img, reduction="mean").item() * batch_size
            val_psnr += _compute_psnr(pred_hr, hr_img).sum().item()
            val_ssim += _ssim_per_image(pred_hr, hr_img).sum().item()

    if num_samples == 0:
        return None

    return {
        "val/loss": val_loss / num_samples,
        "val/psnr": val_psnr / num_samples,
        "val/ssim": val_ssim / num_samples,
    }


def train(
    model,
    optimizer,
    train_loader,
    val_loader,
    scheduler,
    device,
    epochs,
    log_fn=None,
    image_log_fn=None,
    image_log_every=1,
    num_image_samples=4,
    val_every=1,
    val_tile_size=None,
    val_tile_overlap=0,
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
        val_metrics = None
        if val_loader is not None and val_every > 0 and epoch % val_every == 0:
            val_metrics = evaluate(
                model=model,
                val_loader=val_loader,
                device=device,
                tile_size=val_tile_size,
                overlap=val_tile_overlap,
            )
            model.train()

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_state_dict = deepcopy(model.state_dict())

        if val_metrics is None:
            print(f"Epoch : [{epoch}] Train Loss : [{epoch_loss:.5f}]")
        else:
            print(
                f"Epoch : [{epoch}] Train Loss : [{epoch_loss:.5f}] "
                f"Val PSNR : [{val_metrics['val/psnr']:.4f}] "
                f"Val SSIM : [{val_metrics['val/ssim']:.4f}]"
            )
        if log_fn is not None:
            metrics = {
                "epoch": epoch,
                "train/loss": epoch_loss,
                "train/best_loss": best_loss,
                "train/lr": optimizer.param_groups[0]["lr"],
            }
            if val_metrics is not None:
                metrics.update(val_metrics)
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
