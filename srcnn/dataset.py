import os

import albumentations as A
import cv2
from albumentations.pytorch.transforms import ToTensorV2
from torch.utils.data import DataLoader, Dataset

from .config import CFG


class CustomDataset(Dataset):
    def __init__(self, df, transforms, train_mode):
        self.df = df
        self.transforms = transforms
        self.train_mode = train_mode

    def __getitem__(self, index):
        lr_path = self.df["LR"].iloc[index]
        lr_img = cv2.imread(lr_path)
        if lr_img is None:
            raise FileNotFoundError(f"Failed to load LR image: {lr_path}")

        if self.train_mode:
            hr_path = self.df["HR"].iloc[index]
            hr_img = cv2.imread(hr_path)
            if hr_img is None:
                raise FileNotFoundError(f"Failed to load HR image: {hr_path}")
            # SRCNN learns residual details on bicubic-upsampled LR in HR space.
            hr_h, hr_w = hr_img.shape[:2]
            lr_img = cv2.resize(lr_img, (hr_w, hr_h), interpolation=cv2.INTER_CUBIC)

            if self.transforms is not None:
                transformed = self.transforms(image=lr_img, label=hr_img)
                lr_img = transformed["image"] / 255.0
                hr_img = transformed["label"] / 255.0
            return lr_img, hr_img

        file_name = os.path.basename(lr_path)
        lr_h, lr_w = lr_img.shape[:2]
        lr_img = cv2.resize(
            lr_img,
            (lr_w * CFG["UPSCALE_FACTOR"], lr_h * CFG["UPSCALE_FACTOR"]),
            interpolation=cv2.INTER_CUBIC,
        )
        if self.transforms is not None:
            transformed = self.transforms(image=lr_img)
            lr_img = transformed["image"] / 255.0
        return lr_img, file_name

    def __len__(self):
        return len(self.df)


def get_train_transform():
    patch_size = CFG["TRAIN_PATCH_SIZE"]
    return A.Compose(
        [
            A.RandomCrop(height=patch_size, width=patch_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            ToTensorV2(p=1.0),
        ],
        additional_targets={"label": "image"},
    )


def get_val_transform():
    patch_size = CFG["TRAIN_PATCH_SIZE"]
    return A.Compose(
        [A.CenterCrop(height=patch_size, width=patch_size), ToTensorV2(p=1.0)],
        additional_targets={"label": "image"},
    )


def get_test_transform():
    return A.Compose([ToTensorV2(p=1.0)])


def create_dataloaders(
    train_df,
    val_df,
    test_df,
    batch_size,
    val_batch_size=2,
    test_batch_size=2,
    num_workers=6,
):
    train_dataset = CustomDataset(train_df, get_train_transform(), True)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )

    val_loader = None
    if val_df is not None and len(val_df) > 0:
        val_dataset = CustomDataset(val_df, get_val_transform(), True)
        val_loader = DataLoader(
            val_dataset,
            batch_size=val_batch_size,
            shuffle=False,
            num_workers=num_workers,
        )

    test_dataset = CustomDataset(test_df, get_test_transform(), False)
    test_loader = DataLoader(
        test_dataset,
        batch_size=test_batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, val_loader, test_loader
