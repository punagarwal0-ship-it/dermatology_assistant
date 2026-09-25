"""
data/transforms.py

Preprocessing and augmentation for dermatology images.

Deliberately conservative on color: we do NOT apply strong hue shifts,
color jitter with wide ranges, or grayscale conversion, because skin
tone (Fitzpatrick / Monk Skin Tone) is diagnostically and evaluatively
important and must be preserved.

Only geometric augmentation (flip, rotation, mild scaling) and mild
brightness/contrast jitter are used for training. Validation/test use
resize + normalize only.
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_train_transforms(image_size: int = 224) -> A.Compose:
    return A.Compose([
        A.Resize(image_size, image_size),
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=15, p=0.5, border_mode=0),
        # Mild brightness/contrast only - keeps hue/saturation (skin tone) intact
        A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.3),
        A.GaussianBlur(blur_limit=(3, 3), p=0.1),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def get_eval_transforms(image_size: int = 224) -> A.Compose:
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])
