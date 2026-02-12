from dataclasses import dataclass
from torchvision import transforms

@dataclass
class AugmentCfg:
    random_resized_crop: bool = True
    hflip_prob: float = 0.5
    color_jitter: bool = True
    gaussian_blur: bool = True
    random_rotation_deg: int = 12
    random_erasing: bool = True

@dataclass
class EvalCfg:
    resize_short: int = 256
    center_crop: int = 224

def build_train_tf(image_size: int, cfg: AugmentCfg):
    t = []
    if cfg.random_resized_crop:
        t.append(transforms.RandomResizedCrop(image_size, scale=(0.6, 1.0)))
    else:
        t.append(transforms.Resize((image_size, image_size)))

    if cfg.hflip_prob > 0:
        t.append(transforms.RandomHorizontalFlip(p=cfg.hflip_prob))

    if cfg.random_rotation_deg and cfg.random_rotation_deg > 0:
        t.append(transforms.RandomRotation(cfg.random_rotation_deg))
    
    if cfg.color_jitter:
        t.append(transforms.ColorJitter(0.2, 0.2, 0.2, 0.02))
        #  brightness, contrast, saturation, hue

    if cfg.gaussian_blur:
        t.append(transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)))
        # sigma chosen randomly between 0.2 and 2.0

    t.append(transforms.ToTensor())
    # Converts a PIL image (H×W×C, 0–255) into a PyTorch tensor (C×H×W, 0–1 float).

    t.append(transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                  std=(0.229, 0.224, 0.225)))
    #  mean/std values are ImageNet statistics

    if cfg.random_erasing:
        t.append(transforms.RandomErasing(p=0.2, scale=(0.02, 0.10), ratio=(0.3, 3.3), value='random'))
        # randomly cuts out a rectangle region in the tensor; pixel values = 0 or random values
        # apply to 25% of images (p)
        # scale signifies the % of the area erased

    return transforms.Compose(t)
    # Compose returns a callable object that iterates the image over all the added transforms 


def build_eval_tf(image_size: int, cfg: EvalCfg):
    return transforms.Compose([
        transforms.Resize(cfg.resize_short),
        transforms.CenterCrop(cfg.center_crop if cfg.center_crop else image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225)),
    ])