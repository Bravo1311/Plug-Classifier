from dataclasses import dataclass
from typing import Optional, Sequence
from torch.utils.data import Dataset, Subset
from torchvision.datasets import ImageFolder

"""
    Imagefolder: PyTroch dataset that defines the image labels as folder names for Image classification tasks
    creates attributes: classes and class_to_idx
"""

@dataclass
class DatasetInfo:
    classes: Sequence[str]
    class_to_idx: dict

class FolderDataset(Dataset):
    """
    Wraps torchvision ImageFolder + split subsets.
    """
    def __init__(self, root_dir: str):
        self.ds = ImageFolder(root=root_dir)
        self.info = DatasetInfo(classes=self.ds.classes, class_to_idx=self.ds.class_to_idx)
        self.transform = None

    def set_transform(self, transform):
        self.transform = transform

    def subset(self, indices):
        return Subset(self, indices)
    
    def __len__(self):
        return len(self.ds)    

    def __getitem__(self, idx):
        img, label = self.ds[idx]
        if self.transform:
            img = self.transform(img)
        return img, label