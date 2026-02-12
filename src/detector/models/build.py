from dataclasses import dataclass
import torch.nn as nn
import torchvision.models as tvm

"""
    Contains the model factory function that: reads config, build correct architecture, adjusts final layer and returns nn.Module
"""

@dataclass
class ModelCfg:
    name: str
    pretrained: bool = True
    dropout: float = 0.0          # percentage of nerons to be dropped in a batch: [0, 1]

def build_model(cfg: ModelCfg, num_classes: int) -> nn.Module:
    name = cfg.name.lower()

    if name == "resnet18":
        m = tvm.resnet18(weights = tvm.ResNet18_Weights.DEFAULT if cfg.pretrained else None)
        in_f = m.fc.in_features
        m.fc = nn.Sequential(
            nn.Dropout(cfg.dropout),
            nn.Linear(in_f, num_classes)
        )
        return m
    
    if name == "resnet34":
        m = tvm.resnet34(weights=tvm.ResNet34_Weights.DEFAULT if cfg.pretrained else None)
        in_f = m.fc.in_features
        m.fc = nn.Sequential(nn.Dropout(cfg.dropout), nn.Linear(in_f, num_classes))
        return m

    if name == "mobilenet_v3_small":
        m = tvm.mobilenet_v3_small(
            weights=tvm.MobileNet_V3_Small_Weights.DEFAULT if cfg.pretrained else None
        )
        in_f = m.classifier[-1].in_features
        m.classifier[-1] = nn.Linear(in_f, num_classes)
        return m

    raise ValueError(f"Unknown model.name={cfg.name}")