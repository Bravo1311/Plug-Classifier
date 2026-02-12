from dataclasses import dataclass
from typing import List, Tuple
import torch
from PIL import Image
from torchvision import transforms

@dataclass
class PredictResult:
    label: str
    prob: float
    topk: List[Tuple[str, float]]

def load_checkpoint(ckpt_path: str):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    return ckpt["model_state"], ckpt["classes"]

@torch.no_grad()
def predict_image(model, image_path: str, classes: List[str], device: str, tfm, topk: int = 5) -> PredictResult:
    model.eval().to(device)
    img = Image.open(image_path).convert("RGB")
    x = tfm(img).unsqueeze(0).to(device)
    logits = model(x)
    probs = torch.softmax(logits, dim=1).squeeze(0)

    k = min(topk, len(classes))
    vals, idx = torch.topk(probs, k=k)
    top = [(classes[i], float(v)) for v, i in zip(vals.cpu(), idx.cpu())]

    best_label, best_prob = top[0]
    return PredictResult(label=best_label, prob=best_prob, topk=top)
