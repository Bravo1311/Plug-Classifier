import argparse
from pathlib import Path
import yaml
from PIL import Image
import torch

from detector.data.transforms import build_eval_tf, EvalCfg
from detector.models.build import build_model, ModelCfg
from .seed import seed_everything
from .logging_utils import log_info
from .data.dataset import FolderDataset
from .data.splits import SplitSpec, make_indices
from .data.transforms import build_train_tf, build_eval_tf, AugmentCfg, EvalCfg
from .models.build import build_model, ModelCfg
from .train.trainer import train_classifier, TrainCfg

def load_yaml(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def cmd_train(args):
    dataset_cfg = load_yaml(args.dataset)
    aug_cfg = load_yaml(args.augment)
    model_cfg = load_yaml(args.model)
    train_cfg = load_yaml(args.train)

    seed_everything(train_cfg["train"]["seed"])

    root_dir = dataset_cfg["dataset"]["root_dir"]
    img_size = dataset_cfg["dataset"]["image_size"]
    num_workers = dataset_cfg["dataset"]["num_workers"]

    train_tf = build_train_tf(img_size, AugmentCfg(**aug_cfg["augment"]["train"]))
    eval_tf = build_eval_tf(img_size, EvalCfg(**aug_cfg["augment"]["eval"]))

    full_train = FolderDataset(root_dir=root_dir)
    full_train.set_transform(train_tf)

    full_eval = FolderDataset(root_dir=root_dir)
    full_eval.set_transform(eval_tf)


    n = len(full_train)
    split = dataset_cfg["splits"]
    spec = SplitSpec(train=split["train"], val=split["val"], test=split["test"], seed=split["seed"])
    train_idx, val_idx, _ = make_indices(n, spec)

    train_ds = full_train.subset(train_idx)
    val_ds   = full_eval.subset(val_idx)

    classes = full_train.info.classes
    log_info(f"Found {len(classes)} classes: {classes}")
    log_info(f"Samples: total={n}, train={len(train_ds)}, val={len(val_ds)}")

    mcfg = ModelCfg(**model_cfg["model"])
    model = build_model(mcfg, num_classes=len(classes))

    tcfg = TrainCfg(**train_cfg["train"])
    out_dir = "runs"
    result = train_classifier(
        model=model,
        train_ds=train_ds,
        val_ds=val_ds,
        cfg=tcfg,
        out_dir=out_dir,
        class_names=classes,
    )
    log_info(f"Done. Best checkpoint: {result['best_checkpoint']} (best val loss {result['best_val_loss']:.4f})")

def cmd_infer(args):
    dataset_cfg = load_yaml(args.dataset)
    aug_cfg = load_yaml(args.augment)
    model_cfg = load_yaml(args.model)

    img_size = dataset_cfg["dataset"]["image_size"]

    # Build eval transform
    eval_tf = build_eval_tf(img_size, EvalCfg(**aug_cfg["augment"]["eval"]))

    # Load checkpoint
    ckpt = torch.load(args.ckpt, map_location="cpu")
    classes = ckpt["classes"]

    # Build model with correct head size, load weights
    mcfg = ModelCfg(**model_cfg["model"])
    model = build_model(mcfg, num_classes=len(classes))
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # Device
    device = "cuda" if (args.device == "auto" and torch.cuda.is_available()) else (args.device if args.device != "auto" else "cpu")
    model.to(device)

    # Collect images
    in_path = Path(args.input)
    if in_path.is_dir():
        exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
        paths = sorted([p for p in in_path.rglob("*") if p.suffix.lower() in exts])
    else:
        paths = [in_path]

    if not paths:
        raise SystemExit(f"No images found at: {in_path}")

    topk = args.topk

    for p in paths:
        img = Image.open(p).convert("RGB")
        x = eval_tf(img).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=1).squeeze(0)

        k = min(topk, len(classes))
        vals, idx = torch.topk(probs, k=k)
        preds = [(classes[i], float(v)) for v, i in zip(vals.cpu(), idx.cpu())]

        best_label, best_prob = preds[0]
        print(f"{p} -> {best_label} ({best_prob:.3f}) | top{len(preds)}={preds}")


def main():
    p = argparse.ArgumentParser("port-detector-pytorch")
    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train")
    t.add_argument("--dataset", default="configs/dataset.yaml")
    t.add_argument("--augment", default="configs/augment.yaml")
    t.add_argument("--model", default="configs/model.yaml")
    t.add_argument("--train", default="configs/train.yaml")
    t.set_defaults(func=cmd_train)

    i = sub.add_parser("infer")
    i.add_argument("--ckpt", required=True, help="Path to checkpoint, e.g. runs/exp_001/checkpoints/best.pt")
    i.add_argument("--input", required=True, help="Image path or folder path, e.g. data/inference/")
    i.add_argument("--topk", type=int, default=5)
    i.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    i.add_argument("--dataset", default="configs/dataset.yaml")
    i.add_argument("--augment", default="configs/augment.yaml")
    i.add_argument("--model", default="configs/model.yaml")
    i.set_defaults(func=cmd_infer)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
