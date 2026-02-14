# Plugs and Ports Identifier

## Overview
This repository contains a Pytorch-based image classification model that identifies different types of electrical ports and plugs (USB-A, USB-C, VGA, Ethernet, HDMI, etc.).

This project was initiated as preparation for [AI For Industry Challenge](https://www.intrinsic.ai/events/ai-for-industry-challenge) and follows production-style ML engineering practices with modular components, YAML configurations, CLI tooling and logging.

This project represents Phase 1 (semantic recognition) of a braoder robotics perception-to-manipulation pipeline.

## System Architecture
- Transfer learning with pretrained backbone: ResNet18
- Custom classification head
- optinal backbone freezing during early epochs
- Cross entropy loss
- AdamW optimizer
- On-the-fly augmentation

## Datasource
The dataset used for training was obtained from Kaggle: \
**Dataset** : [Plug Identifier Training Set](https://www.kaggle.com/datasets/pk1282/plugidentifier-extractedobjects-trainingset)
All data rights belong to the original dataset authors. This repository does not redistribute the dataset.

## Dataset Format
```
data/raw/
  usb_c/
  micro_usb/
  ethernet/
  hdmi/
  displayport/
  ...
```
Each folder represents a class label. 

## Training
```
python -m detector.cli train \
  --dataset configs/dataset.yaml \
  --augment configs/augment.yaml \
  --model configs/model.yaml \
  --train configs/train.yaml
```

# Outputs
- `runs/exp_xxx/checkpoints/best.pt`
- TensorBoard logs
- Per-epoch checkpoints\
### Launch TensorBoard: 
`tensorboard --logdir runs/exp_xxx/tb`

## Inference
### Single Image
```
python -m detector.cli infer \
  --ckpt runs/exp_001/checkpoints/best.pt \
  --input data/inference/test.jpg

```

### Folder Inference
```
python -m detector.cli infer \
  --ckpt runs/exp_001/checkpoints/best.pt \
  --input data/inference/

```

## Data Augmentation
Training uses strong on-the-fly augmentation:
- Random resized crop
- Random rotation
- Horizontal flip
- Color jitter
- Gaussian blur
- Random erasing
- Normalization\
This augmentation improves generatlization given limited dataset size.

## Motivation
In robotic manipulation tasks (cable handling here), the system must:
1. Identify connector type
2. Locate target port
3. Align pose
4. Execute Insertion <br/>

This classifier represents Phase 1: semantic recognition in a broader perception-to-action pipeline.

## Future Extensions
- [YOLO-based port & plug detection](https://github.com/Bravo1311/Plugs-Detector-Yolo)
- Real-time video inference (IP Webcam / OpenCV)
- Domain adaptation for cluttered scenes
- ROS2 integration
- Pose estimation for alignment
- Perception → manipulation pipeline
- VLA-based decision layer

## Tech Stack
- Python  3.10+
- PyTorch
- torchvision
- TensorBoard
- YAML configs
- Rich CLI logging
