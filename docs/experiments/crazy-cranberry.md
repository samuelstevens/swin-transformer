# Crazy Cranberry

Crazy cranberry is using SwinV2 with ImageNet1K pre-training on fishnet for instance segmentation.

Unfortunately I cannot get wandb to work with mmdet so for now I am just going to parse the log files using the tools provided.

```yaml
configs:
- object-detection/configs/hierarchical-vision-project/crazy_cranberry_fishnet.py
codename: crazy-cranberry
```
