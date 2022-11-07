# Groovy Grape

This experiment trains swin-v2-base from scratch on iNat21 using the multitask objective using 1/4 of the default learning rate, which works out to 1.25e-4.
It also does 90 epochs at 192.

```yaml
configs:
- configs/hierarchical-vision-project/groovy-grape-192.yaml
- object-detection/configs/hierarchical-vision-project/groovy_grape_fishnet.py
codename: groovy-grape
```

## Log

This model trained the first 90 on 8x V100, and did 8/30 epochs at 256 on 8x V100. 
I am storing the 8th checkpoint on S3 as groovy-grape-256-epoch8.pth.
Now it is stored at /local/scratch/stevens.994/hierarchical-vision/groovy-grape-256/v0
It was originally haunted-broomstick on wandb, but is now groovy-grape-256.

I messed it up by switching std/mean, so I am restarting this run on 8xV100.

After pre-training finished at 192x192, I am fine-tuning it for object detection on 8xV100 (imageomics-aws-east-1-0) where it is due to finish 12 epochs at 8PM on Monday (Nov 7th).
