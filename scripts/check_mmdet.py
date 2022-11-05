from mmdet.apis import inference_detector, init_detector

config_file = "/mnt/10tb/models/yolov3/yolov3_mobilenetv2_320_300e_coco.py"
checkpoint_file = "/mnt/10tb/models/yolov3/yolov3_mobilenetv2_320_300e_coco_20210719_215349-d18dff72.pth"
model = init_detector(config_file, checkpoint_file, device="cpu")  # or device='cuda:0'
print(
    inference_detector(
        model,
        "/mnt/10tb/data/fishnet/raw/images/fcba762e-5952-11ec-ad1c-2b9c06dc107d.jpg",
    )
)
