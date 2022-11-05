import datetime
import os.path


def check(prev, current):
    diff = current - prev
    assert diff > 0
    delta = datetime.timedelta(seconds=diff)
    print(delta)


def main():
    path = "/home/ubuntu/projects/swin-transformer/runs/swinv2_base_patch4_window12_192_inat21_hierarchical_lr2.5/v0"
    prev_time = None
    for epoch in range(90):
        mtime = os.path.getmtime(f"{path}/ckpt_epoch_{epoch}.pth")
        if prev_time is not None:
            check(prev_time, mtime)
        prev_time = mtime


if __name__ == "__main__":
    main()
