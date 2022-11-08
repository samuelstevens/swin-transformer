import argparse

from . import coco, helpers


def normalize_cli(args):
    mean, std = helpers.load_statistics(args.directory)

    print("Add this to a datasets config .py file:")
    print(
        f"""
img_norm_cfg = dict(
    mean={(mean * args.scale).tolist()},
    std={(std * args.scale).tolist()},
    to_rgb=True,
)"""
    )


def cocofy_cli(args):
    coco.cocofy(args.images, args.labels, args.output, args.size)


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(help="Available commands.")

    # Cocofy
    cocofy_parser = subparsers.add_parser(
        "cocofy", help="Convert fishnet to COCO format."
    )
    cocofy_parser.add_argument("images", help="Images folder")
    cocofy_parser.add_argument("labels", help="Labels file")
    cocofy_parser.add_argument("output", help="Output folder")
    cocofy_parser.add_argument(
        "--size",
        help="How much of the training data to include. Integers imply a COUNT while floats imply a FRACTION.",
        default=None,
        type=float,
    )
    cocofy_parser.set_defaults(func=cocofy_cli)

    # Normalize
    normalize_parser = subparsers.add_parser(
        "normalize", help="Measure mean and std of dataset."
    )
    normalize_parser.add_argument("directory", help="Data folder")
    normalize_parser.add_argument(
        "--scale", help="What to multiply values by.", default=1
    )
    normalize_parser.set_defaults(func=normalize_cli)

    args = parser.parse_args()

    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_usage()


if __name__ == "__main__":
    main()
