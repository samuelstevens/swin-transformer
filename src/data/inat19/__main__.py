"""
Script to turn inat19 data into a train/val set.
"""

import argparse

from . import organize


def organize_cli(args):
    organize.organize(
        args.images,
        args.train_annotations,
        args.val_annotations,
        args.category_annotations,
        args.output,
    )


def inat21_cli(args):
    organize.inat21(args.inat19, args.inat21, args.output)


def main():
    parser = argparse.ArgumentParser(prog="src.data.inat19")
    subparsers = parser.add_subparsers(help="Available commands.")

    # Organize
    organize_parser = subparsers.add_parser(
        "organize", help="Move images around to match a nice tree structure."
    )
    organize_parser.add_argument("images", help="Images folder")
    organize_parser.add_argument("train_annotations", help="Train annotations file")
    organize_parser.add_argument("val_annotations", help="Validation annotations file")
    organize_parser.add_argument(
        "category_annotations", help="Categories annotations file"
    )
    organize_parser.add_argument("output", help="Output folder")
    organize_parser.set_defaults(func=organize_cli)

    # Get iNat21 subset
    inat21_parser = subparsers.add_parser(
        "inat21",
        help="Get the validation images that belong to classes seen in iNat21.",
    )
    inat21_parser.add_argument("inat19", help="Path to iNat19 validation folder")
    inat21_parser.add_argument("inat21", help="Path to iNat21 validation folder")
    inat21_parser.add_argument("output", help="Where to put the subset of images.")
    inat21_parser.set_defaults(func=inat21_cli)

    args = parser.parse_args()

    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_usage()


if __name__ == "__main__":
    main()
