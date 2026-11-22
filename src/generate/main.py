"""CLI entry point for generating the paper clip dataset.

Usage:
    python -m src.generate.main --n-images 100 --seed 42 --out-dir data
"""

import argparse
from pathlib import Path

from src.generate.generate import generate_dataset


def main() -> None:
    """Parse CLI arguments and generate the dataset."""
    args = parse_args()
    generate_dataset(n_images=args.n_images, seed=args.seed, out_dir=args.out_dir)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for dataset generation."""
    parser = argparse.ArgumentParser(
        description="Generate synthetic paper clip images with labels.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--n-images",
        type=int,
        default=5,
        help="number of images to generate",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="generation seed",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data"),
        help="output directory for images and labels",
    )

    return parser.parse_args()


if __name__ == "__main__":
    main()
