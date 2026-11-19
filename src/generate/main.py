import argparse
from pathlib import Path
from src.generate.generate import generate_dataset

def main() -> None:
    args = parse_args()
    generate_dataset(n_images=args.n_images, seed=args.seed, out_dir=args.out_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="generates images with paper clips")
    
    parser.add_argument("--n-images", type=int, default=5, help="number of images to generate")
    parser.add_argument("--seed", type=int, default=42, help="generation seed")
    parser.add_argument("--out-dir", type=Path, default=Path("data"), help="output path for images")

    return parser.parse_args()


if __name__ == "__main__":
    main()
