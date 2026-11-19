import cv2
import numpy as np
from pathlib import Path

ASSET_DIR = Path("assets")
IMAGE_SIZE = (1024, 768)

def generate_dataset(n_images: int, seed: int, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    backgrounds = load_backgrounds(Path(ASSET_DIR, "backgrounds"))
    clips = load_clips(Path(ASSET_DIR, "paper_clips"))

    # test
    print(backgrounds[0].shape)
    print(clips["red"].shape)
    cv2.imwrite(str(out_dir / "test_bg.jpg"), backgrounds[0])


def load_backgrounds(directory: Path) -> list[np.ndarray]:
    background_dir = sorted(directory.glob("*.jpg"))
    if len(background_dir) == 0:
        raise FileNotFoundError(f"Could not find any backgrounds at: {directory}")

    backgrounds = []
    for path in background_dir:
        background = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if background is None:
            raise FileNotFoundError(f"Could not read {path}")
        background = cv2.resize(background, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
        backgrounds.append(background)

    return backgrounds


def load_clips(directory: Path) -> dict[str, np.ndarray]:
    clips_dir = sorted(directory.glob("*.png"))
    if len(clips_dir) == 0:
        raise FileNotFoundError(f"Could not find any paper clips at: {directory}")
    
    clips = {}
    for path in clips_dir:
        clip = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if clip is None:
            raise FileNotFoundError(f"Could not read {path}")
        clips[path.stem.split("_")[0]] = clip

    return clips