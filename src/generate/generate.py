"""Generate synthetic paper clip images with count and bounding box labels."""

import json
from pathlib import Path

import cv2
import numpy as np

ASSET_DIR = Path("assets")
IMAGE_SIZE = (1024, 768)
MIN_CLIPS = 0
MAX_CLIPS = 10
MAX_ATTEMPTS = 1000


def generate_dataset(n_images: int, seed: int, out_dir: Path) -> None:
    """Generate a dataset of labeled images.

    Creates a dataset of images with random backgrounds and multiple random paper clips
    and stores the count and position in a JSON file. Additionally it stores debug
    images with bounding boxes for each paper clip.

    Args:
        n_images: Number of images to be generated.
        seed: Seed for reproducible results.
        out_dir: Output path for the images.
    """
    rng = np.random.default_rng(seed)

    out_dir.mkdir(parents=True, exist_ok=True)
    result_dir = out_dir / "result"
    debug_dir = out_dir / "debug"
    result_dir.mkdir(parents=True, exist_ok=True)
    debug_dir.mkdir(parents=True, exist_ok=True)

    backgrounds = load_backgrounds(Path(ASSET_DIR, "backgrounds"))
    clips = load_clips(Path(ASSET_DIR, "paper_clips"))
    labels = {"seed": seed, "images": []}

    for i in range(n_images):
        canvas, debug_img, boxes = generate_image(backgrounds, clips, rng)

        cv2.imwrite(Path(result_dir, f"result_{i}.jpg"), canvas)
        cv2.imwrite(Path(debug_dir, f"debug_{i}.jpg"), debug_img)

        labels["images"].append({
            "file": f"result_{i}.jpg",
            "debug_img": f"debug_{i}.jpg",
            "count": len(boxes),
            "boxes": boxes}
        )

    with open(out_dir / "labels.json", "w") as file:
        json.dump(labels, file, indent=2)


def generate_image(
        backgrounds: list[np.ndarray],
        clips: dict[str, np.ndarray],
        rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Compose one synthetic image with non-overlapping paper clips.

    Draws a random clip count and places clips at random positions until the count is
    reached or MAX_ATTEMPTS is exhausted, so fewer clips may be placed.

    Args:
        backgrounds: Preloaded background images, all of IMAGE_SIZE.
        clips: Cropped clip images with alpha channel, keyed by color.
        rng: Seeded generator used for all random choices.

    Returns:
        The composed image, a copy with bounding boxes drawn, and the box of each
        placed clip.
    """
    clip_count = rng.integers(MIN_CLIPS, MAX_CLIPS+1)
    canvas = random_background(backgrounds, rng)
    boxes = []

    attempts = 0
    while len(boxes) < clip_count and attempts < MAX_ATTEMPTS:
        attempts += 1
        color, clip = random_clip(clips, rng)
        x, y = random_position(canvas, clip, rng)
        h, w = clip.shape[:2]
        box = make_box(color, x, y, w, h)

        overlaps = False
        for b in boxes:
            if boxes_overlap(box, b):
                overlaps = True
                break

        if not overlaps:
            box = place_clip(canvas, clip, color, x, y)
            boxes.append(box)

    debug_img = canvas.copy()
    for box in boxes:
        debug_img = cv2.rectangle(
            debug_img,
            (box["left"], box["top"]),
            (box["right"], box["bottom"]),
            (0, 255, 0),
            2
        )

    return canvas, debug_img, boxes


def load_backgrounds(directory: Path) -> list[np.ndarray]:
    """Load background files and resize them.

    Args:
        directory: Path to backgrounds.

    Returns:
        List of backgrounds stored as numpy arrays.

    Raises:
        FileNotFoundError: If no files are found or a file cannot be read.
    """
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
    """Load paper clip files and crop them.

    Args:
        directory: Path to paper clips.

    Returns:
        Dictionary with each paper clip color (extracted from file name)
        and the paper clips stored as numpy arrays.

    Raises:
        FileNotFoundError: If no files are found or a file cannot be read.
    """
    clips_dir = sorted(directory.glob("*.png"))
    if len(clips_dir) == 0:
        raise FileNotFoundError(f"Could not find any paper clips at: {directory}")

    clips = {}
    for path in clips_dir:
        clip = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if clip is None:
            raise FileNotFoundError(f"Could not read {path}")
        cropped_clip = crop_to_content(clip)
        clips[path.stem.split("_")[0]] = cropped_clip

    return clips


def random_clip(
        clips: dict[str, np.ndarray],
        rng: np.random.Generator
    ) -> tuple[str, np.ndarray]:
    """Choose, scale and rotate a random paper clip."""
    color = str(rng.choice(list(clips.keys())))
    clip = clips[color]
    clip_s = scale_clip(clip, rng.integers(150, 201))
    return (color, rotate_clip(clip_s, rng.uniform(0, 360)))


def random_position(
        background: np.ndarray,
        clip: np.ndarray,
        rng: np.random.Generator
    ) -> tuple[int, int]:
    """Find a random position on the background to place the clip."""
    H, W = background.shape[:2]
    h, w = clip.shape[:2]
    x = rng.integers(0, W-w+1)
    y = rng.integers(0, H-h+1)
    return (int(x), int(y))


def random_background(
        backgrounds: list[np.ndarray],
        rng: np.random.Generator
    ) -> np.ndarray:
    """Choose a random background and return a copy of it."""
    return backgrounds[rng.integers(len(backgrounds))].copy()


def crop_to_content(img: np.ndarray) -> np.ndarray:
    """Crop image with alpha channel to the true content size."""
    alpha = img[:, :, 3]
    x, y, w, h = cv2.boundingRect(alpha)
    return img[y:y+h, x:x+w]


def scale_clip(img: np.ndarray, length: int) -> np.ndarray:
    """Scale the paper clip.

    Args:
        img: The image to be scaled.
        length: Target height in pixels; width scales proportionally.

    Returns:
        The scaled version of the image.
    """
    s = length / img.shape[0]
    return cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)


def rotate_clip(clip: np.ndarray, angle: float) -> np.ndarray:
    """Rotate paper clip by a given angle.

    The angle is applied counterclockwise and the dimensions can increase to prevent
    cut offs.
    """
    h, w = clip.shape[:2]
    center = (w/2, h/2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    cos = abs(matrix[0, 0])
    sin = abs(matrix[0, 1])
    new_w = int(h * sin + w * cos)
    new_h = int(h * cos + w * sin)
    matrix[0, 2] += new_w / 2 - center[0]
    matrix[1, 2] += new_h / 2 - center[1]
    clip_r = cv2.warpAffine(
        clip,
        matrix,
        (new_w, new_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0)
    )
    return crop_to_content(clip_r)


def place_clip(background: np.ndarray, clip: np.ndarray, color: str, x: int, y: int) -> dict:
    """Place paper clip on the background in-place.

    Args:
        background: The background.
        clip: The paper clip to be placed.
        x: X-position (upper left corner).
        y: Y-position (upper left corner).

    Returns:
        Dictionary with the bounding box of the paper clip.
    """
    h, w = clip.shape[:2]
    roi = background[y:y+h, x:x+w]
    alpha = clip[:, :, 3:4] / 255.0
    colors = clip[:, :, :3]
    background[y:y+h, x:x+w] = (alpha * colors + (1 - alpha) * roi).astype(np.uint8)
    return make_box(color, x, y, w, h)


def make_box(color: str, x: int, y: int, w: int, h: int) -> dict:
    """Return a bounding box dict from top-left corner and size."""
    return {"color": color, "top": y, "bottom": y+h, "left": x, "right": x+w}


def boxes_overlap(a: dict, b: dict) -> bool:
    """Determine whether two boxes a and b overlap with each other."""
    return a["left"] < b["right"] and b["left"] < a["right"] and a["top"] < b["bottom"] and b["top"] < a["bottom"]