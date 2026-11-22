import cv2
import numpy as np
from pathlib import Path
import json

ASSET_DIR = Path("assets")
IMAGE_SIZE = (1024, 768)
MIN_CLIPS = 0
MAX_CLIPS = 10
MAX_ATTEMPTS = 1000

def generate_dataset(n_images: int, seed: int, out_dir: Path) -> None:
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


def generate_image(backgrounds: np.ndarray, clips: np.ndarray, rng) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    clip_count = rng.integers(MIN_CLIPS, MAX_CLIPS+1)
    canvas = random_background(backgrounds, rng)
    boxes = []

    attempts = 0
    while len(boxes) < clip_count and attempts < MAX_ATTEMPTS:
        attempts += 1
        clip = random_clip(clips, rng)
        x, y = random_position(canvas, clip, rng)
        h, w = clip.shape[:2]
        box = make_box(x, y, w, h)

        overlaps = False
        for b in boxes:
            if boxes_overlap(box, b):
                overlaps = True
                break

        if not overlaps:
            box = place_clip(canvas, clip, x, y)
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
        cropped_clip = crop_to_content(clip)
        clips[path.stem.split("_")[0]] = cropped_clip

    return clips


def crop_to_content(img: np.ndarray) -> np.ndarray:
    alpha = img[:, :, 3]
    x, y, w, h = cv2.boundingRect(alpha)
    return img[y:y+h, x:x+w]


def scale_clip(img: np.ndarray, length: int) -> np.ndarray:
    s = length / img.shape[0]
    return cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)


def make_box(x, y, w, h) -> dict:
    return {"top": y, "bottom": y+h, "left": x, "right": x+w}


def place_clip(background: np.ndarray, clip: np.ndarray, x: int, y: int) -> dict:
    h, w = clip.shape[:2]
    roi = background[y:y+h, x:x+w]
    alpha = clip[:, :, 3:4] / 255.0
    colors = clip[:, :, :3]
    background[y:y+h, x:x+w] = (alpha * colors + (1 - alpha) * roi).astype(np.uint8)
    return make_box(x, y, w, h)


def rotate_clip(clip: np.ndarray, angle: float) -> np.ndarray:
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


def random_clip(clips: dict, rng) -> np.ndarray:
    color = rng.choice(list(clips.keys()))
    clip = clips[color]
    clip_s = scale_clip(clip, rng.integers(150, 201))
    return rotate_clip(clip_s, rng.uniform(0, 360))


def random_position(background: np.ndarray, clip: np.ndarray, rng) -> tuple[int, int]:
    H, W = background.shape[:2]
    h, w = clip.shape[:2]
    x = rng.integers(0, W-w+1)
    y = rng.integers(0, H-h+1)
    return (int(x), int(y))


def random_background(backgrounds: np.ndarray, rng) -> np.ndarray:
    return rng.choice(backgrounds)


def boxes_overlap(a: dict, b: dict) -> bool:
    return a["left"] < b["right"] and b["left"] < a["right"] and a["top"] < b["bottom"] and b["top"] < a["bottom"]
    
