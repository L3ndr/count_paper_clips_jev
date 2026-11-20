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
    result_img = backgrounds[0].copy()
    clip_s = scale_clip(clips["red"], 180)
    clip_sr = rotate_clip(clip_s, 45)
    box = place_clip(result_img, clip_sr, 400, 300)
    debug_img = result_img.copy()
    debug_img = cv2.rectangle(
        debug_img, 
        (box["left"], box["top"]), 
        (box["right"], box["bottom"]), 
        (0, 255, 0), 
        2
    )
    cv2.imwrite(Path(out_dir, "result.jpg"), result_img)
    cv2.imwrite(Path(out_dir, "debug.jpg"), debug_img)


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


def place_clip(background: np.ndarray, clip: np.ndarray, x: int, y: int) -> dict:
    h, w = clip.shape[:2]
    roi = background[y:y+h, x:x+w]
    alpha = clip[:, :, 3:4] / 255.0
    colors = clip[:, :, :3]
    background[y:y+h, x:x+w] = (alpha * colors + (1 - alpha) * roi).astype(np.uint8)
    return {"top": y, "bottom": y+h, "left": x, "right": x+w}


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
