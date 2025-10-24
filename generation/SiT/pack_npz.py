import argparse, re
from pathlib import Path
from PIL import Image
import numpy as np
from tqdm import tqdm

def to_rgb(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    elif arr.ndim == 3 and arr.shape[2] == 4:
        arr = arr[:, :, :3]
    return arr

def natural_sort_key(path: Path):

    m = re.search(r"(\d+)", path.stem)
    return int(m.group(1)) if m else path.stem

def create_npz_from_folder(sample_dir: str, out_path: str = None, key: str = "arr_0"):
    folder = Path(sample_dir)

    imgs = sorted(
        [p for p in folder.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}],
        key=natural_sort_key,
    )
    if not imgs:
        raise FileNotFoundError(f"No images found in {sample_dir}")


    arr0 = to_rgb(np.asarray(Image.open(imgs[0])).astype(np.uint8))
    H, W = arr0.shape[:2]
    samples = np.empty((len(imgs), H, W, 3), dtype=np.uint8)
    samples[0] = arr0

    for i, p in enumerate(tqdm(imgs[1:], desc=f"Packing {folder.name} -> npz", initial=1)):
        arr = to_rgb(np.asarray(Image.open(p)).astype(np.uint8))
        if arr.shape[:2] != (H, W):
            raise ValueError(f"Image {p} size {arr.shape[:2]} != ({H}, {W})")
        samples[i+1] = arr

    out_path = out_path or f"{folder}.npz"
    np.savez(out_path, **{key: samples})
    print(f"Saved {out_path}, shape={samples.shape}, key='{key}'")
    return out_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-dir", type=str, default="samples")
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--key", type=str, default="arr_0")
    args = parser.parse_args()
    create_npz_from_folder(args.sample_dir, args.out, args.key)
    print("Done.")
