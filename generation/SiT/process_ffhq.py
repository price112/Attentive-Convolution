#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, os, torch, PIL.Image, PIL.ImageOps, shutil, pathlib, threading
from pathlib import Path
from shutil import copystat
from tqdm.contrib.concurrent import thread_map          # 一行搞定线程池
from torchvision.transforms import Resize, InterpolationMode
from functools import partial

# ------------------- 与原版完全一致的工具函数 -------------------
def save_same_format(img: PIL.Image.Image, dst_path: Path):
    ext = dst_path.suffix.lower()
    if img.mode != "RGB":
        img = img.convert("RGB")
    if ext in (".jpg", ".jpeg"):
        img.save(dst_path, format="JPEG", quality=100, subsampling=0, optimize=False)
    elif ext == ".png":
        img.save(dst_path, format="PNG", compress_level=0, optimize=False)
    elif ext == ".webp":
        img.save(dst_path, format="WEBP", lossless=True, quality=100)
    elif ext in (".bmp",):
        img.save(dst_path, format="BMP")
    elif ext in (".tif", ".tiff"):
        img.save(dst_path, format="TIFF", compression="raw")
    else:
        img.save(dst_path)

# ------------------- 单张图处理（可被并发调用） -------------------
def work(sp: Path, *, src_root: Path, dst_root: Path, size: int, resize_cuda):
    rp = sp.relative_to(src_root)
    dp = dst_root / rp
    dp.parent.mkdir(parents=True, exist_ok=True)

    im = PIL.Image.open(sp)
    im = PIL.ImageOps.exif_transpose(im)

    tensor = torch.from_numpy(
        (torch.ByteTensor(torch.ByteStorage.from_buffer(im.tobytes()))
         .view(im.height, im.width, len(im.getbands()))
         .numpy())
    ).permute(2, 0, 1).unsqueeze(0).float()          # [1,C,H,W]
    if torch.cuda.is_available():
        tensor = tensor.cuda(non_blocking=True)
    resized = resize_cuda(tensor).clamp(0, 255).byte().squeeze(0).permute(1, 2, 0).cpu().numpy()
    im_resized = PIL.Image.fromarray(resized, mode="RGB")

    save_same_format(im_resized, dp)
    try:
        copystat(sp, dp)
    except Exception:
        pass

# ------------------- 主入口 -------------------
def main():
    ap = argparse.ArgumentParser(description="Fast high-fidelity resize (multi-thread + CUDA)")
    ap.add_argument("--src", required=True, )
    ap.add_argument("--dst", required=True, )
    ap.add_argument("--size", type=int, default=512, )
    ap.add_argument("-j", "--jobs", type=int, default=min(16, (os.cpu_count() or 1) + 4),)
    args = ap.parse_args()

    src, dst = Path(args.src).resolve(), Path(args.dst).resolve()
    dst.mkdir(parents=True, exist_ok=True)

    exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}
    files = sorted([p for p in src.rglob("*") if p.is_file() and p.suffix.lower() in exts])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    resize_cuda = Resize((args.size, args.size),
                         interpolation=InterpolationMode.BICUBIC,
                         antialias=True).to(device)

    worker = partial(work,
                        src_root=src,
                        dst_root=dst,
                        size=args.size,
                        resize_cuda=resize_cuda)

    thread_map(worker, files,
               max_workers=args.jobs,
               chunksize=1,
               desc="Resizing")

    out_files = sorted([p.relative_to(dst) for p in dst.rglob("*") if p.is_file() and p.suffix.lower() in exts])
    in_files = [p.relative_to(src) for p in files]
    assert len(out_files) == len(in_files)
    assert out_files == sorted(in_files)

    print(f"Finished：{len(out_files)} saved to {dst}")

if __name__ == "__main__":
    main()