import argparse
import os
import random
import sys
from typing import Dict, List, Tuple, Callable, Optional

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

mpl.rcParams.update({
    "figure.dpi": 800,
    "savefig.dpi": 400,
    "axes.linewidth": 1.0,
    "axes.labelsize": 16,
    "axes.titlesize": 20,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "legend.fontsize": 18,
    "font.family": "DejaVu Sans",
})

sys.path.append("/mnt/data")
from models import AttNet_T4, ATConv


def set_seed(seed: int = 0):
    torch.manual_seed(seed);
    np.random.seed(seed);
    random.seed(seed)


def get_device(device_str: str):
    if device_str == "cuda" and not torch.cuda.is_available():
        print("[warn] CUDA unavailable, fallback to CPU.")
        return torch.device("cpu")
    return torch.device(device_str)


def sync(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize()


# ---- single image loader ----
def load_single_image(img_path: str, size: int = 224) -> torch.Tensor:
    if not os.path.isfile(img_path):
        raise FileNotFoundError(f"No such file: {img_path}")
    tfm = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    img = Image.open(img_path).convert("RGB")
    x = tfm(img).unsqueeze(0)  # (1,3,H,W)
    return x


# ---- random images fallback ----
def gaussian_kernel2d(ks: int, sigma: float, device="cpu") -> torch.Tensor:
    c = ks // 2
    y, x = torch.meshgrid(torch.arange(-c, c + 1, device=device),
                          torch.arange(-c, c + 1, device=device), indexing='ij')
    k = torch.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
    k = k / k.sum()
    return k[None, None, ...]


def make_random_images(num_images: int, size: int = 224, seed: Optional[int] = None) -> torch.Tensor:
    if seed is not None: set_seed(seed)
    imgs = torch.randn(num_images, 3, size, size) * 0.6
    k = gaussian_kernel2d(7, 1.2, device="cpu")
    blur = F.conv2d(imgs, k.expand(3, 1, 7, 7), padding=3, groups=3)
    sharp = imgs + (imgs - blur) * 0.6
    x = 0.6 * imgs + 0.4 * sharp
    # normalize per-image then map to ImageNet stats
    x = (x - x.mean(dim=[2, 3], keepdim=True)) / (x.std(dim=[2, 3], keepdim=True) + 1e-6)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    x = x * std + mean
    return x


# ---------------- KD toggle (monkey-patch) ----------------
class KDGuard:
    """
    Temporarily toggle AttConv kernel difference on/off WITHOUT touching parameters.
    enabled=False -> patch _apply_kernel_difference to identity
    """

    def __init__(self, model: nn.Module, enabled: bool):
        self.model = model
        self.enabled = enabled
        self._stash: List[Tuple[ATConv, Callable]] = []

    def __enter__(self):
        if self.enabled:
            return self

        def identity_kdiff(kernels):  # same signature
            return kernels

        for m in self.model.modules():
            if isinstance(m, ATConv):
                self._stash.append((m, m._apply_kernel_difference))
                m._apply_kernel_difference = identity_kdiff  # type: ignore[attr-defined]
        return self

    def __exit__(self, exc_type, exc, tb):
        for m, fn in self._stash:
            m._apply_kernel_difference = fn  # type: ignore[attr-defined]
        self._stash.clear()
        return False


# ---------------- Enumerate AttConvs in FORWARD order ----------------
def enumerate_attconvs_forward(model: nn.Module) -> List[Tuple[str, ATConv]]:
    convs = []
    for si, stage in enumerate(model.stages):
        for bi, block in enumerate(stage):
            if hasattr(block, "token_mixer") and isinstance(block.token_mixer, ATConv):
                name = f"S{si}_B{bi}"  # 人类可读名
                convs.append((name, block.token_mixer))
    return convs


# ---------------- Hook registration (stable indexing) ----------------
def register_atconv_outputs_forward(model: nn.Module):
    feats: Dict[int, List[torch.Tensor]] = {}
    handles = []
    convs = enumerate_attconvs_forward(model)
    for idx, (tag, m) in enumerate(convs):
        def _hook(i):
            def fn(mod, inp, out):
                feats.setdefault(i, []).append(out.detach())

            return fn

        h = m.register_forward_hook(_hook(idx))
        handles.append(h)
    return feats, handles, convs  # feats keyed by idx (0..K-1), convs = [(tag, module), ...]


def remove_handles(handles):
    for h in handles:
        h.remove()


# ---------------- Metrics ----------------
def center_surround_contrast(x: torch.Tensor, sigma: float = 1.2) -> torch.Tensor:
    """CSC: mean(|x - GaussianBlur(x)|) / mean(|x|), per-sample."""
    B, C, H, W = x.shape
    ks = int(max(3, round(6 * sigma))) | 1
    device = x.device
    # build kernel
    c = ks // 2
    yy, xx = torch.meshgrid(torch.arange(-c, c + 1, device=device),
                            torch.arange(-c, c + 1, device=device), indexing='ij')
    k = torch.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2));
    k = k / k.sum()
    k = k[None, None, ...]
    x_blur = F.conv2d(x, k.expand(C, 1, ks, ks), padding=ks // 2, groups=C)
    diff = (x - x_blur).abs().mean(dim=[1, 2, 3])
    base = x.abs().mean(dim=[1, 2, 3]) + 1e-6
    return diff / base  # (B,)


def channel_decorrelation(x: torch.Tensor, max_channels: int = 256) -> torch.Tensor:
    """1 - mean absolute correlation across channels (flatten spatial), per-sample."""
    B, C, H, W = x.shape
    c = min(C, max_channels)
    if c < C: x = x[:, :c]
    X = x.view(B, c, -1)
    X = X - X.mean(dim=2, keepdim=True)
    X = X / (X.std(dim=2, keepdim=True) + 1e-6)
    N = X.size(2)
    corr = torch.bmm(X, X.transpose(1, 2)) / (N - 1 + 1e-6)  # (B,c,c)
    eye = torch.eye(c, device=x.device).unsqueeze(0)
    off = corr - eye
    mean_abs_off = off.abs().sum(dim=[1, 2]) / (c * c - c + 1e-6)
    return 1.0 - mean_abs_off  # (B,)


def channel_effective_rank(x: torch.Tensor, max_channels: int = 256) -> torch.Tensor:
    """Effective Rank via entropy of eigenvalue spectrum of channel covariance, normalized by c."""
    B, C, H, W = x.shape
    c = min(C, max_channels)
    if c < C: x = x[:, :c]
    X = x.view(B, c, -1)  # (B,c,N)
    X = X - X.mean(dim=2, keepdim=True)
    cov = torch.bmm(X, X.transpose(1, 2)) / (X.size(2) - 1 + 1e-6)  # (B,c,c)
    evals = torch.linalg.eigvalsh(cov).clamp_min(1e-12)  # (B,c)
    ps = evals / (evals.sum(dim=1, keepdim=True) + 1e-12)
    ent = -(ps * (ps + 1e-12).log()).sum(dim=1)
    erank = torch.exp(ent) / float(c)  # [0,1]
    return erank  # (B,)


METRIC_FNS: Dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
    "CSC (Center–Surround Contrast)": center_surround_contrast,
    "Channel Decorrelation": channel_decorrelation,
    "Channel Effective Rank": channel_effective_rank,
}


@torch.inference_mode()
def collect_values_for_metric(feats: Dict[int, List[torch.Tensor]],
                              metric_fn: Callable[[torch.Tensor], torch.Tensor]) -> Dict[int, np.ndarray]:
    out: Dict[int, np.ndarray] = {}
    for idx, lst in feats.items():
        x = torch.cat(lst, dim=0)  # (N,C,H,W)
        vals = metric_fn(x).detach().cpu().numpy()
        out[idx] = vals
    return out


# ---------------- Panel (A) plotting ONLY ----------------
def _mean_only(arr: np.ndarray) -> float:
    a = np.asarray(arr, dtype=float)
    a = a[np.isfinite(a)]
    return float(a.mean()) if a.size else 0.0


def plot_panelA_only(metric_name: str,
                     vals_off: Dict[int, np.ndarray],
                     vals_on: Dict[int, np.ndarray],
                     idx2tag: Dict[int, str],
                     save_path: str):
    layers = sorted(set(vals_off.keys()).intersection(vals_on.keys()))
    L = len(layers)
    idxs = np.array(layers, dtype=int)

    means_off = np.array([_mean_only(vals_off[i]) for i in layers])
    means_on = np.array([_mean_only(vals_on[i]) for i in layers])

    c_off = "#0072B2"  # Royal blue
    c_on = "#E69F00"  # Golden orange
    bar_w = 0.35
    edge_lw = 0.9

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(11, 4.4))

    ax.bar(idxs, means_off, width=bar_w,
           color=c_off, edgecolor="#333333", linewidth=edge_lw,
           label="Diff. Modulation OFF", alpha=0.55)

    ax.bar(idxs, means_on, width=bar_w,
           color=c_on, edgecolor="#333333", linewidth=edge_lw,
           label="Diff. Modulation ON", alpha=0.55)

    ax.plot(idxs, means_off, color=c_off, marker="o", markersize=6,
            linewidth=2.0, linestyle="-", alpha=0.85, label="_nolegend_")
    ax.plot(idxs, means_on, color=c_on, marker="o", markersize=6,
            linewidth=2.0, linestyle="-", alpha=0.85, label="_nolegend_")

    tick_every = 4 if L >= 12 else max(1, L // 3) if L > 0 else 1
    tick_positions = idxs
    tick_labels = [str(i) if (k % tick_every == 0) else "" for k, i in enumerate(idxs)]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, fontsize=16)

    ax.set_xlim(idxs.min() - 0.6 if L else -0.5, idxs.max() + 0.6 if L else 0.5)
    ax.set_ylabel("Score (↑ better)", fontsize=20)
    ax.set_xlabel("ATConv layer index (forward order)", fontsize=20)
    ax.set_title(f"{metric_name} — per-layer means", fontsize=20)

    ax.tick_params(axis="x", which="major", labelsize=18)
    ax.tick_params(axis="y", which="major", labelsize=18)

    ax.grid(which="major", axis="y", linestyle="--", linewidth=0.8, alpha=0.55)
    ax.grid(which="major", axis="x", linestyle=":", linewidth=0.4, alpha=0.55)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)

    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, dpi=600, bbox_inches="tight")
    print(f"[Saved] {save_path}")
    plt.close(fig)

    handles, labels = ax.get_legend_handles_labels()

    from matplotlib.patches import Patch
    c_off = "#0072B2"
    c_on = "#E69F00"
    handles = [
        Patch(facecolor=c_off, edgecolor="#333333", label="Diff. Modulation OFF", alpha=0.85),
        Patch(facecolor=c_on, edgecolor="#333333", label="Diff. Modulation ON", alpha=0.85),
    ]
    labels = [h.get_label() for h in handles]

    fig_leg = plt.figure(figsize=(4, 0.8))  # 宽高可调
    fig_leg.legend(
        handles, labels,
        loc="center",
        ncol=2,
        frameon=True,
        fontsize=20,
    )

    fig_leg.savefig("kd_figs/legend.png", dpi=600, bbox_inches="tight")
    plt.close(fig_leg)


# ---------------- main ----------------
@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=str, default="cpu", choices=["cuda", "cpu"])
    ap.add_argument("--img", type=str, default='test.jpg', help="Path to a single image (RGB)")
    ap.add_argument("--rand", action="store_true", help="Use random synthetic images")
    ap.add_argument("--num-images", type=int, default=8)
    ap.add_argument("--image-size", type=int, default=224)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--variant", type=str, default="T4", help="AttNet variant")
    ap.add_argument("--ckpt_with_kd", type=str, default="", help="checkpoint path")
    ap.add_argument("--ckpt_without_kd", type=str, default="", help="checkpoint path")
    ap.add_argument("--outdir", type=str, default="kd_figs", help="Output directory for figures")
    args = ap.parse_args()

    set_seed(args.seed)
    device = get_device(args.device)

    # ------- Data -------
    if args.img and os.path.isfile(args.img):
        imgs = load_single_image(args.img, size=args.image_size)
    elif args.rand:
        imgs = make_random_images(args.num_images, size=args.image_size, seed=args.seed)
    else:
        imgs = make_random_images(1, size=args.image_size, seed=args.seed)
    imgs = imgs.to(device)

    model_with_kd = AttNet_T4(num_classes=1000)
    model_with_kd = model_with_kd.to(device).eval()
    if args.ckpt_with_kd and os.path.isfile(args.ckpt_with_kd):
        ckpt = torch.load(args.ckpt_with_kd, map_location=device, weights_only=False)
        sd = ckpt.get("model", ckpt)
        missing, unexpected = model_with_kd.load_state_dict(sd, strict=False)
        if missing or unexpected:
            print(f"[ckpt] missing={len(missing)} unexpected={len(unexpected)}")

    model_without_kd = AttNet_T4(num_classes=1000)
    model_without_kd = model_without_kd.to(device).eval()
    if args.ckpt_without_kd and os.path.isfile(args.ckpt_without_kd):
        ckpt = torch.load(args.ckpt_without_kd, map_location=device, weights_only=False)
        sd = ckpt.get("model", ckpt)
        missing, unexpected = model_without_kd.load_state_dict(sd, strict=False)
        if missing or unexpected:
            print(f"[ckpt] missing={len(missing)} unexpected={len(unexpected)}")

    # ------- Forward once with KD-ON / KD-OFF -------
    # KD ON
    with KDGuard(model_with_kd, enabled=True):
        feats_on, h_on, convs = register_atconv_outputs_forward(model_with_kd)
        _ = model_with_kd.forward_features(imgs)
        remove_handles(h_on)

    # KD OFF
    with KDGuard(model_without_kd, enabled=False):
        feats_off, h_off, _ = register_atconv_outputs_forward(model_without_kd)
        _ = model_without_kd.forward_features(imgs)
        remove_handles(h_off)

    idx2tag = {i: tag for i, (tag, _) in enumerate(convs)}
    print("[AttConv order]")
    for i in range(len(convs)):
        print(f"  idx {i:02d}  ->  {idx2tag[i]}")

    os.makedirs(args.outdir, exist_ok=True)
    for metric_name, metric_fn in METRIC_FNS.items():
        vals_off = collect_values_for_metric(feats_off, metric_fn)
        vals_on = collect_values_for_metric(feats_on, metric_fn)

        common = sorted(set(vals_off.keys()).intersection(vals_on.keys()))
        vals_off = {k: vals_off[k] for k in common}
        vals_on = {k: vals_on[k] for k in common}

        png = os.path.join(args.outdir, f"{metric_name.replace(' ', '_').replace('–', '-')}_panelA.png")
        plot_panelA_only(metric_name, vals_off, vals_on, idx2tag, png)

    print("\nDone. Figures saved to:", os.path.abspath(args.outdir))


if __name__ == "__main__":
    main()
