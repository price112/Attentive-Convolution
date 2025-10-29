import os, math, csv, re
from typing import Tuple, Dict, Callable, Optional, List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrow, Circle

# ---------- timm backbones ----------
from timm.models.convnext import convnext_tiny
from timm.models.swin_transformer import swin_tiny_patch4_window7_224
from timm.models.tiny_vit import tiny_vit_21m_224  # TinyViT-21M

# === ADD: extra backbones ===
from timm.models.vision_transformer import vit_tiny_patch16_224   # ViT-Tiny/16
from torchvision.models import resnet50 as tv_resnet50            # torchvision ResNet50
from torchvision.models import ResNet50_Weights                   # weights enum

from models import AttNet_T2


# ==============================
# Utils
# ==============================
def load_image(img_path: str = "test.jpg", size: int = 224) -> torch.Tensor:
    tfm = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])
    image = Image.open(img_path).convert("RGB")
    return tfm(image).unsqueeze(0)

def denorm_image_for_viz(x: torch.Tensor) -> np.ndarray:
    mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1,3,1,1)
    std  = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1,3,1,1)
    img = (x * std + mean).clamp(0,1)[0].permute(1,2,0).detach().cpu().numpy()
    return (img * 255.0 + 0.5).astype(np.uint8)

def dim_background(img_rgb: np.ndarray, bg_dim: float = 0.0) -> np.ndarray:
    bg_dim = float(np.clip(bg_dim, 0.0, 1.0))
    return (img_rgb.astype(np.float32) * (1.0 - bg_dim)).clip(0,255).astype(np.uint8)

def add_img_token_grid(ax, H_img: int, W_img: int, Hf: int, Wf: int, alpha: float = 0.35, lw: float = 0.6):
    step_y = H_img / Hf; step_x = W_img / Wf
    for i in range(1, Hf): ax.axhline(i * step_y, color='w', linewidth=lw, alpha=alpha)
    for j in range(1, Wf): ax.axvline(j * step_x, color='w', linewidth=lw, alpha=alpha)
    ax.set_xticks([]); ax.set_yticks([])

def resize_map_to_img(hmap: torch.Tensor, H_img: int, W_img: int) -> np.ndarray:
    hm = F.interpolate(hmap[None,None,...], size=(H_img, W_img), mode="bilinear", align_corners=False)[0,0]
    hm = hm - hm.min(); hm = hm / (hm.max() + 1e-12)
    return hm.detach().cpu().numpy().astype(np.float32)

def hex2rgb01(color_hex: str) -> tuple:
    color_hex = color_hex.lstrip('#')
    r = int(color_hex[0:2], 16) / 255.0
    g = int(color_hex[2:4], 16) / 255.0
    b = int(color_hex[4:6], 16) / 255.0
    return (r, g, b)

def hex2rgb255(color_hex: str) -> np.ndarray:
    r, g, b = hex2rgb01(color_hex)
    return np.array([int(round(r*255)), int(round(g*255)), int(round(b*255))], dtype=np.uint8)

def blend_fixedcolor_alpha(img_rgb: np.ndarray, alpha_map: np.ndarray, color_hex: str, alpha_scale: float = 1.0, alpha_min: float = 0.0) -> np.ndarray:
    H, W, _ = img_rgb.shape
    a = np.clip(alpha_min + alpha_scale * alpha_map, 0.0, 1.0).astype(np.float32)
    if a.ndim == 2: a = a[..., None]
    color = hex2rgb255(color_hex)[None, None, :].repeat(H, axis=0).repeat(W, axis=1)
    out = img_rgb.astype(np.float32) * (1.0 - a) + color.astype(np.float32) * a
    return out.clip(0, 255).astype(np.uint8)

def blend_overlay(img_rgb: np.ndarray, heat: np.ndarray, alpha: float = 0.55, cmap: str = "magma") -> np.ndarray:
    cm = plt.get_cmap(cmap)
    heat_rgb = (cm(heat)[..., :3] * 255.0).astype(np.uint8)
    out = (img_rgb.astype(np.float32) * (1 - alpha) + heat_rgb.astype(np.float32) * alpha)
    return out.clip(0,255).astype(np.uint8)

def even_grid_anchors(H: int, W: int, grid: int) -> List[Tuple[int,int]]:
    ys = np.linspace(0, H-1, grid).round().astype(int)
    xs = np.linspace(0, W-1, grid).round().astype(int)
    return [(int(y), int(x)) for y in ys for x in xs]

def gaussian_kernel2d(ks: int, sigma: float, device) -> torch.Tensor:
    c = ks//2
    y, x = torch.meshgrid(torch.arange(-c, c+1, device=device), torch.arange(-c, c+1, device=device), indexing='ij')
    k = torch.exp(-(x**2 + y**2)/(2*sigma**2)); k = k / k.sum()
    return k[None,None,...]

def smooth_scatter(H: int, W: int, anchors: List[Tuple[int,int]], values: List[float], sigma: float = 1.0, device='cpu') -> torch.Tensor:
    m = torch.zeros((1,1,H,W), device=device)
    w = torch.zeros((1,1,H,W), device=device)
    for (y,x), v in zip(anchors, values):
        m[0,0,y,x] += float(v); w[0,0,y,x] += 1.0
    ks = int(max(3, round(6*sigma))) | 1
    k = gaussian_kernel2d(ks, sigma, device=device)
    m_blur = F.conv2d(m, k, padding=ks//2)
    w_blur = F.conv2d(w, k, padding=ks//2).clamp_min(1e-6)
    return (m_blur / w_blur)[0,0]

# Sobel gradient (for vf_mode='grad')
_SOBEL_GX = torch.tensor([[1,0,-1],[2,0,-2],[1,0,-1]], dtype=torch.float32).view(1,1,3,3) / 8.0
_SOBEL_GY = torch.tensor([[1,2,1],[0,0,0],[-1,-2,-1]], dtype=torch.float32).view(1,1,3,3) / 8.0


# ==============================
# Build operators (unchanged)
# ==============================
def build_attnet_operator(att: nn.Module, img: torch.Tensor, stage_idx: int = 0):
    att.eval()
    with torch.no_grad():
        x = img
        for s in range(stage_idx + 1):
            x = att.downsample_layers[s](x)
            stage = att.stages[s]
            if s < stage_idx:
                for blk in stage: x = blk(x)
            else:
                if len(stage) > 1:
                    for blk in stage[:-1]: x = blk(x)
                last: nn.Module = stage[-1]
                z_in = last.norm_1(x)
    def op(z: torch.Tensor) -> torch.Tensor:
        return last.token_mixer(z)
    return z_in, op

def build_convnext_operator(net: nn.Module, img: torch.Tensor, stage_idx: int = 0):
    net.eval()
    with torch.no_grad():
        x = net.stem(img)
        for s in range(stage_idx): x = net.stages[s](x)
        stage = net.stages[stage_idx]
        x = stage.downsample(x)
        blocks = list(getattr(stage, "blocks", stage))
        for blk in blocks[:-1]: x = blk(x)
        last = blocks[-1]
        dw = None
        for name in ("conv_dw","dwconv","dw_conv","depthwise_conv"):
            if hasattr(last, name): dw = getattr(last, name); break
        if dw is None:
            for m in last.modules():
                if isinstance(m, nn.Conv2d) and m.groups == m.in_channels == m.out_channels:
                    dw = m; break
        if dw is None: raise AttributeError("ConvNeXtBlock: depthwise conv not found.")
        z_in = x
    def op(z: torch.Tensor) -> torch.Tensor:
        return dw(z)
    return z_in, op

def build_swin_operator(net: nn.Module, img: torch.Tensor, stage_idx: int = 0):
    net.eval()
    stages = getattr(net, "layers", None) or net.stages
    with torch.no_grad():
        x = net.patch_embed(img)        # NHWC
        for s in range(stage_idx): x = stages[s](x)
        stage = stages[stage_idx]
        x = stage.downsample(x)
        blocks = list(stage.blocks)
        for blk in blocks[:-1]: x = blk(x)
        last = blocks[-1]
        z_in_nhwc = x.detach().clone()
    z_in = z_in_nhwc.permute(0,3,1,2).contiguous()
    def op(z_bchw: torch.Tensor) -> torch.Tensor:
        x = z_bchw.permute(0,2,3,1).contiguous()
        x_norm = last.norm1(x)
        y = last._attn(x_norm)
        return y.permute(0,3,1,2).contiguous()
    return z_in, op

def build_tinyvit_operator(net: nn.Module, img: torch.Tensor, stage_idx: int = 1):
    net.eval()
    with torch.no_grad():
        x = net.patch_embed(img)  # BCHW
        stages = net.stages
        for s in range(stage_idx): x = stages[s](x)
        stage = stages[stage_idx]
        if hasattr(stage, "downsample"):
            x = stage.downsample(x)
            x_bhwc = x.permute(0, 2, 3, 1)
            blocks = list(stage.blocks)
            for blk in blocks[:-1]: x_bhwc = blk(x_bhwc)
            last = blocks[-1]  # TinyVitBlock
            z_in = x_bhwc.permute(0, 3, 1, 2).contiguous()
            ws = getattr(last, "window_size", 7)
            def op(z_img: torch.Tensor) -> torch.Tensor:
                x = z_img.permute(0, 2, 3, 1).contiguous()  # BHWC
                B, H, W, C = x.shape; L = H * W
                if H == ws and W == ws:
                    t = x.reshape(B, L, C); t = last.attn(t); out = t.view(B, H, W, C)
                else:
                    pad_b = (ws - H % ws) % ws; pad_r = (ws - W % ws) % ws
                    padding = pad_b > 0 or pad_r > 0
                    if padding: x = F.pad(x, (0, 0, 0, pad_r, 0, pad_b))
                    pH, pW = H + pad_b, W + pad_r; nH, nW = pH // ws, pW // ws
                    xw = x.view(B, nH, ws, nW, ws, C).transpose(2, 3).reshape(B*nH*nW, ws*ws, C)
                    xw = last.attn(xw)
                    xw = xw.view(B, nH, nW, ws, ws, C).transpose(2, 3).reshape(B, pH, pW, C)
                    out = xw[:, :H, :W].contiguous() if padding else xw
                return out.permute(0, 3, 1, 2).contiguous()
            return z_in, op
        else:
            blocks = list(stage.blocks)
            for blk in blocks[:-1]: x = blk(x)
            last = blocks[-1]  # MBConv
            z_in = x
            def op(z_img: torch.Tensor) -> torch.Tensor:
                y = last.conv1(z_img); y = last.act1(y); y = last.conv2(y)
                return y
            return z_in, op


# === ADD: ResNet 3×3 Conv (Bottleneck.conv2) ===
def build_resnet3x3_operator(net: nn.Module, img: torch.Tensor, stage_idx: int = 2):

    net.eval()
    with torch.no_grad():
        # stem
        x = net.conv1(img); x = net.bn1(x); x = net.relu(x); x = net.maxpool(x)
        # pick layer
        layers = [net.layer1, net.layer2, net.layer3, net.layer4]
        s = max(1, min(4, int(stage_idx))) - 1
        for i in range(s):
            x = layers[i](x)
        layer = layers[s]
        blocks = list(layer.children())
        for b in blocks[:-1]:
            x = b(x)
        last = blocks[-1]  # Bottleneck
        x1 = last.conv1(x); x1 = last.bn1(x1); x1 = last.relu(x1)
        z_in = x1.detach()
    def op(z: torch.Tensor) -> torch.Tensor:
        return last.conv2(z)
    return z_in, op


# === ADD: ViT vanilla self-attention（timm VisionTransformer） ===
def build_vit_operator(net: nn.Module, img: torch.Tensor):

    net.eval()
    with torch.no_grad():
        B = img.shape[0]
        x = net.patch_embed(img)
        if getattr(net, 'pos_embed', None) is not None:
            x = x + net.pos_embed[:, 1:, :]
        # 前 N-1 个 block
        blocks = list(net.blocks)
        for blk in blocks[:-1]:
            x = blk(x)
        last = blocks[-1]
        # 计算 H,W（优先取 grid_size）
        gs = getattr(net.patch_embed, 'grid_size', None)
        if gs is None:
            H = W = int(net.patch_embed.img_size[0] // net.patch_embed.patch_size[0])
        else:
            H, W = gs
        C = x.shape[-1]
        # (B,N,C) -> (B,C,H,W)
        z_in = x.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()
    def op(z_bchw: torch.Tensor) -> torch.Tensor:
        B, C, H, W = z_bchw.shape
        t = z_bchw.permute(0, 2, 3, 1).reshape(B, H * W, C)  # (B, N, C)
        t = last.norm1(t)
        t = last.attn(t)                                     # (B, N, C)
        out = t.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        return out
    return z_in, op


# ==============================
# Influence maps & stats
# ==============================
def token_influence_map(op: Callable[[torch.Tensor], torch.Tensor],
                        z_in: torch.Tensor,
                        anchor_hw: Tuple[int, int],
                        ch: int = 0,
                        reduce: str = "l1") -> torch.Tensor:
    z = z_in.clone().detach().requires_grad_(True)
    y = op(z)
    _, C, H, W = y.shape
    ah, aw = int(anchor_hw[0]), int(anchor_hw[1])
    ah = max(0, min(H - 1, ah)); aw = max(0, min(W - 1, aw))
    ch = max(0, min(C - 1, int(ch)))
    y_scalar = y[0, ch, ah, aw]
    (gz,) = torch.autograd.grad(y_scalar, z, retain_graph=False, create_graph=False)
    gmap = torch.sqrt((gz[0] ** 2).sum(0) + 1e-12) if reduce == "l2" else gz[0].abs().sum(0)
    gmap = gmap - gmap.min(); gmap = gmap / (gmap.max() + 1e-12)
    return gmap

def routing_stats_from_map(infl: torch.Tensor, anchor_hw: Tuple[int,int]) -> Dict[str, float]:
    H, W = infl.shape
    y, x = torch.meshgrid(torch.arange(H, device=infl.device),
                          torch.arange(W, device=infl.device), indexing='ij')
    w = infl / infl.sum().clamp_min(1e-9)
    cy = (y * w).sum(); cx = (x * w).sum()
    dy = float((cy - anchor_hw[0]).item()); dx = float((cx - anchor_hw[1]).item())
    dist = float((dy**2 + dx**2) ** 0.5)
    yv = (y - cy); xv = (x - cx)
    C = torch.stack([torch.stack([(yv*yv*w).sum(), (yv*xv*w).sum()]),
                     torch.stack([(xv*yv*w).sum(), (xv*xv*w).sum()])])
    evals = torch.linalg.eigvalsh(C).clamp_min(1e-12)
    aniso = float((evals.max() / evals.min()).item())
    p = (w.reshape(-1)).clamp_min(1e-12)
    entropy = float(-(p * p.log()).sum().item())
    return {"dy": dy, "dx": dx, "dist": dist, "aniso": aniso, "entropy": entropy}

def far_attention_ratio(infl: torch.Tensor, anchor_hw: Tuple[int,int], r0: int = 3) -> float:
    H, W = infl.shape
    ay, ax = anchor_hw
    y, x = torch.meshgrid(torch.arange(H, device=infl.device),
                          torch.arange(W, device=infl.device), indexing='ij')
    d = torch.sqrt((y - ay)**2 + (x - ax)**2)
    tot = infl.sum().clamp_min(1e-9)
    far = infl[d > float(r0)].sum()
    return float((far / tot).item())


# ==============================
# Col-1: Dynamic routing (arrows)
# ==============================
def draw_routing_overlay(
    ax,
    base_img: np.ndarray,
    Hf: int,
    Wf: int,
    anchors: List[Tuple[int,int]],
    vecs_feat: List[Tuple[float,float]],
    strength: List[float],
    fixed_color: str = "#00EEFF",
    alpha_min: float = 0.35,
    alpha_max: float = 1.00,
    arrow_scale: float = 1.25,
    width_frac: float = 0.30,
    head_frac: float = 1.05,
    min_arrow_px: float = 5.0,
    draw_anchor_dot: bool = True,
    bg_dim: float = 0.0,
    arrow_size_gain: float = 1.0,
    c1_uniform: bool = False,
    c1_alpha_fixed: Optional[float] = None,
    c1_shaft_w_ratio: Optional[float] = None,
    c1_head_w_ratio: Optional[float] = None,
    c1_head_l_ratio: Optional[float] = None,
):
    H_img, W_img = base_img.shape[:2]
    ax.imshow(dim_background(base_img, bg_dim))
    add_img_token_grid(ax, H_img, W_img, Hf, Wf, alpha=0.35, lw=0.6)

    cell_h = H_img / Hf; cell_w = W_img / Wf; cell_sz = min(cell_h, cell_w)
    scale_y = H_img / Hf; scale_x = W_img / Wf
    max_s = max(strength) if len(strength) else 1.0

    if draw_anchor_dot:
        for (ay, axf) in anchors:
            cy = (ay + 0.5) * scale_y - 0.5
            cx = (axf + 0.5) * scale_x - 0.5
            circ = Circle((cx, cy), radius=max(1.8, 0.18 * cell_sz)*arrow_size_gain,
                          edgecolor='w', facecolor='k', linewidth=1.0, alpha=0.90, zorder=3)
            ax.add_patch(circ)

    for (ay, axf), (dy, dx), s in zip(anchors, vecs_feat, strength):
        cy, cx = (ay + 0.5) * scale_y - 0.5, (axf + 0.5) * scale_x - 0.5
        vy = dy * scale_y * arrow_scale
        vx = dx * scale_x * arrow_scale
        L = math.sqrt(vx * vx + vy * vy)
        min_len = min_arrow_px * arrow_size_gain
        if L < min_len and L > 1e-6:
            scale = min_len / L; vx *= scale; vy *= scale

        if c1_uniform:
            alpha = (c1_alpha_fixed if c1_alpha_fixed is not None else (alpha_min + alpha_max) * 0.5)
            shaft_ratio = (c1_shaft_w_ratio if c1_shaft_w_ratio is not None else width_frac)
            head_w_ratio = (c1_head_w_ratio if c1_head_w_ratio is not None else head_frac)
            head_l_ratio = (c1_head_l_ratio if c1_head_l_ratio is not None else head_frac)
            shaft_w = cell_sz * shaft_ratio * arrow_size_gain
            head_w  = cell_sz * head_w_ratio * arrow_size_gain
            head_l  = cell_sz * head_l_ratio * arrow_size_gain
        else:
            s_norm = float(s) / (max_s + 1e-12)
            alpha = alpha_min + (alpha_max - alpha_min) * s_norm
            shaft_w = cell_sz * (width_frac * (0.60 + 0.40 * s_norm)) * arrow_size_gain
            head_w  = cell_sz * (head_frac  * (0.70 + 0.30 * s_norm)) * arrow_size_gain
            head_l  = cell_sz * (head_frac  * (0.70 + 0.30 * s_norm)) * arrow_size_gain

        arrow = FancyArrow(
            cx, cy, vx, vy,
            width=max(1.8, shaft_w),
            length_includes_head=True,
            head_width=max(3.4, head_w),
            head_length=max(3.6, head_l),
            linewidth=0,
            color=fixed_color,
            alpha=alpha,
            zorder=4
        )
        ax.add_patch(arrow)

    ax.set_xticks([]); ax.set_yticks([])


# ==============================
# Col-2: Routing streamlines — reversed flow + inflow-only red bg + boosted junctions
# ==============================
def _sobel_grad_2d(m: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    device = m.device
    gx = _SOBEL_GX.to(device)
    gy = _SOBEL_GY.to(device)
    t = m[None,None,...]
    Gx = F.conv2d(t, gx, padding=1)[0,0]
    Gy = F.conv2d(t, gy, padding=1)[0,0]
    return Gy, Gx  # (Gy, Gx)

def _gaussian_blur_numpy(arr: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return arr
    ks = int(max(3, round(6*sigma))) | 1
    k = gaussian_kernel2d(ks, sigma, device='cpu')
    t = torch.from_numpy(arr).float()[None,None,...]
    out = F.conv2d(t, k, padding=ks//2)[0,0].detach().cpu().numpy()
    return out

def draw_streamlines_overlay(
    ax,
    base_img: np.ndarray,
    Hf: int,
    Wf: int,
    anchors: List[Tuple[int,int]],
    vecs_feat: List[Tuple[float,float]],
    strengths: List[float],
    infl_maps: Optional[List[torch.Tensor]] = None,
    far_list: Optional[List[float]] = None,
    # field construction
    vf_mode: str = "grad",      # 'sparse' | 'grad' | 'force'
    vf_sigma: float = 1.2,
    vf_grad_sigma: float = 1.0,
    vf_post_sigma: float = 0.8,
    vf_beta: float = 1.0,
    vf_gamma: float = 1.2,
    vf_drop_below: float = 0.0,
    # seeding
    seed_mode: str = "far_topk",
    stream_n_seeds: int = 16,
    # style
    color_hex: str = "#FF7F00",
    alpha: float = 0.55,
    lw_min: float = 1.6,
    lw_max: float = 3.8,
    density: float = 1.15,
    bg_dim: float = 0.20,
    arrowsize: float = 1.5,
    # background (inflow-only red)
    inout_bg: bool = True,
    out_color: str = "#FF5A5A",
    inout_alpha: float = 0.45,
    inout_blur_sigma: float = 1.0,
    inout_coarse: int = 28,
    inout_region_blur: float = 8.0,
    inout_pow: float = 0.85,
    inout_min: float = 0.06,
    stream_reverse_dir: bool = True,
    inout_inflow_boost: float = 1.25,
    curvature_eps: float = 0.08,
    inflow_peak_gain: float = 0.60,
    inflow_peak_sigma: float = 6.0,
):
    H_img, W_img = base_img.shape[:2]
    base = dim_background(base_img, bg_dim)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Vy = torch.zeros((Hf, Wf), device=device)
    Vx = torch.zeros((Hf, Wf), device=device)

    # build vector field
    if vf_mode == "sparse" or infl_maps is None:
        Vy_s = torch.zeros((1,1,Hf,Wf), device=device)
        Vx_s = torch.zeros((1,1,Hf,Wf), device=device)
        W_s  = torch.zeros((1,1,Hf,Wf), device=device)
        for (ay, axf), (dy, dx), s in zip(anchors, vecs_feat, strengths):
            Vy_s[0,0,ay,axf] += float(dy) * float(s)
            Vx_s[0,0,ay,axf] += float(dx) * float(s)
            W_s [0,0,ay,axf] += float(max(1e-6, s))
        ks = int(max(3, round(6*vf_sigma))) | 1
        k = gaussian_kernel2d(ks, vf_sigma, device=device)
        Vy = (F.conv2d(Vy_s, k, padding=ks//2) / F.conv2d(W_s, k, padding=ks//2).clamp_min(1e-6))[0,0]
        Vx = (F.conv2d(Vx_s, k, padding=ks//2) / F.conv2d(W_s, k, padding=ks//2).clamp_min(1e-6))[0,0]
    else:
        yv, xv = torch.meshgrid(torch.arange(Hf, device=device),
                                torch.arange(Wf, device=device), indexing='ij')
        dmax = math.sqrt((Hf-1)**2 + (Wf-1)**2)

        if vf_mode == "grad":
            for (ay, axf), m in zip(anchors, infl_maps):
                m = m.to(device)
                if vf_grad_sigma > 0:
                    ks = int(max(3, round(6*vf_grad_sigma))) | 1
                    k = gaussian_kernel2d(ks, vf_grad_sigma, device=device)
                    m = F.conv2d(m[None,None,...], k, padding=ks//2)[0,0]
                Gy, Gx = _sobel_grad_2d(m)
                d = torch.sqrt((yv - ay)**2 + (xv - axf)**2)
                w = (m.clamp_min(0) ** float(vf_beta)) * ((d / dmax) ** float(vf_gamma))
                Vy += Gy * w
                Vx += Gx * w
        elif vf_mode == "force":
            for (ay, axf), m in zip(anchors, infl_maps):
                m = m.to(device).clamp_min(0)
                dy = (yv - ay).float()
                dx = (xv - axf).float()
                d = torch.sqrt(dy**2 + dx**2).clamp_min(1e-6)
                w = (m ** float(vf_beta)) * ((d / dmax) ** float(vf_gamma))
                Vy += dy / d * w
                Vx += dx / d * w

        if vf_post_sigma > 0:
            ks = int(max(3, round(6*vf_post_sigma))) | 1
            k = gaussian_kernel2d(ks, vf_post_sigma, device=device)
            Vy = F.conv2d(Vy[None,None,...], k, padding=ks//2)[0,0]
            Vx = F.conv2d(Vx[None,None,...], k, padding=ks//2)[0,0]

    # upscale field to image grid
    Vy_img = F.interpolate(Vy[None,None,...], size=(H_img, W_img), mode="bilinear", align_corners=False)[0,0].detach().cpu().numpy()
    Vx_img = F.interpolate(Vx[None,None,...], size=(H_img, W_img), mode="bilinear", align_corners=False)[0,0].detach().cpu().numpy()
    scale_y = H_img / float(Hf); scale_x = W_img / float(Wf)
    U = Vx_img * scale_x
    V = Vy_img * scale_y

    # reverse arrow direction
    if stream_reverse_dir:
        U, V = -U, -V

    # === Inflow-only red background ===
    if inout_bg:
        dUdx = np.gradient(U, axis=1)
        dVdy = np.gradient(V, axis=0)
        div  = dUdx + dVdy

        if inout_blur_sigma > 0:
            div = _gaussian_blur_numpy(div, inout_blur_sigma)

        if inout_coarse > 1:
            t = torch.from_numpy(div).float()[None, None, ...]
            pooled = F.avg_pool2d(t, kernel_size=inout_coarse, stride=inout_coarse)
            up = F.interpolate(pooled, size=(H_img, W_img), mode='bilinear', align_corners=False)[0,0].numpy()
            div = up

        if inout_region_blur > 0:
            div = _gaussian_blur_numpy(div, inout_region_blur)

        s = np.percentile(np.abs(div), 98) + 1e-6
        D = np.clip(div / s, -1.0, 1.0)

        # inflow map (only red)
        N = np.clip(-D, 0.0, 1.0) ** float(inout_pow)
        N[N < float(inout_min)] = 0.0

        if inout_inflow_boost is not None and inout_inflow_boost > 1.0:
            N = np.clip(N * float(inout_inflow_boost), 0.0, 1.0)

        if inflow_peak_gain is not None and inflow_peak_gain > 0:
            N_sharp = N.copy()
            N_big = _gaussian_blur_numpy(N_sharp, float(inflow_peak_sigma))
            local_contrast = N_sharp / (N_big + 1e-6)
            N = np.clip(N_sharp * (1.0 + float(inflow_peak_gain) * local_contrast), 0.0, 1.0)

        back = blend_fixedcolor_alpha(base, N, out_color, alpha_scale=inout_alpha, alpha_min=0.0)
        ax.imshow(back)
    else:
        ax.imshow(base)

    # grid on top of background
    add_img_token_grid(ax, H_img, W_img, Hf, Wf, alpha=0.30, lw=0.6)

    # slight curvature: rotate vectors by a tiny angle depending on local speed
    if curvature_eps is not None and curvature_eps > 0:
        speed = np.sqrt(U**2 + V**2)
        if speed.max() > 1e-12:
            s_norm = speed / (speed.max() + 1e-12)
            theta = float(curvature_eps) * s_norm  # small angle per-pixel
            ct = np.cos(theta); st = np.sin(theta)
            U_rot = U * ct - V * st
            V_rot = U * st + V * ct
            U, V = U_rot, V_rot

    # prune weak flows
    speed = np.sqrt(U**2 + V**2)
    if speed.max() > 0 and 0 < vf_drop_below < 1:
        thr = np.quantile(speed, vf_drop_below)
        mask = speed >= thr
        U = U * mask
        V = V * mask
        speed = speed * mask

    # linewidth map
    if speed.max() > 0:
        s_norm = speed / (speed.max() + 1e-12)
    else:
        s_norm = np.zeros_like(speed)
    lw = float(lw_min) + (float(lw_max) - float(lw_min)) * s_norm

    # seeds
    start_points = None
    if seed_mode != "none":
        if seed_mode == "far_topk" and far_list is not None and len(far_list) == len(anchors):
            idx = np.argsort(np.array(far_list))[::-1][:int(stream_n_seeds)]
            seeds = []
            for i in idx:
                ay, axf = anchors[i]
                py = (ay + 0.5) * (H_img / Hf) - 0.5
                px = (axf + 0.5) * (W_img / Wf) - 0.5
                seeds.append([px, py])
            start_points = np.array(seeds, dtype=float) if len(seeds) > 0 else None
        elif seed_mode == "grid":
            gs = max(2, int(math.sqrt(stream_n_seeds)))
            ys = np.linspace(0, H_img-1, gs).round().astype(int)
            xs = np.linspace(0, W_img-1, gs).round().astype(int)
            seeds = [[x, y] for y in ys for x in xs]
            start_points = np.array(seeds[:stream_n_seeds], dtype=float)

    r, g, b = hex2rgb01(color_hex)
    color_rgba = (r, g, b, float(alpha))

    X, Y = np.meshgrid(np.arange(W_img), np.arange(H_img))
    ax.streamplot(
        X, Y, U, V,
        color=color_rgba,
        linewidth=lw,
        density=float(density),
        arrowsize=float(arrowsize),
        minlength=0.10,
        maxlength=4.0,
        integration_direction='both',
        start_points=start_points,
        zorder=3
    )

    ax.set_xlim([0, W_img])
    ax.set_ylim([H_img, 0])
    ax.set_xticks([]); ax.set_yticks([])


# ==============================
# Col-3: Lateral Inhibition — original heatmap
# ==============================
def draw_inhibition_heat_overlay(
    ax,
    base_img: np.ndarray,
    lii_map_dense: torch.Tensor,
    cmap: str = "magma",
    alpha: float = 0.65,
    bg_dim: float = 0.25,
    show_grid: bool = True
):
    H_img, W_img = base_img.shape[:2]
    base = dim_background(base_img, bg_dim)
    m = lii_map_dense.clone()
    m = m - m.min(); m = m / (m.max() + 1e-12)
    heat = resize_map_to_img(m, H_img, W_img)
    over = blend_overlay(base, heat, alpha=alpha, cmap=cmap)
    ax.imshow(over)
    if show_grid:
        add_img_token_grid(ax, H_img, W_img, lii_map_dense.shape[0], lii_map_dense.shape[1], alpha=0.35, lw=0.6)
    ax.set_xticks([]); ax.set_yticks([])


# ==============================
# Lateral inhibition (SCE) for stats
# ==============================
def sce_lateral_inhibition_at_anchor(op: Callable[[torch.Tensor], torch.Tensor],
                                     z_in: torch.Tensor,
                                     anchor_hw: Tuple[int,int],
                                     ch: int = 0,
                                     K: int = 7,
                                     delta: float = 1e-2) -> float:
    def effective_kernel_grad(z: torch.Tensor, pos_hw: Tuple[int,int]) -> torch.Tensor:
        z = z.clone().detach().requires_grad_(True)
        y = op(z); h, w = pos_hw
        scalar = y[0, ch, h, w]
        (gz,) = torch.autograd.grad(scalar, z, retain_graph=False, create_graph=False)
        g = gz[0].abs().sum(0); g = g - g.min(); g = g / (g.max() + 1e-12)
        return g
    _, _, H, W = z_in.shape
    ah, aw = anchor_hw
    g0 = effective_kernel_grad(z_in, (ah, aw))
    z1 = z_in.clone().detach(); z1[:, :, ah, aw] += delta
    g1 = effective_kernel_grad(z1, (ah, aw))
    diff = g1 - g0
    p = K // 2
    hs, he = max(0, ah - p), min(H, ah + p + 1)
    ws, we = max(0, aw - p), min(W, aw + p + 1)
    patch = torch.zeros((K,K), device=diff.device)
    dh, dw = p - (ah - hs), p - (aw - ws)
    if he > hs and we > ws:
        patch[dh:dh + (he - hs), dw:dw + (we - ws)] = diff[hs:he, ws:we]
    center = (K//2, K//2)
    diag_gain = patch[center[0], center[1]].clamp(min=1e-9)
    off = patch.clone(); off[center[0], center[1]] = 0.0
    lii = float(((-off).clamp_min(0).sum() / diag_gain).item())
    return lii


# ============ helper for safe filenames ============
def _safe_name(s: str) -> str:
    s = re.sub(r"[^\w\-\.]+", "_", s.strip())
    s = re.sub(r"_+", "_", s)
    return s.strip("_")

# ============ NEW: unify dynamic-arrow thickness across models ============
def uniform_arrow_kwargs(H_img, W_img, Hf, Wf,
                         target_shaft_px=3.2,
                         target_head_px=7.0,
                         target_head_len_px=7.0
                         ):
    cell = min(H_img / float(Hf), W_img / float(Wf))
    return dict(
        c1_uniform=True,
        c1_shaft_w_ratio = float(target_shaft_px)    / max(cell, 1e-6),
        c1_head_w_ratio  = float(target_head_px)     / max(cell, 1e-6),
        c1_head_l_ratio  = float(target_head_len_px) / max(cell, 1e-6),
        arrow_size_gain=1.0,
    )


# ==============================
# Main
# ==============================
def run_routing_inhibition(
    img_path: str = "test.jpg",
    save_prefix: str = "routing_inhibition",
    use_attnet_ckpt: Optional[str] = None,
    stages: Optional[List[int]] = None,
    grid_anchors: int = 5,
    anchor_channel: int = 0,
    reduce: str = "l1",
    K: int = 7,
    delta: float = 1e-2,
    smooth_sigma: float = 1.0,
    # col-1
    arrow_scale: float = 1.25,
    width_frac: float = 0.30,
    head_frac: float = 1.05,
    min_arrow_px: float = 5.0,
    fixed_color: str = "#00EEFF",
    alpha_min: float = 0.35,
    alpha_max: float = 1.00,
    draw_anchor_dot: bool = True,
    c1_arrow_size_gain: float = 1.0,
    c1_uniform: bool = False,
    c1_alpha_fixed: Optional[float] = None,
    c1_shaft_w_ratio: Optional[float] = None,
    c1_head_w_ratio: Optional[float] = None,
    c1_head_l_ratio: Optional[float] = None,
    # far vector rule
    r_excl: int = 2,
    topk: int = 5,
    gamma: float = 1.6,
    far_r0: int = 3,
    # col-2 streamlines
    vf_mode: str = "force",          # 'sparse' | 'grad' | 'force'
    vf_sigma: float = 1.20,
    vf_grad_sigma: float = 0.5,
    vf_post_sigma: float = 0.55,
    vf_beta: float = 0.6,
    vf_gamma: float = 1.0,
    vf_drop_below: float = 0.001,
    seed_mode: str = "far_topk",
    stream_n_seeds: int = 224,
    stream_density: float = 2.7,
    stream_lw_min: float = 2.7,
    stream_lw_max: float = 5.4,
    stream_color: str = "#FF7F00",
    stream_alpha_overlay: float = 0.65,
    bg_dim_stream: float = 0.40,
    stream_arrowsize: float = 1.8,
    # inflow-only red background & boosts
    inout_bg: bool = False,
    out_color: str = "#FF5A5A",
    inout_alpha: float = 0.45,
    inout_blur_sigma: float = 1.5,
    inout_coarse: int = 28,
    inout_region_blur: float = 8.0,
    inout_pow: float = 0.85,
    inout_min: float = 0.06,
    stream_reverse_dir: bool = True,
    inout_inflow_boost: float = 1.25,
    curvature_eps: float = 0.08,
    inflow_peak_gain: float = 0.60,
    inflow_peak_sigma: float = 6.0,
    # col-3 LII heatmap
    lii_cmap: str = "magma",
    lii_alpha: float = 0.65,
    bg_dim_lii: float = 0.30,
    # col-1 bg
    bg_dim_routing: float = 0.40,
    # ===== NEW: save each subplot hi-res =====
    save_each_subplot: bool = True,
    subplot_dir: Optional[str] = None,     # default: f"{save_prefix}_subplots"
    subplot_dpi: int = 420,
    subplot_format: str = "png",
    subplot_pad_inches: float = 0.02,
    subplot_bbox_expand: Tuple[float, float] = (1.02, 1.06),  # (x_scale, y_scale)
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x_img = load_image(img_path).to(device)
    base_img = denorm_image_for_viz(x_img)
    H_img, W_img = base_img.shape[:2]

    TVIT = tiny_vit_21m_224(pretrained=True).to(device).eval()
    CNX  = convnext_tiny(pretrained=True).to(device).eval()
    SWIN = swin_tiny_patch4_window7_224(pretrained=True).to(device).eval()
    Att  = AttNet_T2().to(device)
    if use_attnet_ckpt and os.path.isfile(use_attnet_ckpt):
        ckpt = torch.load(use_attnet_ckpt, map_location=device, weights_only=False)
        sd = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        Att.load_state_dict(sd, strict=False)
    Att.eval()

    # === ADD: instantiate ResNet & ViT models
    try:
        RES  = tv_resnet50(weights=ResNet50_Weights.IMAGENET1K_V2).to(device).eval()
    except Exception:
        RES  = tv_resnet50(weights=ResNet50_Weights.DEFAULT).to(device).eval()
    VIT  = vit_tiny_patch16_224(pretrained=True).to(device).eval()

    stage_list = stages if stages is not None else [1, 2]

    for s in stage_list:
        z_tvt, op_tvt   = build_tinyvit_operator(TVIT, x_img, stage_idx=s)
        z_cnx, op_cnx   = build_convnext_operator(CNX, x_img, stage_idx=s if s < 4 else 3)
        z_swin, op_swin = build_swin_operator(SWIN, x_img, stage_idx=min(s, 3))
        z_att, op_att   = build_attnet_operator(Att, x_img, stage_idx=min(s, 3))

        packs = [
            ("TinyViT(Attn)" if s >= 1 else "TinyViT(Conv)", z_tvt,  op_tvt),
            ("ConvNeXt(DWConv)",                               z_cnx,  op_cnx),
            ("Swin(Attention)",                                z_swin, op_swin),
            ("AttNet(ATConv)",                                 z_att,  op_att),
        ]

        overlay_rows, stats_rows = [], []

        for name, z_in, op in packs:
            _, C, Hf, Wf = z_in.shape
            anchors_grid = even_grid_anchors(Hf, Wf, grid_anchors)

            vecs_feat, strengths = [], []
            lii_vals, lii_positions = [], []
            far_vals, infl_list = [], []

            for (ay, axf) in anchors_grid:
                infl = token_influence_map(op, z_in, (ay, axf), ch=anchor_channel, reduce=reduce)
                infl_list.append(infl.detach())
                st = routing_stats_from_map(infl, (ay, axf))

                def vector_far_topk(infl_map: torch.Tensor, anchor_hw: Tuple[int,int],
                                    r_excl: int = 1, topk: int = 5, gamma: float = 1.5):
                    H, W = infl_map.shape; ay, ax = anchor_hw
                    y, x = torch.meshgrid(torch.arange(H, device=infl_map.device),
                                          torch.arange(W, device=infl_map.device), indexing='ij')
                    d = torch.sqrt((y - ay)**2 + (x - ax)**2)
                    d_max = max(1.0, float(d.max().item()))
                    w = infl_map.clone()
                    if r_excl > 0: w[d <= float(r_excl)] = 0.0
                    w_far = w * (d / d_max) ** gamma
                    flat = w_far.view(-1)
                    if flat.max() <= 0: return 0.0, 0.0, 0.0
                    k = min(topk, int(H * W))
                    vals, inds = torch.topk(flat, k=k)
                    ys = (inds // W).float(); xs = (inds % W).float()
                    ssum = vals.sum().clamp_min(1e-9)
                    cy = float((ys * vals).sum() / ssum); cx = float((xs * vals).sum() / ssum)
                    dy = cy - ay; dx = cx - ax
                    d_mean = float(torch.sqrt((ys - ay)**2 + (xs - ax)**2).mean().item()) / d_max
                    s_vis = float(vals.mean().item()) * (0.5 + 0.5 * d_mean)
                    return dy, dx, s_vis

                dy, dx, s_vis = vector_far_topk(infl, (ay, axf), r_excl=r_excl, topk=topk, gamma=gamma)
                vecs_feat.append((dy, dx)); strengths.append(max(1e-9, s_vis))
                lii = sce_lateral_inhibition_at_anchor(op, z_in, (ay, axf), ch=anchor_channel, K=K, delta=delta)
                lii_vals.append(lii); lii_positions.append((ay, axf))
                far = far_attention_ratio(infl, (ay, axf), r0=far_r0); far_vals.append(far)

                stats_rows.append({
                    "stage": s, "operator": name, "Hf": Hf, "Wf": Wf,
                    "anchor_y": ay, "anchor_x": axf,
                    "routing_dy": st["dy"], "routing_dx": st["dx"], "routing_dist": st["dist"],
                    "routing_anisotropy": st["aniso"], "routing_entropy": st["entropy"],
                    "FAR@r0": far, "LII": lii
                })

            lii_dense = smooth_scatter(Hf, Wf, lii_positions, lii_vals, sigma=smooth_sigma, device=z_in.device)
            overlay_rows.append({
                "name": name, "Hf": Hf, "Wf": Wf,
                "anchors": anchors_grid, "vecs_feat": vecs_feat,
                "strengths": strengths, "lii_dense": lii_dense,
                "far_mean": float(np.mean(far_vals)),
                "infl_list": infl_list,
                "far_list": far_vals,
            })

        fig, axes = plt.subplots(4, 3, figsize=(18.5, 14.5))
        for i, row in enumerate(overlay_rows):
            name = row["name"]; Hf, Wf = row["Hf"], row["Wf"]
            anchors = row["anchors"]; vecs = row["vecs_feat"]
            strengths = row["strengths"]; lii_dense = row["lii_dense"]

            # Col-1
            axL = axes[i, 0]
            draw_routing_overlay(
                axL, base_img, Hf, Wf, anchors, vecs, strengths,
                fixed_color="#00EEFF",
                alpha_min=0.35, alpha_max=1.00,
                arrow_scale=1.25, width_frac=0.30, head_frac=1.05,
                min_arrow_px=5.0, draw_anchor_dot=True,
                bg_dim=bg_dim_routing,

                **uniform_arrow_kwargs(H_img, W_img, Hf, Wf,
                                       target_shaft_px=3.2, target_head_px=10.0, target_head_len_px=10.0),
            )
            axL.set_title(f"{name}  (stage {s})  Dynamic Routing  |  FAR@{far_r0}={row['far_mean']:.2f}", fontsize=12)

            # Col-2
            axM = axes[i, 1]
            draw_streamlines_overlay(
                ax=axM, base_img=base_img, Hf=Hf, Wf=Wf,
                anchors=anchors, vecs_feat=vecs, strengths=strengths,
                infl_maps=row["infl_list"], far_list=row["far_list"],
                vf_mode=vf_mode,
                vf_sigma=vf_sigma,
                vf_grad_sigma=vf_grad_sigma,
                vf_post_sigma=vf_post_sigma,
                vf_beta=vf_beta,
                vf_gamma=vf_gamma,
                vf_drop_below=vf_drop_below,
                seed_mode=seed_mode,
                stream_n_seeds=stream_n_seeds,
                color_hex=stream_color,
                alpha=stream_alpha_overlay,
                lw_min=stream_lw_min, lw_max=stream_lw_max,
                density=stream_density,
                bg_dim=bg_dim_stream,
                arrowsize=stream_arrowsize,
                inout_bg=inout_bg,
                out_color=out_color,
                inout_alpha=inout_alpha,
                inout_blur_sigma=inout_blur_sigma,
                inout_coarse=inout_coarse,
                inout_region_blur=inout_region_blur,
                inout_pow=inout_pow,
                inout_min=inout_min,
                stream_reverse_dir=stream_reverse_dir,
                inout_inflow_boost=inout_inflow_boost,
                curvature_eps=curvature_eps,
                inflow_peak_gain=inflow_peak_gain,
                inflow_peak_sigma=inflow_peak_sigma,
            )
            axM.set_title(f"{name}  (stage {s})  Routing Streamlines (inflow-only red)", fontsize=12)

            # Col-3
            axR = axes[i, 2]
            draw_inhibition_heat_overlay(
                axR, base_img, lii_dense,
                cmap=lii_cmap, alpha=lii_alpha, bg_dim=bg_dim_lii, show_grid=True
            )
            axR.set_title(f"{name}  (stage {s})  Lateral Inhibition (heatmap)", fontsize=12)

        plt.tight_layout()
        fig.canvas.draw()
        out_png = f"{save_prefix}_overlay_stage{s}.png"
        plt.savefig(out_png, dpi=220, bbox_inches="tight")
        print(f"[Stage {s}] Saved overlay: {out_png}")

        if save_each_subplot:
            out_dir = subplot_dir or f"{save_prefix}_subplots"
            os.makedirs(out_dir, exist_ok=True)
            renderer = fig.canvas.get_renderer()
            col_names = ["dynamic", "stream", "lii"]

            _MAX_PX = 64000

            for i, row in enumerate(overlay_rows):
                op_name_safe = _safe_name(row["name"])
                for j, col in enumerate(col_names):
                    ax = axes[i, j]
                    bbox_px = ax.get_tightbbox(renderer)
                    bbox_in = bbox_px.transformed(fig.dpi_scale_trans.inverted())
                    if subplot_bbox_expand is not None:
                        bbox_in = bbox_in.expanded(float(subplot_bbox_expand[0]),
                                                   float(subplot_bbox_expand[1]))
                    dpi_use = int(subplot_dpi)
                    w_px = bbox_in.width * dpi_use
                    h_px = bbox_in.height * dpi_use
                    if w_px > _MAX_PX or h_px > _MAX_PX:
                        scale = min(_MAX_PX / max(w_px, 1.0),
                                    _MAX_PX / max(h_px, 1.0))
                        dpi_use = max(72, int(dpi_use * scale))
                    out_one = os.path.join(out_dir, f"{save_prefix}_stage{s}_{op_name_safe}_{col}.{subplot_format}")
                    fig.savefig(out_one,
                                dpi=dpi_use,
                                bbox_inches=bbox_in,
                                pad_inches=float(subplot_pad_inches))
                    print(f"   [+] subplot saved: {out_one}")

        plt.close(fig)

        header = ["stage","operator","Hf","Wf","anchor_y","anchor_x",
                  "routing_dy","routing_dx","routing_dist","routing_anisotropy","routing_entropy","FAR@r0","LII"]
        csv_path = f"{save_prefix}_stats_stage{s}.csv"
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=header); w.writeheader()
            for r in stats_rows: w.writerow(r)
        print(f"[Stage {s}] Saved stats CSV: {csv_path}")

        def summarize(opname: str, key: str):
            vals = [r[key] for r in stats_rows if r["operator"] == opname]
            m = float(np.mean(vals)); sd = float(np.std(vals)); return m, sd
        ops = [p[0] for p in packs]
        print(f"=== Stage {s} Summary (mean ± std over {grid_anchors**2} anchors) ===")
        for opn in ops:
            m_far, s_far = summarize(opn, "FAR@r0")
            m_ent, s_ent = summarize(opn, "routing_entropy")
            m_lii, s_lii = summarize(opn, "LII")
            print(f"[{opn:<16}] FAR@{far_r0}={m_far:.2f}±{s_far:.2f}  entropy={m_ent:.2f}±{s_ent:.2f}  LII={m_lii:.2f}±{s_lii:.2f}")

        z_res, op_res = build_resnet3x3_operator(RES, x_img, stage_idx=min(max(s,1),4))
        z_vit, op_vit = build_vit_operator(VIT, x_img)

        packs_extra = [
            ("ResNet(3x3Conv)",  z_res, op_res),
            ("ViT(Self-Attn)",   z_vit, op_vit),
        ]

        overlay_rows2, stats_rows2 = [], []

        for name, z_in, op in packs_extra:
            _, C2, Hf2, Wf2 = z_in.shape
            anchors_grid2 = even_grid_anchors(Hf2, Wf2, grid_anchors)
            vecs2, str2 = [], []
            lii_vals2, lii_pos2 = [], []
            far_vals2, infl_list2 = [], []

            for (ay2, ax2) in anchors_grid2:
                infl2 = token_influence_map(op, z_in, (ay2, ax2), ch=anchor_channel, reduce=reduce)
                infl_list2.append(infl2.detach())
                st2 = routing_stats_from_map(infl2, (ay2, ax2))

                def vector_far_topk2(infl_map: torch.Tensor, anchor_hw: Tuple[int,int],
                                     r_excl: int = 1, topk: int = 5, gamma: float = 1.5):
                    H, W = infl_map.shape; ay, ax = anchor_hw
                    y, x = torch.meshgrid(torch.arange(H, device=infl_map.device),
                                          torch.arange(W, device=infl_map.device), indexing='ij')
                    d = torch.sqrt((y - ay)**2 + (x - ax)**2)
                    d_max = max(1.0, float(d.max().item()))
                    w = infl_map.clone()
                    if r_excl > 0: w[d <= float(r_excl)] = 0.0
                    w_far = w * (d / d_max) ** gamma
                    flat = w_far.view(-1)
                    if flat.max() <= 0: return 0.0, 0.0, 0.0
                    k = min(topk, int(H * W))
                    vals, inds = torch.topk(flat, k=k)
                    ys = (inds // W).float(); xs = (inds % W).float()
                    ssum = vals.sum().clamp_min(1e-9)
                    cy = float((ys * vals).sum() / ssum); cx = float((xs * vals).sum() / ssum)
                    dy = cy - ay; dx = cx - ax
                    d_mean = float(torch.sqrt((ys - ay)**2 + (xs - ax)**2).mean().item()) / d_max
                    s_vis = float(vals.mean().item()) * (0.5 + 0.5 * d_mean)
                    return dy, dx, s_vis

                dy2, dx2, s_vis2 = vector_far_topk2(infl2, (ay2, ax2), r_excl=r_excl, topk=topk, gamma=gamma)
                vecs2.append((dy2, dx2)); str2.append(max(1e-9, s_vis2))
                lii2 = sce_lateral_inhibition_at_anchor(op, z_in, (ay2, ax2), ch=anchor_channel, K=K, delta=delta)
                lii_vals2.append(lii2); lii_pos2.append((ay2, ax2))
                far2 = far_attention_ratio(infl2, (ay2, ax2), r0=far_r0); far_vals2.append(far2)

                stats_rows2.append({
                    "stage": s, "operator": name, "Hf": Hf2, "Wf": Wf2,
                    "anchor_y": ay2, "anchor_x": ax2,
                    "routing_dy": st2["dy"], "routing_dx": st2["dx"], "routing_dist": st2["dist"],
                    "routing_anisotropy": st2["aniso"], "routing_entropy": st2["entropy"],
                    "FAR@r0": far2, "LII": lii2
                })

            lii_dense2 = smooth_scatter(Hf2, Wf2, lii_pos2, lii_vals2, sigma=smooth_sigma, device=z_in.device)
            overlay_rows2.append({
                "name": name, "Hf": Hf2, "Wf": Wf2,
                "anchors": anchors_grid2, "vecs_feat": vecs2,
                "strengths": str2, "lii_dense": lii_dense2,
                "far_mean": float(np.mean(far_vals2)),
                "infl_list": infl_list2,
                "far_list": far_vals2,
            })

        fig2, axes2 = plt.subplots(2, 3, figsize=(18.5, 7.4))
        for i2, row2 in enumerate(overlay_rows2):
            name2 = row2["name"]; Hf2, Wf2 = row2["Hf"], row2["Wf"]
            anchors2 = row2["anchors"]; vecs2 = row2["vecs_feat"]
            strengths2 = row2["strengths"]; lii_dense2 = row2["lii_dense"]

            # Col-1
            axL2 = axes2[i2, 0]
            draw_routing_overlay(
                axL2, base_img, Hf2, Wf2, anchors2, vecs2, strengths2,
                fixed_color="#00EEFF",
                alpha_min=0.35, alpha_max=1.00,
                arrow_scale=1.25, width_frac=0.30, head_frac=1.05,
                min_arrow_px=5.0, draw_anchor_dot=True,
                bg_dim=bg_dim_routing,

                **uniform_arrow_kwargs(H_img, W_img, Hf2, Wf2,
                                       target_shaft_px=3.2, target_head_px=10.0, target_head_len_px=10.0),
            )
            axL2.set_title(f"{name2}  (stage {s})  Dynamic Routing  |  FAR@{far_r0}={row2['far_mean']:.2f}", fontsize=12)

            # Col-2
            axM2 = axes2[i2, 1]
            draw_streamlines_overlay(
                ax=axM2, base_img=base_img, Hf=Hf2, Wf=Wf2,
                anchors=anchors2, vecs_feat=vecs2, strengths=strengths2,
                infl_maps=row2["infl_list"], far_list=row2["far_list"],
                vf_mode=vf_mode,
                vf_sigma=vf_sigma,
                vf_grad_sigma=vf_grad_sigma,
                vf_post_sigma=vf_post_sigma,
                vf_beta=vf_beta,
                vf_gamma=vf_gamma,
                vf_drop_below=vf_drop_below,
                seed_mode=seed_mode,
                stream_n_seeds=stream_n_seeds,
                color_hex=stream_color,
                alpha=stream_alpha_overlay,
                lw_min=stream_lw_min, lw_max=stream_lw_max,
                density=stream_density,
                bg_dim=bg_dim_stream,
                arrowsize=stream_arrowsize,
                inout_bg=inout_bg,
                out_color=out_color,
                inout_alpha=inout_alpha,
                inout_blur_sigma=inout_blur_sigma,
                inout_coarse=inout_coarse,
                inout_region_blur=inout_region_blur,
                inout_pow=inout_pow,
                inout_min=inout_min,
                stream_reverse_dir=stream_reverse_dir,
                inout_inflow_boost=inout_inflow_boost,
                curvature_eps=curvature_eps,
                inflow_peak_gain=inflow_peak_gain,
                inflow_peak_sigma=inflow_peak_sigma,
            )
            axM2.set_title("Routing Streamlines (inflow-only red)", fontsize=12)

            # Col-3
            axR2 = axes2[i2, 2]
            draw_inhibition_heat_overlay(
                axR2, base_img, lii_dense2,
                cmap=lii_cmap, alpha=lii_alpha, bg_dim=bg_dim_lii, show_grid=True
            )
            axR2.set_title("Lateral Inhibition (heatmap)", fontsize=12)

        plt.tight_layout()
        fig2.canvas.draw()
        out_png2 = f"{save_prefix}_overlay_stage{s}_extra.png"
        plt.savefig(out_png2, dpi=220, bbox_inches="tight")
        print(f"[Stage {s}] Saved EXTRA overlay: {out_png2}")

        if save_each_subplot:
            out_dir = subplot_dir or f"{save_prefix}_subplots"
            os.makedirs(out_dir, exist_ok=True)
            renderer2 = fig2.canvas.get_renderer()
            col_names = ["dynamic", "stream", "lii"]
            _MAX_PX = 64000
            for i2, row2 in enumerate(overlay_rows2):
                op_name_safe = _safe_name(row2["name"])
                for j2, col in enumerate(col_names):
                    ax = axes2[i2, j2]
                    bbox_px = ax.get_tightbbox(renderer2)
                    bbox_in = bbox_px.transformed(fig2.dpi_scale_trans.inverted())
                    if subplot_bbox_expand is not None:
                        bbox_in = bbox_in.expanded(float(subplot_bbox_expand[0]),
                                                   float(subplot_bbox_expand[1]))
                    dpi_use = int(subplot_dpi)
                    w_px = bbox_in.width * dpi_use
                    h_px = bbox_in.height * dpi_use
                    if w_px > _MAX_PX or h_px > _MAX_PX:
                        scale = min(_MAX_PX / max(w_px, 1.0),
                                    _MAX_PX / max(h_px, 1.0))
                        dpi_use = max(72, int(dpi_use * scale))
                    out_one = os.path.join(out_dir, f"{save_prefix}_stage{s}_{op_name_safe}_{col}.{subplot_format}")
                    fig2.savefig(out_one,
                                 dpi=dpi_use,
                                 bbox_inches=bbox_in,
                                 pad_inches=float(subplot_pad_inches))
                    print(f"   [+] subplot saved: {out_one}")

        plt.close(fig2)

        header2 = ["stage","operator","Hf","Wf","anchor_y","anchor_x",
                   "routing_dy","routing_dx","routing_dist","routing_anisotropy","routing_entropy","FAR@r0","LII"]
        csv_path2 = f"{save_prefix}_stats_stage{s}_extra.csv"
        with open(csv_path2, "w", newline="") as f2:
            w2 = csv.DictWriter(f2, fieldnames=header2); w2.writeheader()
            for r in stats_rows2: w2.writerow(r)
        print(f"[Stage {s}] Saved EXTRA stats CSV: {csv_path2}")

        csv_path_all = f"{save_prefix}_stats_stage{s}_all.csv"
        with open(csv_path_all, "w", newline="") as fa:
            w_all = csv.DictWriter(fa, fieldnames=header2)
            w_all.writeheader()
            for r in stats_rows:  w_all.writerow(r)
            for r in stats_rows2: w_all.writerow(r)
        print(f"[Stage {s}] Saved ALL stats CSV: {csv_path_all}")


if __name__ == "__main__":
    IMG_PATH = "test.jpg"
    Att_CKPT = "AttNet_T2_289_84.39875250976563.pth"
    run_routing_inhibition(
        img_path=IMG_PATH,
        save_prefix="routing_inhibition",
        use_attnet_ckpt=Att_CKPT,
        stages=[1, 2],
        grid_anchors=5,
        anchor_channel=0,
        reduce="l1",
        K=9,
        delta=1e-2,
        smooth_sigma=1.0,
        bg_dim_routing=0.40,
        c1_arrow_size_gain=1.0,
        vf_mode="force",
        vf_sigma=1.20,
        vf_grad_sigma=0.5,
        vf_post_sigma=0.55,
        vf_beta=0.6,
        vf_gamma=1.0,
        vf_drop_below=0.001,
        seed_mode="far_topk",
        stream_n_seeds=224,
        stream_density=2.7,
        stream_lw_min=2.7,
        stream_lw_max=5.4,
        stream_arrowsize=1.8,
        stream_color="#FF7F00",
        stream_alpha_overlay=0.65,
        bg_dim_stream=0.40,
        inout_bg=False,
        out_color="#FF5A5A",
        inout_alpha=0.45,
        inout_blur_sigma=1.5,
        inout_coarse=28,
        inout_region_blur=8.0,
        inout_pow=0.85,
        inout_min=0.06,
        stream_reverse_dir=True,
        inout_inflow_boost=1.25,
        curvature_eps=0.08,
        inflow_peak_gain=0.60,
        inflow_peak_sigma=6.0,
        # col-3：LII heatmap
        lii_cmap="magma",
        lii_alpha=0.65,
        bg_dim_lii=0.30,
        # NEW: hi-res subplots
        save_each_subplot=True,
        subplot_dir=None,           # default: "<save_prefix>_subplots"
        subplot_dpi=420,
        subplot_format="png",
        subplot_pad_inches=0.02,
        subplot_bbox_expand=(1.02, 1.06),
    )
