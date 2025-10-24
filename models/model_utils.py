import torch
import torch.nn as nn
import math
from functools import partial
import torch.nn.functional as F
from functools import partial
from typing import List, Tuple, Optional, Dict, Any, Type

class ASF_GRN(nn.Module):
    """
    Adaptive Shift-Fusion GRN (ASF-GRN):
      - Global L2-norm → Ng  (B,C,1,1)
      - Shift-based Local L2-norm → Nl  (B,C,H,W)
      - Dynamic weight w = sigmoid(Ng - spatial_mean(Nl))
      - Fusion: N = w*Ng + (1-w)*Nl
      - Output: γ·(x * N) + β + x
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.gamma   = nn.Parameter(torch.zeros(1, dim, 1, 1))
        self.beta    = nn.Parameter(torch.zeros(1, dim, 1, 1))
        self.eps     = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape

        Gx = torch.norm(x, p=2, dim=(2,3), keepdim=True)
        Ng = Gx / (Gx.mean(dim=1, keepdim=True) + self.eps)

        x2 = x.pow(2)
        shift_u = torch.roll(x2, shifts=-1, dims=2)
        shift_d = torch.roll(x2, shifts= 1, dims=2)
        shift_l = torch.roll(x2, shifts=-1, dims=3)
        shift_r = torch.roll(x2, shifts= 1, dims=3)
        L2_local = (x2 + shift_u + shift_d + shift_l + shift_r) * 0.2

        Nl = torch.sqrt(L2_local + self.eps)
        Nl = Nl / (Nl.mean(dim=1, keepdim=True) + self.eps)

        mNl = Nl.mean(dim=(2,3), keepdim=True)

        w  = torch.sigmoid(Ng - mNl)

        N = w * Ng + (1.0 - w) * Nl
        return self.gamma * (x * N) + self.beta + x


class GRN(nn.Module):

    def __init__(self, dim: int):
        """
        dim: 通道数 C
        """
        super().__init__()

        self.gamma = nn.Parameter(torch.zeros(1, dim, 1, 1))
        self.beta  = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        Gx = torch.norm(x, p=2, dim=(2, 3), keepdim=True)

        Nx = Gx / (Gx.mean(dim=1, keepdim=True) + 1e-6)

        return self.gamma * (x * Nx) + self.beta + x

class ConvocationV3(nn.Module):

    def __init__(self,
                 dim: int,
                 act_layer: Type[nn.Module],
                 kernel_size: int = 3,
                 bias: bool = True,
                 forward_impl: str = 'conv'):
        super().__init__()
        if forward_impl not in ['conv', 'vmap']:
            raise ValueError("forward_impl must be 'conv' or 'vmap'.")

        self.dim = dim
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2

        # Layers to generate kernel weights from global context
        self.qk_proj = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.pool = nn.AdaptiveAvgPool2d(output_size=(kernel_size, kernel_size))
        self.kernel_gen = nn.Sequential(
            nn.Conv2d(dim, dim//4,kernel_size=1,stride=1, bias=bias),
            act_layer(),
            nn.Conv2d(dim//4, dim,kernel_size=1,stride=1, bias=bias),
        )

        self.v_proj = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.proj = nn.Conv2d(dim, dim,1, bias=bias)

        self.beta_residual_kernels = nn.Parameter(torch.zeros(dim))

    def no_weight_decay(self):
        """Specifies parameters to exclude from weight decay."""
        return {"beta_residual_kernels"}

    def _generate_kernels(self, x: torch.Tensor) -> torch.Tensor:

        qk = self.qk_proj(x)
        qk = self.pool(qk)
        kernels = self.kernel_gen(qk)
        return kernels

    def _apply_residual_to_kernels(self, kernels: torch.Tensor) -> torch.Tensor:
        """Applies a learnable residual modification to the kernels."""
        mean_per_kernel = kernels.mean(dim=(2, 3), keepdim=True)
        factor = torch.sigmoid(self.beta_residual_kernels).reshape(1, -1, 1, 1)
        return kernels - factor * mean_per_kernel

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Batch-as-group convolution implementation."""
        B, C, H, W = x.shape

        value = self.v_proj(x)
        kernels = self._generate_kernels(x)
        kernels = self._apply_residual_to_kernels(kernels)

        value = value.reshape(1, B * C, H, W)
        kernels = kernels.reshape(B * C, 1, self.kernel_size, self.kernel_size)

        out = F.conv2d(value, kernels, padding=self.padding, groups=B * C)
        out = out.reshape(B, C, H, W)

        return self.proj(out)


class ConvocationV4(nn.Module):

    def __init__(self,
                 dim: int,
                 act_layer: Type[nn.Module],
                 kernel_size: int = 3,
                 bias: bool = True,
                 forward_impl: str = 'conv'):
        super().__init__()
        if forward_impl not in ['conv', 'vmap']:
            raise ValueError("forward_impl must be 'conv' or 'vmap'.")

        self.dim = dim
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2

        # Layers to generate kernel weights from global context
        self.q_proj = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.k_proj = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.pool_kernel = nn.AdaptiveAvgPool2d(output_size=(kernel_size, kernel_size))
        self.pool_embedding = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.kernel_gen = nn.Linear(kernel_size*kernel_size, kernel_size*kernel_size, bias=bias)

        # Value projection and final projection
        self.v_proj = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.proj = nn.Conv2d(dim, dim,1, bias=bias)


    def _generate_kernels(self, x: torch.Tensor) -> torch.Tensor:
        B,C,H,W = x.shape
        qk = self.q_proj(x)
        qk = self.pool_kernel(qk)
        qk = qk.reshape(B,C,-1)
        kernels = self.kernel_gen(qk)
        return kernels.reshape(B,C,self.kernel_size,self.kernel_size)

    def _apply_residual_to_kernels(self, kernels,k) -> torch.Tensor:
        """Applies a learnable residual modification to the kernels."""
        B,C,H,W = k.shape
        mean_per_kernel = kernels.mean(dim=(2, 3), keepdim=True)
        factor = torch.sigmoid(k).reshape(B, -1, 1, 1)
        return kernels - factor * mean_per_kernel

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Batch-as-group convolution implementation."""
        B, C, H, W = x.shape

        value = self.v_proj(x)
        kernels = self._generate_kernels(x)
        k = self.k_proj(self.pool_embedding(x))

        kernels = self._apply_residual_to_kernels(kernels,k)

        value = value.reshape(1, B * C, H, W)
        kernels = kernels.reshape(B * C, 1, self.kernel_size, self.kernel_size)

        out = F.conv2d(value, kernels, padding=self.padding, groups=B * C)
        out = out.reshape(B, C, H, W)

        return self.proj(out)

class LayerNorm2d(nn.LayerNorm):
    def forward(self, x):
        # x (B,C,H,W) → (B,H,W,C)
        return super().forward(x.permute(0,2,3,1)).permute(0,3,1,2)

class PatchEmbed_1(nn.Module):

    def __init__(self,in_chans, embed_dim):
        super().__init__()


        self.proj_1 = nn.Conv2d(in_chans, embed_dim//2, kernel_size=3, stride=2,
                              padding=1)

        self.proj_2 = nn.Conv2d(embed_dim//2, embed_dim, kernel_size=3, stride=2,
                              padding=1)

        self.norm_1 = LayerNorm2d(embed_dim//2)
        self.norm_2 = LayerNorm2d(embed_dim)
        self.act = nn.GELU()
    def forward(self, x):
        x = self.proj_1(x)
        x = self.norm_1(x)
        x = self.act(x)
        x = self.proj_2(x)
        x = self.norm_2(x)
        return x

class PatchEmbed_2(nn.Module):

    def __init__(self,in_chans, embed_dim):
        super().__init__()


        self.proj_1 = nn.Conv2d(in_chans, embed_dim//2, kernel_size=3, stride=2,
                              padding=1)

        self.proj_2 = nn.Conv2d(embed_dim//2, embed_dim, kernel_size=3, stride=2,
                              padding=1)

        self.norm_1 = nn.BatchNorm2d(embed_dim//2)
        self.norm_2 = nn.BatchNorm2d(embed_dim)
        self.act = nn.GELU()
    def forward(self, x):
        x = self.proj_1(x)
        x = self.norm_1(x)
        x = self.act(x)
        x = self.proj_2(x)
        x = self.norm_2(x)
        return x

class PatchEmbed_3(nn.Module):

    def __init__(self,in_chans, embed_dim):
        super().__init__()


        self.proj_1 = nn.Conv2d(in_chans, embed_dim//2, kernel_size=3, stride=2,
                              padding=1)

        self.proj_2 = nn.Conv2d(embed_dim//2, embed_dim//2, kernel_size=3, stride=1,
                              padding=1)

        self.proj_3 = nn.Conv2d(embed_dim//2, embed_dim, kernel_size=3, stride=2,
                              padding=1)

        self.norm_1 = LayerNorm2d(embed_dim//2,eps=1e-6)
        self.norm_2 = LayerNorm2d(embed_dim//2, eps=1e-6)
        self.norm_3 = LayerNorm2d(embed_dim, eps=1e-6)
        self.act_1 = nn.GELU()
        self.act_2 = nn.GELU()
    def forward(self, x):
        x = self.proj_1(x)
        x = self.norm_1(x)
        x = self.act_1(x)
        x = self.proj_2(x)
        x = self.norm_2(x)
        x = self.act_2(x)
        x = self.proj_3(x)
        x = self.norm_3(x)
        return x

class SC_Mixer_GLU(nn.Module):

    def __init__(self, in_features: int, hidden_features: int, act_layer: Type[nn.Module], drop: float = 0.):
        super().__init__()
        self.hidden_features = hidden_features

        # 空间混合路径 (Spatial Mixer Path) - 作为门控
        # 高效地在低维输入上进行DWConv
        self.spatial_mixer = nn.Sequential(
            nn.Conv2d(in_features, in_features, kernel_size=3, padding=1,
                      groups=in_features, bias=True),
            act_layer(),  # 可以在这里加一个激活
            nn.Conv2d(in_features, hidden_features, kernel_size=1)
        )

        # 通道混合路径 (Channel Mixer Path) - 作为数据
        self.channel_mixer = nn.Conv2d(in_features, hidden_features, kernel_size=1)

        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, in_features, kernel_size=1)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 两条路径并行计算
        gate = self.spatial_mixer(x)
        data = self.channel_mixer(x)

        x = self.act(gate) * data

        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)

        return x


class GhostShiftGLU(nn.Module):

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        act_layer: Type[nn.Module] = nn.GELU,
        drop: float = 0.,
        ghost_ratio: float = 0.5,
    ):
        super().__init__()

        hidden_features = make_divisible(hidden_features, 4)

        main_ch  = make_divisible(int(hidden_features * (1 - ghost_ratio)), 4)
        ghost_ch = hidden_features - main_ch         # 仍然 %4 == 0
        assert main_ch > 0 and ghost_ch > 0

        self.hidden   = hidden_features
        self.main_ch  = main_ch
        self.ghost_ch = ghost_ch

        self.fc1 = nn.Conv2d(in_features, hidden_features * 2, 1)

        self.dwconv = nn.Conv2d(main_ch, main_ch, 3, padding=1,
                                groups=main_ch, bias=True)

        self.shift    = ChannelShift(ghost_ch)
        self.shift_proj = nn.Conv2d(ghost_ch, ghost_ch, 1,groups=ghost_ch, bias=True)

        self.act  = act_layer()
        self.fc2  = nn.Conv2d(hidden_features, in_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, v = self.fc1(x).chunk(2, dim=1)                 # (B, hidden, H, W)

        x_main, x_ghost = torch.split(x, [self.main_ch, self.ghost_ch], dim=1)

        x_main = self.dwconv(x_main)

        x_ghost = self.shift_proj(self.shift(x_ghost))

        x_mix = torch.cat([x_main, x_ghost], dim=1)
        x_mix = self.act(x_mix) * v

        x_mix = self.drop(x_mix)
        x_mix = self.fc2(x_mix)
        x_mix = self.drop(x_mix)
        return x_mix

class ChannelShift(nn.Module):
    """Zero-FLOP 4-direction shift (→, ←, ↓, ↑)"""
    def __init__(self, channels: int):
        super().__init__()
        assert channels % 4 == 0
        self.c_split = channels // 4

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c = self.c_split
        xs = torch.zeros_like(x)
        xs[:, 0*c:1*c, :, 1:]   = x[:, 0*c:1*c, :, :-1]     # →
        xs[:, 1*c:2*c, :, :-1]  = x[:, 1*c:2*c, :, 1:]      # ←
        xs[:, 2*c:3*c, 1:, :]   = x[:, 2*c:3*c, :-1, :]     # ↓
        xs[:, 3*c:4*c, :-1, :]  = x[:, 3*c:4*c, 1:, :]      # ↑
        return xs

class ShiftGLU(nn.Module):

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        act_layer: Type[nn.Module] = nn.GELU,
        drop: float = 0.,
    ):
        super().__init__()

        hidden_features = make_divisible(hidden_features, 4)

        self.fc1 = nn.Conv2d(in_features, hidden_features * 2, 1)

        self.shift = ChannelShift(hidden_features)
        self.shift_proj = nn.Conv2d(hidden_features, hidden_features, 1,groups=hidden_features, bias=True)

        self.act  = act_layer()
        self.fc2  = nn.Conv2d(hidden_features, in_features, 1)
        self.drop = nn.Dropout(drop)

    # ------------------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1) Expand & split for GLU
        x, v = self.fc1(x).chunk(2, dim=1)

        x = self.shift(x)
        x = self.shift_proj(x)
        x = self.act(x) * v
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


_OFFSETS4 = ((0, 1), (0, -1), (1, 0), (-1, 0))   # R,L,D,U
_OFFSETS8 = _OFFSETS4 + ((1, 1), (1, -1), (-1, 1), (-1, -1))

def _roll(x: torch.Tensor, dx: int, dy: int) -> torch.Tensor:
    return torch.roll(x, shifts=(dy, dx), dims=(-2, -1))

class DiffShiftFast(nn.Module):

    def __init__(
        self,
        channels: int,
        dirs: int = 4,
        mode: str = "static",
        reduction: int = 4,
        per_channel: bool = False,     # 仅 local 模式
        zero_dc: bool = False,
        backend: str = "roll",         # roll | conv
    ):
        super().__init__()
        assert dirs in (4, 8)
        self.dirs, self.mode, self.zero_dc, self.backend = dirs, mode, zero_dc, backend
        self.per_channel = per_channel
        self.channels = channels

        if mode == "static":
            self.weight = nn.Parameter(torch.zeros(dirs, channels))
            nn.init.uniform_(self.weight, -0.25, 0.25)

        elif mode == "global":
            hidden = max(channels // reduction, 4)
            self.se = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(channels, hidden, 1, bias=False),
                nn.GELU(),
                nn.Conv2d(hidden, dirs * channels, 1, bias=True),
            )

        elif mode == "local":
            out_ch = dirs * (channels if per_channel else 1)
            self.head = nn.Sequential(
                nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
                nn.GELU(),
                nn.Conv2d(channels, out_ch, 1, bias=True),
            )

    def _forward_roll(self, x, w):
        out = torch.zeros_like(x)
        offsets = _OFFSETS4 if self.dirs == 4 else _OFFSETS8       # [(dx,dy),...]

        for i, (dx, dy) in enumerate(offsets):
            diff = _roll(x, dx, dy) - x                            # (B,C,H,W)
            out += w[i] * diff                                     # broadcast 乘

        if self.zero_dc:
            out -= w.sum(dim=0) * x                                # center = -Σ_i w_i
        return out
    def _forward_conv(self, x, w):
        # w shape: static (dirs,C)；global (B,dirs,C)
        B, C, H, W = x.shape
        dirs = self.dirs

        k = torch.zeros(B, C, 3, 3, device=x.device, dtype=x.dtype) if w.dim()==3 \
            else torch.zeros(C, 3, 3, device=x.device, dtype=x.dtype)

        dxdy_idx = { (0,1):(1,2),(0,-1):(1,0),(1,0):(2,1),(-1,0):(0,1),
                     (1,1):(2,2),(1,-1):(2,0),(-1,1):(0,2),(-1,-1):(0,0) }

        offs = _OFFSETS4 if dirs==4 else _OFFSETS8
        for idx,(dx,dy) in enumerate(offs):
            h, w_ = dxdy_idx[(dx,dy)]
            k[..., h, w_] = w[..., idx, :] if w.dim()==3 else w[idx]

        # center
        k[...,1,1] = -w.sum(dim=-2) if w.dim()==2 else -w.sum(dim=1)

        if w.dim()==3:                 # (B,C,3,3) →  (B*C,1,3,3)
            k = k.view(B*C, 1, 3, 3)
            x_ = x.view(1, B*C, H, W)
            out = F.conv2d(x_, k, groups=B*C).view(B, C, H, W)
        else:                           # static，一组 kernel
            k = k.view(C, 1, 3, 3)
            out = F.conv2d(x, k, groups=C)

        return out

    # --------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape

        if self.mode == "static":
            w = self.weight                                              # (dirs,C)

        elif self.mode == "global":
            w = self.se(x).view(B, self.dirs, C)                         # (B,dirs,C)

        elif self.mode == "local":
            maps = self.head(x)                                          # (B,dirs*1,H,W) or (B,dirs*C,H,W)
            if self.per_channel:
                maps = maps.view(B, self.dirs, C, H, W)                  # (B,dirs,C,H,W)
            else:
                maps = maps.view(B, self.dirs, 1, H, W)                  # broadcast
            # roll-based逐方向乘
            offsets = _OFFSETS4 if self.dirs == 4 else _OFFSETS8
            out = torch.zeros_like(x)
            for i, (dx, dy) in enumerate(offsets):
                diff = _roll(x, dx, dy) - x
                out += maps[:, i] * diff                                 # (B,C,H,W) * (B,1/H,W)
            if self.zero_dc:
                out -= maps.sum(dim=1) * x
            return out

        if self.backend == "conv":                       # cuDNN friendly
            return self._forward_conv(x, w)
        else:                                            # default roll
            # broadcast w →   static:(dirs,1,C,1,1)  global:(dirs,B,C,1,1)
            w = w.view(self.dirs, *([1]*(x.dim()-1))).transpose(1, -1) if w.dim()==3 \
                else w.view(self.dirs, 1, C, 1, 1)
            return self._forward_roll(x, w)


class ShiftDiffGLU(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        act_layer,
        dirs: int = 8,
        mode: str = "static",
        zero_dc: bool = False,
        backend: str = "roll",     # roll | conv
        drop: float = 0.,
    ):
        super().__init__()
        hidden_features = make_divisible(hidden_features, 4)
        self.fc_in  = nn.Conv2d(in_features, hidden_features * 2, 1)
        self.diff   = DiffShiftFast(hidden_features, dirs=dirs, mode=mode, zero_dc=zero_dc, backend=backend)
        self.act    = nn.GELU()
        self.drop   = nn.Dropout(drop)
        self.fc_out = nn.Conv2d(hidden_features, in_features, 1)

    def forward(self, x):
        x, v = self.fc_in(x).chunk(2, 1)
        x = self.diff(x)
        x = self.act(x) * v
        x = self.drop(x)
        x = self.fc_out(x)
        return self.drop(x)

def make_divisible(v, divisor = 4, min_val = None) -> int:

    if min_val is None:
        min_val = divisor
    new_v = max(min_val, int((v + divisor - 1) // divisor) * divisor)
    if new_v < 0.9 * v:  # 避免过度降低
        new_v += divisor
    return new_v

class GRN(nn.Module):
    """ GRN (Global Response Normalization) layer
    """
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        Gx = torch.norm(x, p=2, dim=(1,2), keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        x = self.gamma * (x * Nx) + self.beta + x
        x = x.permute(0, 3, 1, 2)
        return x

@torch.no_grad()
def get_relative_position_cpb(query_size, key_size, pretrain_size=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pretrain_size = pretrain_size or query_size
    axis_qh = torch.arange(query_size[0], dtype=torch.float32, device=device)
    axis_kh = F.adaptive_avg_pool1d(axis_qh.unsqueeze(0), key_size[0]).squeeze(0)
    axis_qw = torch.arange(query_size[1], dtype=torch.float32, device=device)
    axis_kw = F.adaptive_avg_pool1d(axis_qw.unsqueeze(0), key_size[1]).squeeze(0)
    axis_kh, axis_kw = torch.meshgrid(axis_kh, axis_kw, indexing='ij')
    axis_qh, axis_qw = torch.meshgrid(axis_qh, axis_qw, indexing='ij')

    axis_kh = torch.reshape(axis_kh, [-1])
    axis_kw = torch.reshape(axis_kw, [-1])
    axis_qh = torch.reshape(axis_qh, [-1])
    axis_qw = torch.reshape(axis_qw, [-1])

    relative_h = (axis_qh[:, None] - axis_kh[None, :]) / (pretrain_size[0] - 1) * 8
    relative_w = (axis_qw[:, None] - axis_kw[None, :]) / (pretrain_size[1] - 1) * 8
    relative_hw = torch.stack([relative_h, relative_w], dim=-1).view(-1, 2)

    relative_coords_table, idx_map = torch.unique(relative_hw, return_inverse=True, dim=0)

    relative_coords_table = torch.sign(relative_coords_table) * torch.log2(
        torch.abs(relative_coords_table) + 1.0) / torch.log2(torch.tensor(8, dtype=torch.float32))

    return idx_map, relative_coords_table

# RoPE2D##################################

def init_2d_freqs(dim: int, num_heads: int, theta: float = 10.0, rotate: bool = True):
    freqs_x = []
    freqs_y = []
    mag = 1 / (theta ** (torch.arange(0, dim, 4)[: (dim // 4)].float() / dim))
    for i in range(num_heads):
        angles = torch.rand(1) * 2 * torch.pi if rotate else torch.zeros(1)
        fx = torch.cat([mag * torch.cos(angles), mag * torch.cos(torch.pi / 2 + angles)], dim=-1)
        fy = torch.cat([mag * torch.sin(angles), mag * torch.sin(torch.pi / 2 + angles)], dim=-1)
        freqs_x.append(fx)
        freqs_y.append(fy)
    freqs_x = torch.stack(freqs_x, dim=0)
    freqs_y = torch.stack(freqs_y, dim=0)
    freqs = torch.stack([freqs_x, freqs_y], dim=0)
    return freqs


def init_t_xy(end_x: int, end_y: int):
    t = torch.arange(end_x * end_y, dtype=torch.float32)
    t_x = (t % end_x).float()
    t_y = torch.div(t, end_x, rounding_mode='floor').float()
    return t_x, t_y


def compute_mixed_cis(freqs: torch.Tensor, t_x: torch.Tensor, t_y: torch.Tensor, num_heads: int):
    N = t_x.shape[0]
    # No float 16 for this range
    with torch.amp.autocast(enabled=False,device_type='cuda'):
        freqs_x = (t_x.unsqueeze(-1) @ freqs[0].unsqueeze(-2)).view(N, num_heads, -1).permute(1, 0, 2)
        freqs_y = (t_y.unsqueeze(-1) @ freqs[1].unsqueeze(-2)).view(N, num_heads, -1).permute(1, 0, 2)
        freqs_cis = torch.polar(torch.ones_like(freqs_x), freqs_x + freqs_y)
    return freqs_cis


def compute_axial_cis(dim: int, end_x: int, end_y: int, theta: float = 100.0):
    freqs_x = 1.0 / (theta ** (torch.arange(0, dim, 4)[: (dim // 4)].float() / dim))
    freqs_y = 1.0 / (theta ** (torch.arange(0, dim, 4)[: (dim // 4)].float() / dim))

    t_x, t_y = init_t_xy(end_x, end_y)
    freqs_x = torch.outer(t_x, freqs_x)
    freqs_y = torch.outer(t_y, freqs_y)
    freqs_cis_x = torch.polar(torch.ones_like(freqs_x), freqs_x)
    freqs_cis_y = torch.polar(torch.ones_like(freqs_y), freqs_y)
    return torch.cat([freqs_cis_x, freqs_cis_y], dim=-1)


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    if freqs_cis.shape == (x.shape[-2], x.shape[-1]):
        shape = [d if i >= ndim - 2 else 1 for i, d in enumerate(x.shape)]
    elif freqs_cis.shape == (x.shape[-3], x.shape[-2], x.shape[-1]):
        shape = [d if i >= ndim - 3 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor):
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq).to(xq.device), xk_out.type_as(xk).to(xk.device)
