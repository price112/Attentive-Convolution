from timm.data import IMAGENET_DEFAULT_STD, IMAGENET_DEFAULT_MEAN
from timm.layers import DropPath, trunc_normal_, LayerNorm2d, RmsNorm2d
from timm.models import register_model
import torch
import torch.nn.functional as F
import torch.nn as nn
from typing import List, Any, Type
from torch.nn.modules.utils import _pair

@torch.jit.script_if_tracing
def per_sample_depthwise_conv2d_unfold(
        x: torch.Tensor,  # (B, C, H, W)
        kernels: torch.Tensor,  # (B, C, kH, kW)
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
):

    B, C, H, W = x.shape
    kH, kW = int(kernels.size(-2)), int(kernels.size(-1))
    stride_h, stride_w = _pair(stride)
    pad_h, pad_w = _pair(padding)
    dil_h, dil_w = _pair(dilation)

    H_out = (H + 2 * pad_h - dil_h * (kH - 1) - 1) // stride_h + 1
    W_out = (W + 2 * pad_w - dil_w * (kW - 1) - 1) // stride_w + 1

    # (B, C*kH*kW, H_out*W_out)
    cols = F.unfold(
        x, kernel_size=(kH, kW),
        dilation=(dil_h, dil_w),
        padding=(pad_h, pad_w),
        stride=(stride_h, stride_w),
    )
    cols = cols.view(B, C, kH * kW, H_out * W_out)

    # (B, C, kH*kW, 1)
    w = kernels.view(B, C, kH * kW, 1)

    y = (cols * w).sum(dim=2)

    return y.view(B, C, H_out, W_out)

class LayerScale(nn.Module):
    def __init__(self, dim: int, init_value: float = 1e-6):
        super().__init__()
        self.gamma = nn.Parameter(init_value * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gamma.reshape(1, -1, 1, 1)

class ConvBN(nn.Sequential):

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 2, padding: int = 1, ):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels)
        )

class GLU(nn.Module):

    def __init__(self, in_features: int, hidden_features: int, act_layer: Type[nn.Module], drop: float = 0.,
                 glu_dconv=False, glu_norm = True):
        super().__init__()

        self.fc1 = nn.Conv2d(in_features, hidden_features * 2, 1)

        if glu_dconv:
            self.dwconv = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, padding=1,
                                    groups=hidden_features, bias=True)
        else:
            self.dwconv = nn.Identity()

        if glu_norm:
            self.norm_layer = RmsNorm2d(hidden_features, eps=1e-6)
        else:
            self.norm_layer = nn.Identity()

        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, in_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, v = self.fc1(x).chunk(2, dim=1)
        x = self.act(self.dwconv(x)) * v
        x = self.norm_layer(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class ATConv(nn.Module):

    def __init__(self,
                 dim: int,
                 act_layer: Type[nn.Module],
                 kernel_size: int = 3,
                 bias: bool = True, ):
        super().__init__()

        self.dim = dim
        self.kernel_size = kernel_size
        k2 = kernel_size * kernel_size
        self.padding = kernel_size // 2

        self.kernel_proj = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.pool = nn.AdaptiveAvgPool1d(output_size=k2)
        self.kernel_act = act_layer()
        self.kernel_gen = nn.Linear(k2, k2, bias=bias)
        self.x_proj = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

        self.proj = nn.Conv2d(dim, dim, 1, bias=bias)
        self.difference_control = nn.Parameter(torch.zeros(dim))

    def no_weight_decay(self):
        return {"difference_control"}

    def _generate_kernels(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        kernels = self.kernel_proj(x)
        kernels = kernels.view(B, C, H * W)
        kernels = self.pool(kernels)
        kernels = self.kernel_act(kernels)
        kernels = self.kernel_gen(kernels)
        kernels = kernels.reshape(B, C, self.kernel_size, self.kernel_size)

        return kernels

    def _apply_kernel_difference(self, kernels):
        mean_kernels = kernels.mean(dim=(2, 3), keepdim=True)
        factor = torch.sigmoid(self.difference_control).view(1, -1, 1, 1)

        return kernels - factor * mean_kernels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape

        kernels = self._generate_kernels(x)
        kernels = self._apply_kernel_difference(kernels)

        x = self.x_proj(x)
        x = x.reshape(1, B * C, H, W)
        kernels = kernels.reshape(B * C, 1, self.kernel_size, self.kernel_size)

        x = F.conv2d(x, kernels, padding=self.padding, groups=B * C)
        # x = per_sample_depthwise_conv2d_unfold(x = x, kernels = kernels,stride = 1, padding = self.padding, dilation = 1)

        x = x.reshape(B, C, H, W)

        return self.proj(x)


class Block(nn.Module):

    def __init__(self,
                 dim: int,
                 act_layer: Type[nn.Module],
                 kernel_size: int,
                 exp_rate: float = 4.0,
                 drop_path_rate: float = 0.,
                 conv_bias: bool = True,
                 use_layer_scale: bool = True,
                 glu_dconv=False,
                 glu_norm = True,):
        super().__init__()
        self.use_layer_scale = use_layer_scale

        self.token_mixer = ATConv(dim, act_layer, kernel_size, conv_bias)

        hidden_features = int(dim * exp_rate)
        glu_hidden_features = int(2 * hidden_features / 3)
        self.channel_mixer = GLU(in_features=dim, hidden_features=glu_hidden_features, act_layer=act_layer,
                                 glu_dconv=glu_dconv, glu_norm=glu_norm)

        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()
        self.layer_scale_1 = LayerScale(dim) if use_layer_scale else nn.Identity()
        self.layer_scale_2 = LayerScale(dim) if use_layer_scale else nn.Identity()
        self.norm_1 = LayerNorm2d(dim, eps=1e-6)
        self.norm_2 = LayerNorm2d(dim, eps=1e-6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path(self.layer_scale_1((self.token_mixer(self.norm_1(x)))))
        x = x + self.drop_path(self.layer_scale_2((self.channel_mixer(self.norm_2(x)))))

        return x


class AttNet(nn.Module):

    def __init__(self,
                 num_classes: int = 1000,
                 drop_path_rate: float = 0.,
                 depths: List[int] = [2, 2, 6, 2],
                 dims: List[int] = [64, 128, 256, 512],
                 exp_rates: List[float] = [4, 4, 4, 4],
                 kernel_sizes: List[int] = [5, 5, 5, 5],
                 conv_bias: bool = True,
                 distillation: bool = False,
                 use_layer_scale: bool = True,
                 glu_dconv=[True, True, True, True],
                 glu_norm=[True, True, True, True],
                 act_layer: Type[nn.Module] = nn.GELU,
                 **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.use_distillation = distillation

        self.downsample_layers = nn.ModuleList()

        self.downsample_layers.append(
            nn.Sequential(ConvBN(in_channels=3, out_channels=dims[0] // 2),
                          act_layer(),
                          ConvBN(in_channels=dims[0] // 2, out_channels=dims[0]), ))

        for i in range(len(dims) - 1):
            self.downsample_layers.append(
                ConvBN(in_channels=dims[i], out_channels=dims[i + 1])
            )

        self.stages = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur_dp_idx = 0
        for i in range(len(dims)):
            stage = nn.Sequential(*[
                Block(
                    dim=dims[i],
                    act_layer=act_layer,
                    exp_rate=exp_rates[i],
                    drop_path_rate=dp_rates[cur_dp_idx + j],
                    kernel_size=kernel_sizes[i],
                    conv_bias=conv_bias,
                    use_layer_scale=use_layer_scale,
                    glu_dconv=glu_dconv[i],
                    glu_norm =glu_norm[i]
                ) for j in range(depths[i])
            ])
            self.stages.append(stage)
            cur_dp_idx += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-5)

        self.head = nn.Linear(dims[-1], num_classes)

        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module):
        """Initialize weights with truncated normal distribution."""
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        for i in range(len(self.stages)):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        x = x.mean(dim=[-2, -1])
        x = self.norm(x)
        return x

    def forward(self, x: torch.Tensor) -> Any:
        x = self.forward_features(x)
        # implement an additional head here if use distillation.
        return self.head(x)


def _create_AttNet(variant: str, pretrained: bool = False, **kwargs):

    model_configs = {

        'XXS': {'depths': [2, 2, 4, 2], 'dims': [32, 64, 128, 240], 'exp_rates': [8, 8, 4, 4],
                'glu_dconv': [True, True, True, True],'glu_norm': [False, False, False, False],
                'kernel_sizes': [3, 3, 3, 3],
                'drop_path_rate': 0.02,
                'use_layer_scale': False},

        'XS': {'depths': [2, 2, 7, 2], 'dims': [40, 80, 160, 320], 'exp_rates': [8, 8, 4, 4],
               'glu_dconv': [True, True, True, True],'glu_norm': [False, False, False, False],
               'kernel_sizes': [3, 3, 3, 3],
               'drop_path_rate': 0.02,
               'use_layer_scale': True},

        'S': {'depths': [2, 3, 10, 3], 'dims': [40, 80, 160, 320], 'exp_rates': [8, 8, 4, 4],
              'glu_dconv': [True, True, True, True], 'glu_norm': [False, False, False, False],
              'kernel_sizes': [3, 3, 3, 3],
              'drop_path_rate': 0.1,
              'use_layer_scale': False},

        'T1': {'depths': [2, 3, 12, 3], 'dims': [48, 96, 224, 384], 'exp_rates': [8, 8, 4, 4],
               'glu_dconv': [True, True, True, True], 'kernel_sizes': [3, 3, 3, 3], 'drop_path_rate': 0.1},

        'T2': {'depths': [3, 3, 16, 3], 'dims': [64, 128, 288, 512], 'exp_rates': [8, 8, 4, 4],
               'glu_dconv': [True, True, True, True], 'kernel_sizes': [3, 3, 3, 3], 'drop_path_rate': 0.2},

        'T3': {'depths': [4, 4, 26, 4], 'dims': [72, 144, 320, 576], 'exp_rates': [8, 8, 4, 4],
               'glu_dconv': [True, True, True, True], 'kernel_sizes': [3, 3, 3, 3], 'drop_path_rate': 0.4},

        'T4': {'depths': [5, 5, 28, 5], 'dims': [96, 192, 384, 768], 'exp_rates': [8, 8, 4, 4],
               'glu_dconv': [True, True, True, True], 'kernel_sizes': [3, 3, 3, 3], 'drop_path_rate': 0.5},
    }

    if variant not in model_configs:
        raise ValueError(f"Unknown variant {variant}, available: {list(model_configs.keys())}")

    config = model_configs[variant]
    # Allow kwargs to override the default config
    config.update(kwargs)

    model = AttNet(**config)

    model.default_cfg = {
        'url': '',  # Add URL if model is pretrained
        'num_classes': kwargs.get('num_classes', 1000),
        'input_size': (3, 224, 224),
        'pool_size': None,
        'crop_pct': 224 / 256,
        'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN,
        'std': IMAGENET_DEFAULT_STD,
        'classifier': 'head',
    }

    return model


# 2.86M
@register_model
def AttNet_XXS(pretrained: bool = False, **kwargs):
    return _create_AttNet('XXS', pretrained=pretrained, **kwargs)


# 5.56M
@register_model
def AttNet_XS(pretrained: bool = False, **kwargs):
    return _create_AttNet('XS', pretrained=pretrained, **kwargs)


# 7.69M
@register_model
def AttNet_S(pretrained: bool = False, **kwargs):
    return _create_AttNet('S', pretrained=pretrained, **kwargs)


# 13.71M
@register_model
def AttNet_T1(pretrained: bool = False, **kwargs):
    return _create_AttNet('T1', pretrained=pretrained, **kwargs)


# 27.01M
@register_model
def AttNet_T2(pretrained: bool = False, **kwargs):
    return _create_AttNet('T2', pretrained=pretrained, **kwargs)


# 49.18M
@register_model
def AttNet_T3(pretrained: bool = False, **kwargs):
    return _create_AttNet('T3', pretrained=pretrained, **kwargs)


# 87.32M
@register_model
def AttNet_T4(pretrained: bool = False, **kwargs):
    return _create_AttNet('T4', pretrained=pretrained, **kwargs)



if __name__ == '__main__':
    from util.utils import cal_complexity

    net = AttNet_S(num_classes=1000)
    print("Model architecture:")
    print(net)

    n_parameters = sum(p.numel()
                       for p in net.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    flops, params = cal_complexity(net, pixel=224)
    print(f"FLOPs: {flops}")
    print(f"Trainable Parameters: {sum(p.numel() for p in net.parameters() if p.requires_grad) / 1_000_000:.2f}M")
