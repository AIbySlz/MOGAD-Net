import math
import itertools
from collections.abc import Sequence
from typing import Optional, Union, Type, List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from torch.nn import LayerNorm
from functools import partial
from torch.nn.init import trunc_normal_
from torch.nn import functional as F


class TokenPacker(nn.Module):
    def __init__(self,
            depth_dim=32,
            height_dim=40,
            width_dim=32,
            embed_dim=1024,
            num_heads=1024//128,
            hidden_size=1024,
            scale_factor=2,
            norm_layer=partial(nn.LayerNorm, eps=1e-6)
    ):
        super().__init__()
        self.grid_depth_size = math.ceil(depth_dim/scale_factor)
        self.grid_height_size = math.ceil(height_dim/scale_factor)
        self.grid_width_size = math.ceil(width_dim/scale_factor)
        self.num_queries = self.grid_depth_size * self.grid_height_size * self.grid_width_size
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.scale_factor = scale_factor
        self.q_proj_1 = nn.Linear(embed_dim, embed_dim, bias=False)

        k_modules = [nn.Linear(embed_dim, embed_dim)]
        for _ in range(1,2):
            k_modules.append(nn.GELU())
            k_modules.append(nn.Linear(embed_dim, embed_dim))
        self.k_proj_1 = nn.Sequential(*k_modules)

        v_modules = [nn.Linear(embed_dim, embed_dim)]
        for _ in range(1,2):
            v_modules.append(nn.GELU())
            v_modules.append(nn.Linear(embed_dim, embed_dim))
        self.v_proj_1 = nn.Sequential(*v_modules)

        self.ln_q_1 = norm_layer(self.embed_dim)
        self.ln_k_1 = norm_layer(self.embed_dim)
        self.ln_v_1 = norm_layer(self.embed_dim)

        self.clip_attn = nn.MultiheadAttention(embed_dim, num_heads)

        modules = [nn.Linear(embed_dim, 2*embed_dim)]
        for _ in range(1, 2):
            modules.append(nn.GELU())
            modules.append(nn.Linear(2*embed_dim, 2*embed_dim))
        self.mlp = nn.Sequential(*modules)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def padding_matrix(self, x, scale_factor):
        padding_needed = [0] * x.ndim
        for i in range(1, 4):
            remainder = x.shape[i] % scale_factor
            if remainder == 0:
                padding_needed[i] = 0
            else:
                padding_needed[i] = scale_factor - remainder

        padding = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        padding[3] = padding_needed[3]
        padding[5] = padding_needed[2]
        padding[7] = padding_needed[1]
        padded_x = F.pad(x, padding, "constant", 0)
        mask_shape = list(padded_x.shape[:-1])
        mask = torch.ones(mask_shape)
        mask[:, :x.shape[1], :x.shape[2], :x.shape[3]] = 0

        x_cpu = x.detach().cpu()  # 修改位置
        padding_config = [(0, 0)] * x.ndim
        padding_config[3] = (0, padding_needed[3])
        padding_config[2] = (0, padding_needed[2])
        padding_config[1] = (0, padding_needed[1])
        padded_x_reflect = np.pad(x_cpu.numpy(), padding_config, mode='edge')
        padded_x_reflect = torch.from_numpy(padded_x_reflect).to(x.device)

        return padded_x, padded_x_reflect, mask

    def divide_feature_to_region(self, x, batch_size, embedding_size):
        x_reshaped = x.view(batch_size, self.grid_depth_size, self.scale_factor, self.grid_height_size, self.scale_factor, self.grid_width_size, self.scale_factor, embedding_size)
        x_permuted = x_reshaped.permute(0, 1, 3, 5, 2, 4, 6, 7)
        cube_size = self.scale_factor * self.scale_factor * self.scale_factor
        x_final = x_permuted.reshape(batch_size, self.grid_depth_size, self.grid_height_size, self.grid_width_size, cube_size, embedding_size)
        return x_final

    def forward(self, x, attn_mask=None):
        padding_mask = None
        for i in range(1, 4):  # 注意：i 在 [1, 2, 3] 范围内
            if x.shape[i] % self.scale_factor != 0:
                x, x_reflect, padding_mask = self.padding_matrix(x, self.scale_factor)
                break

        x_multi = x.reshape(x.shape[0], -1, x.shape[-1]).to(x.dtype) # multi-level

        # if torch.any(padding_mask > 0):
        if padding_mask is not None and torch.any(padding_mask > 0):
            x = x_reflect # original single-level

        key = self.ln_k_1(self.k_proj_1(x_multi))
        value = self.ln_v_1(self.v_proj_1(x_multi))

        N, x_depth, x_height, x_width, c = x.shape

        key_x = key.reshape(N, x_depth, x_height, x_width, c)
        value_x = value.reshape(N, x_depth, x_height, x_width, c)

        token_num = x_depth * x_height * x_width

        q = F.interpolate(x.float().permute(0,4,1,2,3), size=(self.grid_depth_size, self.grid_height_size, self.grid_width_size), mode='trilinear').permute(0,2,3,4,1) ## fix
        q = q.reshape(q.shape[0], -1, q.shape[-1]).to(x.dtype)

        query = self.ln_q_1(self.q_proj_1(q)).permute(1, 0, 2)
        query_x = query.unsqueeze(2)

        key_x = self.divide_feature_to_region(key_x, N, c)
        value_x = self.divide_feature_to_region(value_x, N, c)
        key_x = key_x.reshape(N, -1, self.scale_factor * self.scale_factor * self.scale_factor, c).permute(1, 0, 2, 3)
        value_x = value_x.reshape(N, -1, self.scale_factor * self.scale_factor * self.scale_factor, c).permute(1, 0, 2, 3)
        iteration_num = query_x.size(0)
        batch_num = query_x.size(1)
        query_x = query_x.reshape(-1, query_x.size(2), query_x.size(3))
        key_x = key_x.reshape(-1, key_x.size(2), key_x.size(3))
        value_x = value_x.reshape(-1, value_x.size(2), value_x.size(3))
        output, attn_output_weights = self.clip_attn(query_x.permute(1, 0, 2), key_x.permute(1, 0, 2), value_x.permute(1, 0, 2))
        x = output.permute(1, 0, 2)
        x = x.reshape(iteration_num, batch_num, query_x.size(1), query_x.size(2))
        x = x.reshape(iteration_num, -1, query_x.size(2))
        x = x.permute(1, 0, 2)
        x = self.mlp(x)
        x = x.reshape(x.shape[0], self.grid_depth_size, self.grid_height_size, self.grid_width_size, x.shape[-1]).to(x.dtype)
        return x

    def to(self, device):
        return super().to(device)

def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    """Tensor initialization with truncated normal distribution.
    Based on:
    https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    https://github.com/rwightman/pytorch-image-models

    Args:
       tensor: an n-dimensional `torch.Tensor`.
       mean: the mean of the normal distribution.
       std: the standard deviation of the normal distribution.
       a: the minimum cutoff value.
       b: the maximum cutoff value.
    """

    def norm_cdf(x):
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.0))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor

def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
    """Tensor initialization with truncated normal distribution.
    Based on:
    https://github.com/rwightman/pytorch-image-models

    Args:
       tensor: an n-dimensional `torch.Tensor`
       mean: the mean of the normal distribution
       std: the standard deviation of the normal distribution
       a: the minimum cutoff value
       b: the maximum cutoff value
    """

    if std <= 0:
        raise ValueError("the standard deviation should be greater than zero.")

    if a >= b:
        raise ValueError("minimum cutoff value (a) should be smaller than maximum cutoff value (b).")

    return _no_grad_trunc_normal_(tensor, mean, std, a, b)



class DropPath(nn.Module):
    """Stochastic drop paths per sample for residual blocks.
    Based on:
    https://github.com/rwightman/pytorch-image-models
    """

    def __init__(self, drop_prob: float = 0.0, scale_by_keep: bool = True) -> None:
        """
        Args:
            drop_prob: drop path probability.
            scale_by_keep: scaling by non-dropped probability.
        """
        super().__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

        if not (0 <= drop_prob <= 1):
            raise ValueError("Drop path prob should be between 0 and 1.")

    def drop_path(self, x, drop_prob: float = 0.0, training: bool = False, scale_by_keep: bool = True):
        if drop_prob == 0.0 or not training:
            return x
        keep_prob = 1 - drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        if keep_prob > 0.0 and scale_by_keep:
            random_tensor.div_(keep_prob)
        return x * random_tensor

    def forward(self, x):
        return self.drop_path(x, self.drop_prob, self.training, self.scale_by_keep)


class Mlp(nn.Module):
    """
    A multi-layer perceptron block, based on: "Dosovitskiy et al.,
    An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale <https://arxiv.org/abs/2010.11929>"
    """

    def __init__(
        self, hidden_size: int, mlp_dim: int, dropout_rate: float = 0.0, act_layer=nn.GELU,
    ) -> None:
        """
        Args:
            hidden_size: dimension of hidden layer.
            mlp_dim: dimension of feedforward layer. If 0, `hidden_size` will be used.
            dropout_rate: fraction of the input units to drop.
            act_layer: activation function to use. Defaults to GELU.
        """

        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")
        mlp_dim = mlp_dim or hidden_size
        self.linear1 = nn.Linear(hidden_size, mlp_dim)
        self.linear2 = nn.Linear(mlp_dim, hidden_size)
        self.act = act_layer()  # 修改位置：将 nn.GELU() 替换为 act_layer()
        self.drop1 = nn.Dropout(dropout_rate)
        self.drop2 = self.drop1

    def forward(self, x):
        x = self.act(self.linear1(x))
        x = self.drop1(x)
        x = self.linear2(x)
        x = self.drop2(x)
        return x


def window_partition(x, window_size):
    """window partition operation based on: "Liu et al.,
    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/abs/2103.14030>"
    https://github.com/microsoft/Swin-Transformer

     Args:
        x: input tensor.
        window_size: local window size.
    """
    x_shape = x.size()  # length 4 or 5 only
    if len(x_shape) == 5:
        b, d, h, w, c = x_shape
        x = x.view(
            b,
            d // window_size[0],
            window_size[0],
            h // window_size[1],
            window_size[1],
            w // window_size[2],
            window_size[2],
            c,
        )
        windows = (
            x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(-1, window_size[0] * window_size[1] * window_size[2], c)
        )
    else:  # if len(x_shape) == 4:
        b, h, w, c = x.shape
        x = x.view(b, h // window_size[0], window_size[0], w // window_size[1], window_size[1], c)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0] * window_size[1], c)

    return windows


def window_reverse(windows, window_size, dims):
    """window reverse operation based on: "Liu et al.,
    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/abs/2103.14030>"
    https://github.com/microsoft/Swin-Transformer

     Args:
        windows: windows tensor.
        window_size: local window size.
        dims: dimension values.
    """
    if len(dims) == 4:
        b, d, h, w = dims
        x = windows.view(
            b,
            d // window_size[0],
            h // window_size[1],
            w // window_size[2],
            window_size[0],
            window_size[1],
            window_size[2],
            -1,
        )
        x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(b, d, h, w, -1)

    elif len(dims) == 3:
        b, h, w = dims
        x = windows.view(b, h // window_size[0], w // window_size[1], window_size[0], window_size[1], -1)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(b, h, w, -1)
    return x


def get_window_size(x_size, window_size, shift_size=None):
    """Computing window size based on: "Liu et al.,
    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/abs/2103.14030>"
    https://github.com/microsoft/Swin-Transformer

     Args:
        x_size: input size.
        window_size: local window size.
        shift_size: window shifting size.
    """

    use_window_size = list(window_size)
    if shift_size is not None:
        use_shift_size = list(shift_size)
    for i in range(len(x_size)):
        if x_size[i] <= window_size[i]:
            use_window_size[i] = x_size[i]
            if shift_size is not None:
                use_shift_size[i] = 0

    if shift_size is None:
        return tuple(use_window_size)
    else:
        return tuple(use_window_size), tuple(use_shift_size)


class WindowAttention(nn.Module):
    """
    Window based multi-head self attention module with relative position bias based on: "Liu et al.,
    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/abs/2103.14030>"
    https://github.com/microsoft/Swin-Transformer
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: Sequence[int],
        qkv_bias: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        """
        Args:
            dim: number of feature channels.
            num_heads: number of attention heads.
            window_size: local window size.
            qkv_bias: add a learnable bias to query, key, value.
            attn_drop: attention dropout rate.
            proj_drop: dropout rate of output.
        """

        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5
        mesh_args = torch.meshgrid.__kwdefaults__

        if len(self.window_size) == 3:
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros(
                    (2 * self.window_size[0] - 1) * (2 * self.window_size[1] - 1) * (2 * self.window_size[2] - 1),
                    num_heads,
                )
            )
            coords_d = torch.arange(self.window_size[0])
            coords_h = torch.arange(self.window_size[1])
            coords_w = torch.arange(self.window_size[2])
            if mesh_args is not None:
                coords = torch.stack(torch.meshgrid(coords_d, coords_h, coords_w, indexing="ij"))
            else:
                coords = torch.stack(torch.meshgrid(coords_d, coords_h, coords_w))
            coords_flatten = torch.flatten(coords, 1)
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()
            relative_coords[:, :, 0] += self.window_size[0] - 1
            relative_coords[:, :, 1] += self.window_size[1] - 1
            relative_coords[:, :, 2] += self.window_size[2] - 1
            relative_coords[:, :, 0] *= (2 * self.window_size[1] - 1) * (2 * self.window_size[2] - 1)
            relative_coords[:, :, 1] *= 2 * self.window_size[2] - 1
        elif len(self.window_size) == 2:
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads)
            )
            coords_h = torch.arange(self.window_size[0])
            coords_w = torch.arange(self.window_size[1])
            if mesh_args is not None:
                coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"))
            else:
                coords = torch.stack(torch.meshgrid(coords_h, coords_w))
            coords_flatten = torch.flatten(coords, 1)
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()
            relative_coords[:, :, 0] += self.window_size[0] - 1
            relative_coords[:, :, 1] += self.window_size[1] - 1
            relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1

        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        trunc_normal_(self.relative_position_bias_table, std=0.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask):
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.clone()[:n, :n].reshape(-1)
        ].reshape(n, n, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)
        if mask is not None:
            nw = mask.shape[0]
            attn = attn.view(b // nw, nw, self.num_heads, n, n) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, n, n)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn).to(v.dtype)
        x = (attn @ v).transpose(1, 2).reshape(b, n, c)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x



class SwinTransformerBlock(nn.Module):
    """
    Swin Transformer block based on: "Liu et al.,
    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/abs/2103.14030>"
    https://github.com/microsoft/Swin-Transformer
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: Sequence[int],
        shift_size: Sequence[int],
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        act_layer: str = "GELU",
        norm_layer: type[LayerNorm] = nn.LayerNorm,
        use_checkpoint: bool = False,
    ) -> None:
        """
        Args:
            dim: number of feature channels.
            num_heads: number of attention heads.
            window_size: local window size.
            shift_size: window shift size.
            mlp_ratio: ratio of mlp hidden dim to embedding dim.
            qkv_bias: add a learnable bias to query, key, value.
            drop: dropout rate.
            attn_drop: attention dropout rate.
            drop_path: stochastic depth rate.
            act_layer: activation layer.
            norm_layer: normalization layer.
            use_checkpoint: use gradient checkpointing for reduced memory usage.
        """

        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.use_checkpoint = use_checkpoint
        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim,
            window_size=self.window_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(hidden_size=dim, mlp_dim=mlp_hidden_dim, dropout_rate=drop)


    def forward_part1(self, x, mask_matrix):
        x_shape = x.size()
        x = self.norm1(x)
        if len(x_shape) == 5:
            b, d, h, w, c = x.shape
            window_size, shift_size = get_window_size((d, h, w), self.window_size, self.shift_size)
            pad_l = pad_t = pad_d0 = 0
            pad_d1 = (window_size[0] - d % window_size[0]) % window_size[0]
            pad_b = (window_size[1] - h % window_size[1]) % window_size[1]
            pad_r = (window_size[2] - w % window_size[2]) % window_size[2]
            x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b, pad_d0, pad_d1))
            _, dp, hp, wp, _ = x.shape
            dims = [b, dp, hp, wp]

        else:  # elif len(x_shape) == 4
            b, h, w, c = x.shape
            window_size, shift_size = get_window_size((h, w), self.window_size, self.shift_size)
            pad_l = pad_t = 0
            pad_b = (window_size[0] - h % window_size[0]) % window_size[0]
            pad_r = (window_size[1] - w % window_size[1]) % window_size[1]
            x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
            _, hp, wp, _ = x.shape
            dims = [b, hp, wp]

        if any(i > 0 for i in shift_size):
            if len(x_shape) == 5:
                shifted_x = torch.roll(x, shifts=(-shift_size[0], -shift_size[1], -shift_size[2]), dims=(1, 2, 3))
            elif len(x_shape) == 4:
                shifted_x = torch.roll(x, shifts=(-shift_size[0], -shift_size[1]), dims=(1, 2))
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None
        x_windows = window_partition(shifted_x, window_size)
        attn_windows = self.attn(x_windows, mask=attn_mask)
        attn_windows = attn_windows.view(-1, *(window_size + (c,)))
        shifted_x = window_reverse(attn_windows, window_size, dims)
        if any(i > 0 for i in shift_size):
            if len(x_shape) == 5:
                x = torch.roll(shifted_x, shifts=(shift_size[0], shift_size[1], shift_size[2]), dims=(1, 2, 3))
            elif len(x_shape) == 4:
                x = torch.roll(shifted_x, shifts=(shift_size[0], shift_size[1]), dims=(1, 2))
        else:
            x = shifted_x

        if len(x_shape) == 5:
            if pad_d1 > 0 or pad_r > 0 or pad_b > 0:
                x = x[:, :d, :h, :w, :].contiguous()
        elif len(x_shape) == 4:
            if pad_r > 0 or pad_b > 0:
                x = x[:, :h, :w, :].contiguous()

        return x

    def forward_part2(self, x):
        return self.drop_path(self.mlp(self.norm2(x)))

    def load_from(self, weights, n_block, layer):
        root = f"module.{layer}.0.blocks.{n_block}."
        block_names = [
            "norm1.weight",
            "norm1.bias",
            "attn.relative_position_bias_table",
            "attn.relative_position_index",
            "attn.qkv.weight",
            "attn.qkv.bias",
            "attn.proj.weight",
            "attn.proj.bias",
            "norm2.weight",
            "norm2.bias",
            "mlp.fc1.weight",
            "mlp.fc1.bias",
            "mlp.fc2.weight",
            "mlp.fc2.bias",
        ]
        with torch.no_grad():
            self.norm1.weight.copy_(weights["state_dict"][root + block_names[0]])
            self.norm1.bias.copy_(weights["state_dict"][root + block_names[1]])
            self.attn.relative_position_bias_table.copy_(weights["state_dict"][root + block_names[2]])
            self.attn.relative_position_index.copy_(weights["state_dict"][root + block_names[3]])
            self.attn.qkv.weight.copy_(weights["state_dict"][root + block_names[4]])
            self.attn.qkv.bias.copy_(weights["state_dict"][root + block_names[5]])
            self.attn.proj.weight.copy_(weights["state_dict"][root + block_names[6]])
            self.attn.proj.bias.copy_(weights["state_dict"][root + block_names[7]])
            self.norm2.weight.copy_(weights["state_dict"][root + block_names[8]])
            self.norm2.bias.copy_(weights["state_dict"][root + block_names[9]])
            self.mlp.linear1.weight.copy_(weights["state_dict"][root + block_names[10]])
            self.mlp.linear1.bias.copy_(weights["state_dict"][root + block_names[11]])
            self.mlp.linear2.weight.copy_(weights["state_dict"][root + block_names[12]])
            self.mlp.linear2.bias.copy_(weights["state_dict"][root + block_names[13]])

    def forward(self, x, mask_matrix):
        shortcut = x
        if self.use_checkpoint:
            x = checkpoint.checkpoint(self.forward_part1, x, mask_matrix, use_reentrant=False)
        else:
            x = self.forward_part1(x, mask_matrix)
        x = shortcut + self.drop_path(x)
        if self.use_checkpoint:
            x = x + checkpoint.checkpoint(self.forward_part2, x, use_reentrant=False)
        else:
            x = x + self.forward_part2(x)
        return x



class PatchMergingV2(nn.Module):
    """
    Patch merging layer based on: "Liu et al.,
    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/abs/2103.14030>"
    https://github.com/microsoft/Swin-Transformer
    """

    def __init__(self, dim: int, norm_layer: type[LayerNorm] = nn.LayerNorm, spatial_dims: int = 3) -> None:
        """
        Args:
            dim: number of feature channels.
            norm_layer: normalization layer.
            spatial_dims: number of spatial dims.
        """

        super().__init__()
        self.dim = dim
        if spatial_dims == 3:
            self.reduction = nn.Linear(8 * dim, 2 * dim, bias=False)
            self.norm = norm_layer(8 * dim)
        elif spatial_dims == 2:
            self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
            self.norm = norm_layer(4 * dim)

    def forward(self, x):
        x_shape = x.size()
        if len(x_shape) == 5:
            b, d, h, w, c = x_shape
            pad_input = (h % 2 == 1) or (w % 2 == 1) or (d % 2 == 1)
            if pad_input:
                x = F.pad(x, (0, 0, 0, w % 2, 0, h % 2, 0, d % 2))
            x = torch.cat(
                [x[:, i::2, j::2, k::2, :] for i, j, k in itertools.product(range(2), range(2), range(2))], -1
            )

        elif len(x_shape) == 4:
            b, h, w, c = x_shape
            pad_input = (h % 2 == 1) or (w % 2 == 1)
            if pad_input:
                x = F.pad(x, (0, 0, 0, w % 2, 0, h % 2))
            x = torch.cat([x[:, j::2, i::2, :] for i, j in itertools.product(range(2), range(2))], -1)

        x = self.norm(x)
        x = self.reduction(x)
        return x


class PatchMerging(PatchMergingV2):
    """The `PatchMerging` module previously defined in v0.9.0."""

    def forward(self, x):
        x_shape = x.size()
        if len(x_shape) == 4:
            return super().forward(x)
        if len(x_shape) != 5:
            raise ValueError(f"expecting 5D x, got {x.shape}.")
        b, d, h, w, c = x_shape
        pad_input = (h % 2 == 1) or (w % 2 == 1) or (d % 2 == 1)
        if pad_input:
            x = F.pad(x, (0, 0, 0, w % 2, 0, h % 2, 0, d % 2))
        x0 = x[:, 0::2, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, 0::2, :]
        x3 = x[:, 0::2, 0::2, 1::2, :]
        x4 = x[:, 1::2, 0::2, 1::2, :]
        x5 = x[:, 0::2, 1::2, 0::2, :]
        x6 = x[:, 0::2, 0::2, 1::2, :]
        x7 = x[:, 1::2, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3, x4, x5, x6, x7], -1)
        x = self.norm(x)
        x = self.reduction(x)
        return x

def compute_mask(dims, window_size, shift_size, device):
    """Computing region masks based on: "Liu et al.,
    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/abs/2103.14030>"
    https://github.com/microsoft/Swin-Transformer

     Args:
        dims: dimension values.
        window_size: local window size.
        shift_size: shift size.
        device: device.
    """

    cnt = 0

    if len(dims) == 3:
        d, h, w = dims
        img_mask = torch.zeros((1, d, h, w, 1), device=device)
        for d in slice(-window_size[0]), slice(-window_size[0], -shift_size[0]), slice(-shift_size[0], None):
            for h in slice(-window_size[1]), slice(-window_size[1], -shift_size[1]), slice(-shift_size[1], None):
                for w in slice(-window_size[2]), slice(-window_size[2], -shift_size[2]), slice(-shift_size[2], None):
                    img_mask[:, d, h, w, :] = cnt
                    cnt += 1

    elif len(dims) == 2:
        h, w = dims
        img_mask = torch.zeros((1, h, w, 1), device=device)
        for h in slice(-window_size[0]), slice(-window_size[0], -shift_size[0]), slice(-shift_size[0], None):
            for w in slice(-window_size[1]), slice(-window_size[1], -shift_size[1]), slice(-shift_size[1], None):
                img_mask[:, h, w, :] = cnt
                cnt += 1

    mask_windows = window_partition(img_mask, window_size)
    mask_windows = mask_windows.squeeze(-1)
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

    return attn_mask

class BasicLayer(nn.Module):
    """
    Basic Swin Transformer layer in one stage based on: "Liu et al.,
    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/abs/2103.14030>"
    https://github.com/microsoft/Swin-Transformer
    """

    def __init__(
        self,
        dim: int,
        depth: int,
        num_heads: int,
        window_size: Sequence[int],
        drop_path: list,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        norm_layer: type[LayerNorm] = nn.LayerNorm,
        downsample: Optional[nn.Module] = None,
        use_checkpoint: bool = False,
    ) -> None:
        """
        Args:
            dim: number of feature channels.
            depth: number of layers in each stage.
            num_heads: number of attention heads.
            window_size: local window size.
            drop_path: stochastic depth rate.
            mlp_ratio: ratio of mlp hidden dim to embedding dim.
            qkv_bias: add a learnable bias to query, key, value.
            drop: dropout rate.
            attn_drop: attention dropout rate.
            norm_layer: normalization layer.
            downsample: an optional downsampling layer at the end of the layer.
            use_checkpoint: use gradient checkpointing for reduced memory usage.
        """

        super().__init__()
        self.window_size = window_size
        self.shift_size = tuple(i // 2 for i in window_size)
        self.no_shift = tuple(0 for i in window_size)
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.blocks = nn.ModuleList(
            [
                SwinTransformerBlock(
                    dim=dim,
                    num_heads=num_heads,
                    window_size=self.window_size,
                    shift_size=self.no_shift if (i % 2 == 0) else self.shift_size,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                    norm_layer=norm_layer,
                    use_checkpoint=use_checkpoint,
                )
                for i in range(depth)
            ]
        )

        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer, spatial_dims=len(self.window_size))
        else:
            self.downsample = None

    def forward(self, x):
        x_shape = x.size()        # (4,96,44,52,44)
        if len(x_shape) == 5:
            b, c, d, h, w = x_shape    # d:44 b:4 w:44 h:52 c:96
            window_size, shift_size = get_window_size((d, h, w), self.window_size, self.shift_size)
            x = x.permute(0, 2, 3, 4, 1)   #   (4,44,52,44,96)
            dp = int(np.ceil(d / window_size[0])) * window_size[0]    # dp:49
            hp = int(np.ceil(h / window_size[1])) * window_size[1]    # hp:56
            wp = int(np.ceil(w / window_size[2])) * window_size[2]    # wp:49
            attn_mask = compute_mask([dp, hp, wp], window_size, shift_size, x.device)    #(392,343,343)
            for blk in self.blocks:
                x = blk(x, attn_mask)          # (4,44,52,44,96)
            x = x.view(b, d, h, w, -1)         # (4,44,52,44,96)
            x_after_swin = x
            if self.downsample is not None:
                x = self.downsample(x)         # (4,22,26,22,192)
            x = x.permute(0, 4, 1, 2, 3)       # (4,96,44,52,44)
            x_after_swin = x_after_swin.permute(0, 4, 1, 2, 3)

        elif len(x_shape) == 4:
            b, c, h, w = x_shape
            window_size, shift_size = get_window_size((h, w), self.window_size, self.shift_size)
            x = x.permute(0, 2, 3, 1)
            hp = int(np.ceil(h / window_size[0])) * window_size[0]
            wp = int(np.ceil(w / window_size[1])) * window_size[1]
            attn_mask = compute_mask([hp, wp], window_size, shift_size, x.device)
            for blk in self.blocks:
                x = blk(x, attn_mask)
            x = x.view(b, h, w, -1)
            x_after_swin = x
            if self.downsample is not None:
                x = self.downsample(x)
            x = x.permute(0, 3, 1, 2)
            x_after_swin = x_after_swin.permute(0, 3, 1, 2)
        return x, x_after_swin

class PatchEmbed(nn.Module):
    """
    Patch embedding block based on: "Liu et al.,
    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/abs/2103.14030>"
    https://github.com/microsoft/Swin-Transformer

    Unlike ViT patch embedding block: (1) input is padded to satisfy window size requirements (2) normalized if
    specified (3) position embedding is not used.

    Example::

        >> from monai.networks.blocks import PatchEmbed
        >> PatchEmbed(patch_size=2, in_chans=1, embed_dim=48, norm_layer=nn.LayerNorm, spatial_dims=3)

    """

    def __init__(
        self,
        patch_size: Union[Sequence[int], int] = 2,
        in_chans: int = 1,
        embed_dim: int = 48,
        norm_layer: Type[nn.LayerNorm] = nn.LayerNorm,
        spatial_dims: int = 3,
    ) -> None:
        """
        Args:
            patch_size: dimension of patch size.
            in_chans: dimension of input channels.
            embed_dim: number of linear projection output channels.
            norm_layer: normalization layer.
            spatial_dims: spatial dimension.
        """

        super().__init__()

        if spatial_dims not in (2, 3):
            raise ValueError("spatial dimension should be 2 or 3.")

        if not isinstance(patch_size, (list, tuple)):
            patch_size = (patch_size,) * spatial_dims
        elif len(patch_size) != spatial_dims:
            raise ValueError(f"Sequence must have length {spatial_dims}, got {len(patch_size)}.")
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        if spatial_dims == 3:
            self.proj = nn.Conv3d(
                in_channels=in_chans, out_channels=embed_dim, kernel_size=patch_size, stride=patch_size
            )
        elif spatial_dims == 2:
            self.proj = nn.Conv2d(
                in_channels=in_chans, out_channels=embed_dim, kernel_size=patch_size, stride=patch_size
            )

        # self.proj = Conv[Conv.CONV, spatial_dims](
        #     in_channels=in_chans, out_channels=embed_dim, kernel_size=patch_size, stride=patch_size
        # )
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        x_shape = x.size()
        if len(x_shape) == 5:
            _, _, d, h, w = x_shape
            if w % self.patch_size[2] != 0:
                x = F.pad(x, (0, self.patch_size[2] - w % self.patch_size[2]))
            if h % self.patch_size[1] != 0:
                x = F.pad(x, (0, 0, 0, self.patch_size[1] - h % self.patch_size[1]))
            if d % self.patch_size[0] != 0:
                x = F.pad(x, (0, 0, 0, 0, 0, self.patch_size[0] - d % self.patch_size[0]))

        elif len(x_shape) == 4:
            _, _, h, w = x_shape
            if w % self.patch_size[1] != 0:
                x = F.pad(x, (0, self.patch_size[1] - w % self.patch_size[1]))
            if h % self.patch_size[0] != 0:
                x = F.pad(x, (0, 0, 0, self.patch_size[0] - h % self.patch_size[0]))

        x = self.proj(x)
        if self.norm is not None:
            x_shape = x.size()
            x = x.flatten(2).transpose(1, 2)
            x = self.norm(x)
            if len(x_shape) == 5:
                d, wh, ww = x_shape[2], x_shape[3], x_shape[4]
                x = x.transpose(1, 2).view(-1, self.embed_dim, d, wh, ww)
            elif len(x_shape) == 4:
                wh, ww = x_shape[2], x_shape[3]
                x = x.transpose(1, 2).view(-1, self.embed_dim, wh, ww)
        return x

class PanSwin(nn.Module):
    """
    Swin Transformer based on: "Liu et al.,
    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/abs/2103.14030>"
    https://github.com/microsoft/Swin-Transformer
    """

    def __init__(
        self,
        in_chans: int,
        embed_dim: int,
        window_size: Sequence[int],
        patch_size: Sequence[int],
        depths: Sequence[int],
        num_heads: Sequence[int],
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        norm_layer: type[LayerNorm] = nn.LayerNorm,
        patch_norm: bool = False,
        use_checkpoint: bool = False,
        spatial_dims: int = 3,
        image_size: List[int] = [128, 128, 128]
    ) -> None:
        """
        Args:
            in_chans: dimension of input channels.
            embed_dim: number of linear projection output channels.
            window_size: local window size.
            patch_size: patch size.
            depths: number of layers in each stage.
            num_heads: number of attention heads.
            mlp_ratio: ratio of mlp hidden dim to embedding dim.
            qkv_bias: add a learnable bias to query, key, value.
            drop_rate: dropout rate.
            attn_drop_rate: attention dropout rate.
            drop_path_rate: stochastic depth rate.
            norm_layer: normalization layer.
            patch_norm: add normalization after patch embedding.
            use_checkpoint: use gradient checkpointing for reduced memory usage.
            spatial_dims: spatial dimension.
            downsample: module used for downsampling, available options are `"mergingv2"`, `"merging"` and a
                user-specified `nn.Module` following the API defined in :py:class:`monai.networks.nets.PatchMerging`.
                The default is currently `"merging"` (the original version defined in v0.9.0).
            use_v2: using swinunetr_v2, which adds a residual convolution block at the beginning of each swin stage.
        """

        super().__init__()
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.window_size = window_size
        self.patch_size = patch_size
        self.patch_embed = PatchEmbed(
            patch_size=self.patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None,  # type: ignore
            spatial_dims=spatial_dims,
        )
        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()  # Initialize the norms attribute as an nn.ModuleList
        self.image_size = [dim // 4 for dim in image_size]
        self.token_compression = nn.ModuleList([TokenPacker(
            depth_dim=self.image_size[0]//(2 ** i),
            height_dim=self.image_size[1]//(2 ** i),
            width_dim=self.image_size[2]//(2 ** i),
            embed_dim=self.embed_dim * (2 ** i),
            num_heads=2,
            hidden_size=768,
            scale_factor=2) for i in range(self.num_layers)])

        self.batch_norm_x = nn.ModuleList([
            nn.BatchNorm3d(self.embed_dim * (2 ** i))
            for i in range(1, self.num_layers+1)])

        self.batch_norm_x_packer = nn.ModuleList([
            nn.BatchNorm3d(self.embed_dim * (2 ** i))
            for i in range(1, self.num_layers+1)])

        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=int(embed_dim * 2**i_layer),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=self.window_size,
                drop_path=dpr[sum(depths[:i_layer]) : sum(depths[: i_layer + 1])],
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                norm_layer=norm_layer,
                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                use_checkpoint=use_checkpoint,
            )
            self.layers.append(layer)
            num_features = int(embed_dim * 2**i_layer)
            norm = norm_layer(num_features)
            self.norms.append(norm)

        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        # self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def proj_out(self, x, normalize=False):
        if normalize:
            x_shape = x.size()
            if len(x_shape) == 5:
                n, ch, d, h, w = x_shape
                x = x.permute(0, 2, 3, 4, 1)
                x = F.layer_norm(x, [ch])
                x = x.permute(0, 4, 1, 2, 3)
            elif len(x_shape) == 4:
                n, ch, h, w = x_shape
                x = x.permute(0, 2, 3, 1)
                x = F.layer_norm(x, [ch])
                x = x.permute(0, 3, 1, 2)
        return x

    def transform_output(self, x, norm_layer):
        """
        Transform the output of each layer in the specified way:
        x = x.permute(0, 2, 3, 4, 1)
        x = self.norm(x)
        x = x.permute(0, 4, 1, 2, 3)
        x = x.reshape(x.size(0), x.size(1), -1)
        x = x.permute(0, 2, 1)
        """
        x = x.permute(0, 2, 3, 4, 1)
        x = norm_layer(x)
        # x = self.norm(x)
        x = x.permute(0, 4, 1, 2, 3)
        x = x.reshape(x.size(0), x.size(1), -1)
        x = x.permute(0, 2, 1)
        return x

    def forward(self, x, normalize=True):
        x = self.patch_embed(x)                 # (4,96,44,52,44)
        x = self.pos_drop(x)                    # (4,96,44,52,44)
        x = self.proj_out(x, normalize)         # (4,96,44,52,44)

        layer_outputs = []  # List to store outputs of each layer

        # for layer in self.layers:
        for i, (layer, norm_layer, token_compress,bn_x, bn_x_packer) in enumerate(zip(self.layers, self.norms, self.token_compression, self.batch_norm_x, self.batch_norm_x_packer)):
            x_packer = token_compress(x.permute(0, 2, 3, 4, 1))     #TGIC
            x_packer = x_packer.permute(0, 4, 1, 2, 3)
            x, x_no_merging = layer(x)
            if i < len(self.layers) - 1:                    # GSTB
                x = bn_x(x)
                x_packer = bn_x_packer(x_packer)
                # sample_index = 0
                # x_mean = x[sample_index].mean().item()
                # x_packer_mean = x_packer[sample_index].mean().item()
                # print(f"Iteration {i}: x_mean = {x_mean}, x_packer_mean = {x_packer_mean}")
                x = x_packer + x
            # (4,96,44,52,44)->(4,192,22,26,22)->(4,384,11,13,11)->(4,768,6,7,6)
            transformed_x = self.transform_output(x_no_merging, norm_layer)  # Transform each layer's output
            layer_outputs.append(transformed_x)  # Store the transformed output of each layer

        return layer_outputs
        # return x_no_merging


def test_swin_transformer():
    in_chans = 1
    embed_dim = 96
    window_size = (7, 7, 7)
    patch_size = (4, 4, 4)
    depths = [2, 2, 6, 2]
    num_heads = [3, 6, 12, 24]
    mlp_ratio = 4.0
    qkv_bias = True
    drop_rate = 0.0
    attn_drop_rate = 0.0
    drop_path_rate = 0.1
    norm_layer = nn.LayerNorm
    patch_norm = True
    use_checkpoint = False
    spatial_dims = 3
    image_size = [128, 128, 128]

    model = PanSwin(
        in_chans=in_chans,
        embed_dim=embed_dim,
        window_size=window_size,
        patch_size=patch_size,
        depths=depths,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        qkv_bias=qkv_bias,
        drop_rate=drop_rate,
        attn_drop_rate=attn_drop_rate,
        drop_path_rate=drop_path_rate,
        norm_layer=norm_layer,
        patch_norm=patch_norm,
        use_checkpoint=use_checkpoint,
        spatial_dims=spatial_dims,
        image_size=image_size
    )

    # print(model)

    input_data = torch.randn(4, 1, 128, 128, 128)

    output = model(input_data)

    # print("Output shape:", output.shape)

if __name__ == "__main__":
    test_swin_transformer()