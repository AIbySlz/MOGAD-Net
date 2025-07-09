import torch
import torch.nn as nn
from functools import partial
import numpy as np
from torch.nn.init import trunc_normal_
from torch.nn import functional as F
import math
import unittest

class TGIC(nn.Module):
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

        modules = [nn.Linear(embed_dim, hidden_size)]
        for _ in range(1, 2):
            modules.append(nn.GELU())
            modules.append(nn.Linear(hidden_size, hidden_size))
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

        x_cpu = x.detach().cpu()
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
        for i in range(1, 4):
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
        return x

    def to(self, device):
        return super().to(device)

class TGICBranches:
    def __init__(self, feature_list, num_heads=8, hidden_size=None, scale_factor=2):
        self.num_heads = num_heads
        self.scale_factor = scale_factor
        self.token_packers = [self.create_token_packer_branch(feature.shape, self.num_heads, hidden_size, self.scale_factor) for feature in feature_list]

    def create_token_packer_branch(self, shape, num_heads, hidden_size, scale_factor):
        if hidden_size is None:
            hidden_size = shape[4]
        return TGIC(depth_dim=shape[1], height_dim=shape[2], width_dim=shape[3], embed_dim=shape[4], num_heads=num_heads, hidden_size=hidden_size, scale_factor=scale_factor)

    def process_data(self, data_list):
        results = []
        for token_packer, data in zip(self.token_packers, data_list):
            result = token_packer(data)
            results.append(result)
        return results

    def to(self, device):
        for token_packer in self.token_packers:
            token_packer.to(device)
        return self

    def state_dict(self):
        state_dict = {}
        for idx, token_packer in enumerate(self.token_packers):
            state_dict[f'token_packer_{idx}'] = token_packer.state_dict()
        return state_dict

    def load_state_dict(self, state_dict):
        for idx, token_packer in enumerate(self.token_packers):
            token_packer.load_state_dict(state_dict[f'token_packer_{idx}'])

class ComputeQKV(nn.Module):
    def __init__(self, qkv):
        super(ComputeQKV, self).__init__()
        self.device = 'cpu'
        self.D_list = [tensor.shape[-1] for tensor in qkv]
        self.q_linear_list = nn.ModuleList([nn.Linear(D, D) for D in self.D_list])
        self.k_linear_list = nn.ModuleList([nn.Linear(D, D) for D in self.D_list])
        self.v_linear_list = nn.ModuleList([nn.Linear(D, D) for D in self.D_list])

    def to(self, device):
        self.device = device
        self.q_linear_list.to(device)
        self.k_linear_list.to(device)
        self.v_linear_list.to(device)
        return self

    def process(self, concat_bhg):
        Q_list = []
        K_list = []
        V_list = []

        for i, tensor in enumerate(concat_bhg):
            tensor = tensor.to(self.device)

            Q = self.q_linear_list[i](tensor)
            K = self.k_linear_list[i](tensor)
            V = self.v_linear_list[i](tensor)

            Q_list.append(Q)
            K_list.append(K)
            V_list.append(V)

            del tensor
            torch.cuda.empty_cache()

        return Q_list, K_list, V_list

class MultiheadAttentionProcessor:
    def __init__(self, num_heads, dropout_p=None):
        self.num_heads = num_heads
        self.dropout_p = dropout_p if dropout_p is not None else 0.0

    def process(self, q_list, k_list, v_list):
        attn_output_list = []
        for i in range(len(q_list)):
            Q = q_list[i]
            K = k_list[i]
            V = v_list[i]

            embed_dim = Q.size(-1)
            multihead_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=self.num_heads,
                                                   dropout=self.dropout_p, batch_first=True).to(Q.device)

            attn_output, _ = multihead_attn(Q, K, V)
            attn_output_list.append(attn_output)
            del Q, K, V, multihead_attn
            torch.cuda.empty_cache()

        return attn_output_list

    def to(self, device):
        self.device = device
        return self


class FCLayer(nn.Module):
    def __init__(self, feature_list):
        super(FCLayer, self).__init__()
        self.fc_list = [self.create_fc_branch(feature.shape) for feature in feature_list]

    def create_fc_branch(self, shape):
        token_num = shape[1]
        fc = nn.Linear(token_num, 1)
        return fc

    def process_data(self, data_list):
        results = []
        for fc, data in zip(self.fc_list, data_list):
            data = data.permute(0, 2, 1)
            result = fc(data)
            result = torch.flatten(result, 1)
            results.append(result)
        return results

    def to(self, device):
        for fc in self.fc_list:
            fc.to(device)
        return self

class TestTGIC(unittest.TestCase):
    def setUp(self):
        self.model = TGIC(depth_dim=20,
            height_dim=25,
            width_dim=31,
            embed_dim=768,
            num_heads=8,
            hidden_size=768,
            scale_factor=2,
            norm_layer=partial(nn.LayerNorm, eps=1e-6))

    def test_forward(self):
        # Create dummy input data
        depth_dim = 20
        height_dim = 25
        width_dim = 31
        batch_size = 10
        embed_dim = 768

        x_single_level = torch.randn(batch_size, depth_dim, height_dim, width_dim, embed_dim)

        # Forward pass
        output = self.model(x_single_level)

        print(output.shape)

if __name__ == '__main__':
    unittest.main()
