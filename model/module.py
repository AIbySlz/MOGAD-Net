import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
from model.PanSwin import PanSwin
from typing import List


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, normalize=True):
        super(MLP, self).__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        self.normalize = normalize

    def forward(self, x):
        x = self.projection(x)
        if self.normalize:
            x = nn.functional.normalize(x, dim=1)  # L2 normalize
        return x

class ClipLoss(nn.Module):
    def __init__(self, input_dim, hid_dim, temperature=0.1):
        super(ClipLoss, self).__init__()
        self.map1 = nn.Linear(input_dim, hid_dim)
        self.map2 = nn.Linear(input_dim, hid_dim)
        self.map3 = nn.Linear(input_dim, hid_dim)
        self.temperature = temperature

    def forward(self, feature1, feature2, feature3, device):
        # Move input features and labels to the specified device
        feature1 = feature1.to(device)
        feature2 = feature2.to(device)
        feature3 = feature3.to(device)
        labels = torch.arange(feature1.shape[0], device=device, dtype=torch.long)

        # Perform linear mapping
        feature1 = self.map1(feature1).to(device)
        feature2 = self.map2(feature2).to(device)
        feature3 = self.map3(feature3).to(device)

        # Perform normalization
        feature1 = F.normalize(feature1, p=2, dim=1)
        feature2 = F.normalize(feature2, p=2, dim=1)
        feature3 = F.normalize(feature3, p=2, dim=1)

        # Calculate similarity matrix
        similarity_matrix1 = torch.matmul(feature1, feature2.T) / self.temperature
        similarity_matrix2 = torch.matmul(feature2, feature1.T) / self.temperature

        similarity_matrix3 = torch.matmul(feature1, feature3.T) / self.temperature
        similarity_matrix4 = torch.matmul(feature3, feature1.T) / self.temperature

        # Calculate loss
        loss_fn = nn.CrossEntropyLoss()
        loss1 = loss_fn(similarity_matrix1, labels)
        loss2 = loss_fn(similarity_matrix2, labels)
        loss3 = loss_fn(similarity_matrix3, labels)
        loss4 = loss_fn(similarity_matrix4, labels)
        loss = (loss1+loss2+loss3+loss4) / 2

        return loss


class ClipLossLabel(nn.Module):
    def __init__(self, input_dim, hid_dim, temperature=0.1):
        super(ClipLossLabel, self).__init__()
        self.map1 = nn.Linear(input_dim, hid_dim)
        self.map2 = nn.Linear(input_dim, hid_dim)
        self.map3 = nn.Linear(input_dim, hid_dim)
        self.temperature = temperature

    def forward(self, feature1, feature2, feature3, labels, device):
        # Move input features and labels to the specified device
        feature1 = feature1.to(device)
        feature2 = feature2.to(device)
        feature3 = feature3.to(device)
        labels = labels.to(device)

        # Perform linear mapping
        feature1 = self.map1(feature1).to(device)
        feature2 = self.map2(feature2).to(device)
        feature3 = self.map3(feature3).to(device)

        # Perform normalization
        feature1 = F.normalize(feature1, p=2, dim=1)
        feature2 = F.normalize(feature2, p=2, dim=1)
        feature3 = F.normalize(feature3, p=2, dim=1)

        # Concatenate features along dimension 0
        concatenated_features = torch.cat((feature1, feature2, feature3), dim=1)

        # Calculate similarity matrix
        similarity_matrix = torch.matmul(concatenated_features, concatenated_features.T) / self.temperature

        # Create label matrix
        labels_matrix = (labels.unsqueeze(0) == labels.unsqueeze(1)).float().to(device)

        # Calculate loss
        loss_fn = nn.CrossEntropyLoss()  # Using binary cross entropy loss
        loss = loss_fn(similarity_matrix, labels_matrix)

        return loss


class ClipLossLabelSingal(nn.Module):
    def __init__(self, input_dim, hid_dim, temperature=0.1):
        super(ClipLossLabelSingal, self).__init__()
        self.map1 = nn.Linear(input_dim, hid_dim)
        self.temperature = temperature

    def forward(self, feature1, labels, device):
        # Move input features and labels to the specified device
        feature1 = feature1.to(device)
        labels = labels.to(device)

        # Perform linear mapping
        feature1 = self.map1(feature1).to(device)

        # Perform normalization
        feature1 = F.normalize(feature1, p=2, dim=1)

        # Calculate similarity matrix
        similarity_matrix = torch.matmul(feature1, feature1.T) / self.temperature

        # Create label matrix
        labels_matrix = (labels.unsqueeze(0) == labels.unsqueeze(1)).float().to(device)

        # Calculate loss
        loss_fn = nn.CrossEntropyLoss()  # Using binary cross entropy loss
        loss = loss_fn(similarity_matrix, labels_matrix)

        return loss


# Define the Multi_organ_PanSwin class
class Multi_organ_PanSwin(nn.Module):
    def __init__(self, in_chans=1, embed_dim=96, window_size=[4,4,4], patch_size=[4,4,4], depths=[2,2,2,2], num_heads=[2,4,8,16], mlp_ratio=4,
                 qkv_bias=True, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1, brain_size: List[int]=[128,128,128], heart_size: List[int]=[128,96,96], gut_size: List[int]=[224,160,352], patch_norm=True, use_checkpoint=False, spatial_dims=3):
        super(Multi_organ_PanSwin, self).__init__()
        # Model configurations
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.window_size = window_size
        self.patch_size = patch_size
        self.depths = depths
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias
        self.drop_rate = drop_rate
        self.attn_drop_rate = attn_drop_rate
        self.drop_path_rate = drop_path_rate
        self.patch_norm = patch_norm
        self.use_checkpoint = use_checkpoint
        self.spatial_dims = spatial_dims
        self.brain_size = brain_size
        self.heart_size = heart_size
        self.gut_size = gut_size

        # Define Swin Transformer
        self.brain_swin_transformer = PanSwin(
            in_chans=self.in_chans,
            embed_dim=self.embed_dim,
            window_size=self.window_size,
            patch_size=self.patch_size,
            depths=self.depths,
            num_heads=self.num_heads,
            mlp_ratio=self.mlp_ratio,
            qkv_bias=self.qkv_bias,
            drop_rate=self.drop_rate,
            attn_drop_rate=self.attn_drop_rate,
            drop_path_rate=self.drop_path_rate,
            patch_norm=self.patch_norm,
            use_checkpoint=self.use_checkpoint,
            spatial_dims=self.spatial_dims,
            image_size=self.brain_size)

        self.heart_swin_transformer = PanSwin(
            in_chans=self.in_chans,
            embed_dim=self.embed_dim,
            window_size=self.window_size,
            patch_size=self.patch_size,
            depths=self.depths,
            num_heads=self.num_heads,
            mlp_ratio=self.mlp_ratio,
            qkv_bias=self.qkv_bias,
            drop_rate=self.drop_rate,
            attn_drop_rate=self.attn_drop_rate,
            drop_path_rate=self.drop_path_rate,
            patch_norm=self.patch_norm,
            use_checkpoint=self.use_checkpoint,
            spatial_dims=self.spatial_dims,
            image_size=self.heart_size)

        self.gut_swin_transformer = PanSwin(
            in_chans=self.in_chans,
            embed_dim=self.embed_dim,
            window_size=self.window_size,
            patch_size=self.patch_size,
            depths=self.depths,
            num_heads=self.num_heads,
            mlp_ratio=self.mlp_ratio,
            qkv_bias=self.qkv_bias,
            drop_rate=self.drop_rate,
            attn_drop_rate=self.attn_drop_rate,
            drop_path_rate=self.drop_path_rate,
            patch_norm=self.patch_norm,
            use_checkpoint=self.use_checkpoint,
            spatial_dims=self.spatial_dims,
            image_size=self.gut_size)

    def forward(self, x_brain, x_heart, x_gut):
        hierarchical_brain_out = self.brain_swin_transformer(x_brain)
        hierarchical_heart_out = self.heart_swin_transformer(x_heart)
        hierarchical_gut_out = self.gut_swin_transformer(x_gut)

        return hierarchical_brain_out, hierarchical_heart_out, hierarchical_gut_out


class TransformerClassifier(nn.Module):
    def __init__(self, brain_num_features=768*3, transformer_encoder_head=8, transformer_encoder_dim_feedforward=128, transformer_encoder_dropout=0.1, transformer_encoder_activation='relu', transformer_encoder_num_layers=1, num_classes=2):
        super(TransformerClassifier, self).__init__()
        self.brain_num_features = brain_num_features
        self.transformer_encoder_head = transformer_encoder_head
        self.transformer_encoder_dim_feedforward = transformer_encoder_dim_feedforward
        self.transformer_encoder_dropout = transformer_encoder_dropout
        self.transformer_encoder_activation = transformer_encoder_activation
        self.transformer_encoder_num_layers = transformer_encoder_num_layers
        self.num_classes = num_classes

        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.brain_num_features,
            nhead=self.transformer_encoder_head,
            dim_feedforward=self.transformer_encoder_dim_feedforward,
            dropout=self.transformer_encoder_dropout,
            activation=self.transformer_encoder_activation,
            batch_first=True
        )

        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=self.transformer_encoder_num_layers)
        self.classifier = nn.Linear(self.brain_num_features, self.num_classes)

    def forward(self, brain_out, heart_out, gut_out):
        # Fusion output
        brain_heart_gut_cat = torch.cat((brain_out, heart_out, gut_out), dim=1)
        fusion_out = self.transformer_encoder(brain_heart_gut_cat)
        fusion_logits = self.classifier(fusion_out)
        return fusion_logits

def PatchFlatten(image, patch_size):
    """
    Args:
    - image (np.ndarray or torch.Tensor): Input 3D image tensor with shape (batch, channel, D, H, W)
    - patch_size (tuple): Patch size, e.g. (4, 4, 4)

    Returns:
    - patches (np.ndarray or torch.Tensor): Result of replacing original image positions with patches, with shape (batch, num_patches, patch_elements)
    """
    B, C, D, H, W = image.shape
    PD, PH, PW = patch_size

    # Ensure patch size is divisible by image dimensions
    assert D % PD == 0 and H % PH == 0 and W % PW == 0, \
        "Patch size must be divisible by image dimensions"

    # Use tensor operations instead of loops
    if isinstance(image, np.ndarray):
        patches_matrix = image.reshape(B, C, D // PD, PD, H // PH, PH, W // PW, PW)
        patches_matrix = patches_matrix.transpose(0, 2, 4, 6, 1, 3, 5, 7)
        patches_matrix = patches_matrix.reshape(B, D // PD, H // PH, W // PW, PD * PH * PW)
    else:
        patches_matrix = image.view(B, C, D // PD, PD, H // PH, PH, W // PW, PW)
        patches_matrix = patches_matrix.permute(0, 2, 4, 6, 1, 3, 5, 7).contiguous()
        patches_matrix = patches_matrix.view(B, D // PD, H // PH, W // PW, PD * PH * PW)

    return patches_matrix


def TokenFlatten(x):
    """
    Function to perform patch merging.

    Args:
        x (torch.Tensor): Input tensor with shape either (B, D, H, W, C) or (B, H, W, C).

    Returns:
        torch.Tensor: The processed tensor after patch merging.
    """

    x_shape = x.size()

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

    # B,D,H,W,C to B,num_patches,patch_elements
    B, new_d, new_h, new_w, new_c = x.size()
    patch_elements = new_c
    num_patches = new_d * new_h * new_w

    x_matrix = x.view(B, new_d, new_h, new_w, patch_elements)

    return x_matrix


def process_patches(image_tensor, patch_dimension, depths):
    patches_matrix = PatchFlatten(image_tensor, patch_dimension)
    patches_matrix_list = [patches_matrix]

    num_merges = len(depths)  # Determine the number of merging steps based on the length of tensors

    for _ in range(num_merges - 1):  # The -1 is because patches1 is already created
        new_patches_matrix = TokenFlatten(patches_matrix_list[-1])
        patches_matrix_list.append(new_patches_matrix)

    return patches_matrix_list

def feature_transformation(x, avgpool):
    x = x.permute(0, 2, 1)  # B C D -> B D C
    x = avgpool(x)  # B D C -> B D 1
    x = torch.flatten(x, 1)  # B D 1 -> B D
    return x

def find_minimal_bounding_cube(matrix_sum, data_matrix):
    """
    find_minimal_bounding_cube
    Args:
        matrix_sum (torch.Tensor): Input tensor with shape (batch_size, depth, height, width).
        data_matrix (torch.Tensor): Tensor to be cropped with shape (batch_size, depth, height, width, channels).
    Returns:
        torch.Tensor: Cropped data_matrix.
    """

    # Find range of non-zero elements in depth dimension
    depth_mask = torch.any(matrix_sum != 0, dim=(0, 2, 3))  # Get mask in depth dimension
    min_d, max_d = torch.nonzero(depth_mask).min().item(), torch.nonzero(depth_mask).max().item()

    # Find range of non-zero elements in height dimension
    height_mask = torch.any(matrix_sum != 0, dim=(0, 1, 3))  # Get mask in height dimension
    min_h, max_h = torch.nonzero(height_mask).min().item(), torch.nonzero(height_mask).max().item()

    # Find range of non-zero elements in width dimension
    width_mask = torch.any(matrix_sum != 0, dim=(0, 1, 2))  # Get mask in width dimension
    min_w, max_w = torch.nonzero(width_mask).min().item(), torch.nonzero(width_mask).max().item()

    data_matrix = data_matrix[:, min_d:max_d + 1, min_h:max_h + 1, min_w:max_w + 1, :]

    return data_matrix

def find_minimal_bounding_cube(matrix_sum, scale_factor):
    padding_needed = [0] * matrix_sum.ndim
    for i in range(1, 4):
        remainder = matrix_sum.shape[i] % scale_factor
        if remainder == 0:
            padding_needed[i] = 0
        else:
            padding_needed[i] = scale_factor - remainder

    padding = [0, 0, 0, 0, 0, 0, 0, 0]
    padding[1] = padding_needed[3]
    padding[3] = padding_needed[2]
    padding[5] = padding_needed[1]
    matrix_sum = F.pad(matrix_sum, padding, "constant", 0)
    batch_size, depth, height, width = matrix_sum.shape
    matrix_sum_reshaped = matrix_sum.view(batch_size, math.ceil(depth/scale_factor), scale_factor, math.ceil(height/scale_factor), scale_factor, math.ceil(width/scale_factor), scale_factor)
    matrix_sum_permuted = matrix_sum_reshaped.permute(0, 1, 3, 5, 2, 4, 6)
    cube_size = scale_factor * scale_factor * scale_factor
    matrix_sum_permuted = matrix_sum_permuted.reshape(batch_size, math.ceil(depth/scale_factor), math.ceil(height/scale_factor), math.ceil(width/scale_factor), cube_size)
    matrix_sum_permuted = torch.sum(matrix_sum_permuted, dim=-1)
    nonzero_indices = torch.nonzero(matrix_sum_permuted, as_tuple=True)
    min_d, max_d = nonzero_indices[1].min().item(), nonzero_indices[1].max().item()
    min_h, max_h = nonzero_indices[2].min().item(), nonzero_indices[2].max().item()
    min_w, max_w = nonzero_indices[3].min().item(), nonzero_indices[3].max().item()
    data_matrix = torch.zeros_like(matrix_sum_permuted[0])
    data_matrix[min_d:max_d + 1, min_h:max_h + 1, min_w:max_w + 1] = 1
    data_matrix = data_matrix.reshape(data_matrix.numel(), 1)
    return data_matrix


class Logger():
    def __init__(self, log_dir, log_name='log.txt'):
        self.log_name = os.path.join(log_dir, log_name)
        with open(self.log_name, "a") as log_file:
            log_file.write(f'================ {self.log_name} ================\n')

    def print_message(self, msg):
        print(msg, flush=True)
        with open(self.log_name, 'a') as log_file:
            log_file.write('%s\n' % msg)

    def print_message_nocli(self, msg):
        with open(self.log_name, 'a') as log_file:
            log_file.write('%s\n' % msg)