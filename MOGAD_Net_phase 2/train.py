import argparse
import datetime
import random
import textwrap
import os
import gc
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import matplotlib

matplotlib.use('Agg')
from matplotlib import pyplot as plt
from model.PanSwin import PanSwin
from dataloader.GeneralDataset_whole_body_and_brain import GeneralDataset, custom_collate_fn
from dataloader.GeneralDataset_brain_data import GeneralDataset_brain
from pycm import ConfusionMatrix
from monai.data import decollate_batch
from monai.metrics import CumulativeAverage, ROCAUCMetric
from monai.transforms import Compose, Activations, AsDiscrete
from model.TGIC import TGICBranches, ComputeQKV
from model.module import MLP, Multi_organ_PanSwin, TransformerClassifier, process_patches, find_minimal_bounding_cube, Logger

# Argument parser
parser = argparse.ArgumentParser(description='Swin Transformer Training')
parser.add_argument("--feats_path", default=r"D:\DATA\MNI\FDG_crop1.5\crop_pth", type=str, help="dataset directory")
parser.add_argument("--fold_path_brain", default=r"D:\DATA\MNI\FDG_fold_partition\Brain_Labeled\5fold\42", type=str, help="dataset pdf directory")
parser.add_argument("--fold_path_bhg", default=r"D:\DATA\MNI\FDG_fold_partition\Whole_Body_Labeled\5fold\42", type=str, help="dataset pdf directory")
parser.add_argument("--output", type=str, default=r'D:\DATA\Output\Alignment_Classification_BHG', help="Use gradient checkpointing for reduced memory usage")
parser.add_argument("--save_file_name", default=datetime.datetime.now().strftime("%Y-%m%d-%H%M"), type=str, help="folder name to save subject")
parser.add_argument("--bhg_checkpoint_path", type=str, default=r'D:\DATA\Output\Alignment_Classification_BHG\fold1_best_model_3028762.ckpt', help="Path to BHG hierarchical model checkpoint")
parser.add_argument("--brain_checkpoint_path", type=str, default=r'D:\DATA\Output\Brain_Pretrain\FDG_SwinToken_2025-0309-1625\fold_1_lr_4.9999999999999996e-06_wd_0.001_eh_0.9\best_brain_classifer_pretrain_model_fold1_2025-0309-1625.ckpt', help="Path to brain model checkpoint")
parser.add_argument("--lr", default=5e-6, type=float, help="learning rate")
parser.add_argument("--wd", default=1e-3, type=float, help="weight decay")
parser.add_argument("--enhance_ps", default=0.7, type=float, help="enhance probability")
parser.add_argument("--fold_index", default=1, type=int, help="current fold_index")
parser.add_argument("--batch_size", default=8, type=int, help="number of batch size")
parser.add_argument("--brain_size", default=[128, 128, 128], type=int, nargs='+', help="window_size")
parser.add_argument("--heart_size", default=[128, 96, 96], type=int, nargs='+', help="window_size")
parser.add_argument("--gut_size", default=[224, 160, 352], type=int, nargs='+', help="window_size")
parser.add_argument("--workers", default=4, type=int, help="number of workers")
parser.add_argument("--in_chans", default=1, type=int, help="in_chans")
parser.add_argument("--num_classes", default=2, type=int, help="num_classes")
parser.add_argument("--window_size", default=[4, 4, 4], type=int, nargs='+', help="window_size")
parser.add_argument("--patch_size", default=[4, 4, 4], type=int, nargs='+', help="patch_size")
parser.add_argument("--embed_dim", default=96, type=int, help="embed_dim")
parser.add_argument("--depths", default=[2, 2, 2, 2], type=int, nargs='+', help="depths")
parser.add_argument("--num_heads", default=[2, 4, 8, 16], type=int, nargs='+', help="num_heads")
parser.add_argument("--mlp_ratio", default=4, type=int, help="mlp_ratio")
parser.add_argument("--qkv_bias", default=True, type=bool, help="qkv_bias")
parser.add_argument("--drop_rate", default=0., type=float, help="drop_rate")
parser.add_argument("--attn_drop_rate", default=0., type=float, help="attn_drop_rate")
parser.add_argument("--drop_path_rate", default=0.1, type=float, help="drop_path_rate")
parser.add_argument("--patch_norm", default=True, type=bool, help="patch_norm")
parser.add_argument("--use_checkpoint", default=False, type=bool, help="Use gradient checkpointing for reduced memory usage")
parser.add_argument("--spatial_dims", default=3, type=int, help="spatial_dims")
# Added hyperparameters to parser

args = parser.parse_args()

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def save_checkpoint(state, filename):
    torch.save(state, filename)


def load_checkpoint(filepath, bhg_hierarchical_model, multi_classifier_head, optimizer, clip_criterion, device):
    if os.path.isfile(filepath):
        print(f"Loading checkpoint from '{filepath}'")
        checkpoint = torch.load(filepath, map_location=device)
        bhg_hierarchical_model.load_state_dict(checkpoint['bhg_hierarchical_model'])
        multi_classifier_head.load_state_dict(checkpoint['multi_classifier_head'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        clip_criterion_states = checkpoint['clip_criterion']
        for criterion, state in zip(clip_criterion, clip_criterion_states):
            criterion.load_state_dict(state)
        best_test_auc = checkpoint['best_test_auc']
        return best_test_auc
    else:
        print(f"No checkpoint found at '{filepath}'")
        return None


def plot(data, dir, organ_name_type, image_name):
    try:
        x = list(range(len(data)))
        plt.plot(x, data, label=organ_name_type)
        plt.xlabel('epoch')
        # plt.ylabel(organ_name_type)
        title = organ_name_type
        plt.title(textwrap.fill(title, width=70))
        plt.xlim(0, len(data))
        plt.legend(loc='best')
        plt.savefig(os.path.join(dir, image_name + '.png'))
        plt.close()
    except Exception as e:
        print(e)

def print_cuda_memory_usage(prefix=""):
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    print(f"{prefix} CUDA Memory: Allocated={allocated:.2f}GB, Reserved={reserved:.2f}GB")

# Training for an epoch
def train_epoch(epoch, model, brain_model, brain_classifier_head, dataloader, optimizer, brain_packer, heart_packer, gut_packer, qkv, multihead_atten, head_mlp, device, logger, patch_size, embed_dim, depths):
    for param in model.parameters():
        param.requires_grad = False
    model.train()
    brain_model.train()
    brain_classifier_head.train()
    running_loss = 0.0
    label_pred = []
    label_real = []
    loss_epoch = CumulativeAverage()
    AUC = ROCAUCMetric(average='macro')
    post_pred = Compose([Activations(softmax=True)])
    post_label = Compose([AsDiscrete(to_onehot=args.num_classes)])

    for batch_data in dataloader:
        feats_brain = [data['brain'] for data in batch_data if not data['is_multimodal']]
        labels = [data['label'] for data in batch_data if not data['is_multimodal']]
        multi_feats_brain = [data['brain'] for data in batch_data if data['is_multimodal']]
        multi_feats_heart = [data['heart'] for data in batch_data if data['is_multimodal']]
        multi_feats_gut = [data['gut'] for data in batch_data if data['is_multimodal']]
        multi_labels = [data['label'] for data in batch_data if data['is_multimodal']]
        feats_brain = torch.stack(feats_brain).to(device)
        labels = torch.stack(labels).to(device)
        multi_feats_brain = torch.stack(multi_feats_brain).to(device)
        multi_feats_heart = torch.stack(multi_feats_heart).to(device)
        multi_feats_gut = torch.stack(multi_feats_gut).to(device)
        multi_labels = torch.stack(multi_labels).to(device)

        # Brain-Heart-Gut
        with torch.no_grad():
            raw_brain_out, raw_heart_out, raw_gut_out = model(multi_feats_brain, multi_feats_heart, multi_feats_gut)
        feats_brain_mask = (multi_feats_brain > 0).int()
        feats_heart_mask = (multi_feats_heart > 0).int()
        feats_gut_mask = (multi_feats_gut > 0).int()
        feats_brain_mask_list = process_patches(feats_brain_mask, patch_size, depths)
        feats_heart_mask_list = process_patches(feats_heart_mask, patch_size, depths)
        feats_gut_mask_list = process_patches(feats_gut_mask, patch_size, depths)
        feats_brain_mask_list = [torch.sum(brain_mask, dim=-1) for brain_mask in feats_brain_mask_list]
        feats_heart_mask_list = [torch.sum(heart_mask, dim=-1) for heart_mask in feats_heart_mask_list]
        feats_gut_mask_list = [torch.sum(gut_mask, dim=-1) for gut_mask in feats_gut_mask_list]
        batch, brain_depth, brain_height, brain_width = feats_brain_mask_list[0].shape
        batch, heart_depth, heart_height, heart_width = feats_heart_mask_list[0].shape
        batch, gut_depth, gut_height, gut_width = feats_gut_mask_list[0].shape

        raw_brain_shape = [raw_brain.reshape(batch, brain_depth // (2 ** i), brain_height // (2 ** i), brain_width // (2 ** i), embed_dim * (2 ** i)) for i, raw_brain in enumerate(raw_brain_out)]
        raw_heart_shape = [raw_heart.reshape(batch, heart_depth // (2 ** i), heart_height // (2 ** i), heart_width // (2 ** i), embed_dim * (2 ** i)) for i, raw_heart in enumerate(raw_heart_out)]
        raw_gut_shape = [raw_gut.reshape(batch, gut_depth // (2 ** i), gut_height // (2 ** i), gut_width // (2 ** i), embed_dim * (2 ** i)) for i, raw_gut in enumerate(raw_gut_out)]
        raw_brain_shape_mask = [find_minimal_bounding_cube(brain_mask, scale_factor=2) for brain_mask in feats_brain_mask_list]
        raw_heart_shape_mask = [find_minimal_bounding_cube(heart_mask, scale_factor=2) for heart_mask in feats_heart_mask_list]
        raw_gut_shape_mask = [find_minimal_bounding_cube(gut_mask, scale_factor=2) for gut_mask in feats_gut_mask_list]
        del batch_data, raw_brain_out, raw_heart_out, raw_gut_out, feats_brain_mask, feats_heart_mask, feats_gut_mask, feats_brain_mask_list, feats_heart_mask_list, feats_gut_mask_list
        torch.cuda.empty_cache()

        raw_brain_shape = brain_packer.process_data(raw_brain_shape)
        raw_heart_shape = heart_packer.process_data(raw_heart_shape)
        raw_gut_shape = gut_packer.process_data(raw_gut_shape)
        raw_brain_shape = [brain_shape[:, mask.squeeze(-1).bool(), :] for brain_shape, mask in zip(raw_brain_shape, raw_brain_shape_mask)]
        raw_heart_shape = [brain_shape[:, mask.squeeze(-1).bool(), :] for brain_shape, mask in zip(raw_heart_shape, raw_heart_shape_mask)]
        raw_gut_shape = [brain_shape[:, mask.squeeze(-1).bool(), :] for brain_shape, mask in zip(raw_gut_shape, raw_gut_shape_mask)]
        concat_bhg = [torch.cat((brain, heart, gut), dim=1) for brain, heart, gut in zip(raw_brain_shape, raw_heart_shape, raw_gut_shape)]
        q_list, k_list, v_list = qkv.process(concat_bhg)
        bhg_feature = [multihead(q, k, v)[0] for q, k, v, multihead in zip(q_list, k_list, v_list, multihead_atten)]
        multi_batch_size = bhg_feature[0].shape[0]
        del concat_bhg, q_list, k_list, v_list, raw_brain_shape, raw_heart_shape, raw_gut_shape, raw_brain_shape_mask, raw_heart_shape_mask, raw_gut_shape_mask
        torch.cuda.empty_cache()

        # brain
        feats_brain = torch.cat((multi_feats_brain, feats_brain), dim=0)
        labels = torch.cat((multi_labels, labels), dim=0)
        mciCount = torch.sum(labels == 1).item()
        ncCount = torch.sum(labels == 0).item()
        mciweight = ncCount / (mciCount + ncCount)
        ncweight = mciCount / (mciCount + ncCount)
        weights = [mciweight, ncweight]
        weights = [w if w != 0.0 else 1e-4 for w in weights]
        class_weights = torch.FloatTensor(weights).to(device)
        unibrain_out = brain_model(feats_brain)
        multi_brain_feature = [brain_mask[:multi_batch_size, :, :] for brain_mask in unibrain_out]
        multi_brain_feature = [mlp(brain_feature) for mlp, brain_feature in zip(head_mlp, multi_brain_feature)]
        unibrain_out = unibrain_out[-1]
        unibrain_out = unibrain_out.permute(0, 2, 1)
        adaptive_pool = nn.AdaptiveAvgPool1d(1)
        unibrain_out = adaptive_pool(unibrain_out)
        unibrain_out = torch.flatten(unibrain_out, 1)
        unibrain_out = brain_classifier_head(unibrain_out)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        brain_loss = criterion(unibrain_out, labels)
        bhg_feature = [bhg.mean(dim=1).squeeze(1) for bhg in bhg_feature]
        multi_brain_feature = [bhg.mean(dim=1).squeeze(1) for bhg in multi_brain_feature]
        bhg_feature_l2 = [torch.norm(bhg, p=2, dim=1, keepdim=True) for bhg in bhg_feature]
        multi_brain_feature_l2 = [torch.norm(brain, p=2, dim=1, keepdim=True) for brain in multi_brain_feature]
        bhg_feature = [feature / l2 for feature, l2 in zip(bhg_feature, bhg_feature_l2)]
        multi_brain_feature = [feature / l2 for feature, l2 in zip(multi_brain_feature, multi_brain_feature_l2)]
        feature_difference = [bhg - multi for bhg, multi in zip(bhg_feature, multi_brain_feature)]
        l2_distances = [torch.norm(feature, p=2, dim=1) for feature in feature_difference]
        sums_of_l2_distances = [torch.sum(distance).item() for distance in l2_distances]
        multipliers = [0.1, 0.2, 0.3, 0.4]
        sums_of_l2_distances = [loss * factor for loss, factor in zip(sums_of_l2_distances, multipliers)]
        l2_loss = torch.tensor(sum(sums_of_l2_distances)).to(device)
        brain_loss = brain_loss / (brain_loss.detach() + 1e-8)
        l2_loss = l2_loss / (l2_loss.detach() + 1e-8)
        total_loss = brain_loss + l2_loss

        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        del feature_difference, l2_distances, bhg_feature, multi_brain_feature
        torch.cuda.empty_cache()
        optimizer.step()
        running_loss += total_loss.item()
        loss_epoch.append(total_loss.item())
        # Evaluate AUC and other metrics
        label_real += [i for i in decollate_batch(labels)]
        label_pred += [post_pred(i).detach().cpu().numpy().argmax() for i in decollate_batch(unibrain_out)]
        AUC(y_pred=[post_pred(i) for i in decollate_batch(unibrain_out)],
            y=[post_label(i) for i in decollate_batch(labels, detach=False)])

    loss_results = loss_epoch.aggregate()
    cm_train = ConfusionMatrix(actual_vector=label_real, predict_vector=label_pred)
    logger.print_message(f"Epoch {epoch}")
    logger.print_message(
        f"Training    - Loss:{float(loss_results):.4f} "
        f"ACC:{float(cm_train.Overall_ACC):.4f} "
        f"SEN:{float(list(cm_train.TPR.values())[1]):.4f} "
        f"SPE:{float(list(cm_train.TNR.values())[1]):.4f} "
        f"F1:{float(list(cm_train.F1.values())[1]):.4f} "
        f"AUC:{AUC.aggregate():.4f}")

    return float(AUC.aggregate()), loss_results, cm_train


def validate_epoch(brain_model, brain_classifier_head, dataloader, device, logger):
    brain_model.eval()
    brain_classifier_head.eval()
    running_loss = 0.0
    label_pred = []
    label_real = []
    loss_epoch = CumulativeAverage()
    AUC = ROCAUCMetric(average='macro')
    post_pred = Compose([Activations(softmax=True)])
    post_label = Compose([AsDiscrete(to_onehot=args.num_classes)])

    with torch.no_grad():
        for i, (feats_brain, brain_labels) in enumerate(dataloader):
            feats_brain, brain_labels = feats_brain.to(device), brain_labels.to(device)
            mciCount = torch.sum(brain_labels == 1).item()
            ncCount = torch.sum(brain_labels == 0).item()
            mciweight = ncCount / (mciCount + ncCount)
            ncweight = mciCount / (mciCount + ncCount)
            weights = [mciweight, ncweight]
            weights = [w if w != 0.0 else 1e-4 for w in weights]
            class_weights = torch.FloatTensor(weights).to(device)
            unibrain_out = brain_model(feats_brain)
            unibrain_out = unibrain_out[-1]
            unibrain_out = unibrain_out.permute(0, 2, 1)
            adaptive_pool = nn.AdaptiveAvgPool1d(1)
            unibrain_out = adaptive_pool(unibrain_out)
            unibrain_out = torch.flatten(unibrain_out, 1)
            unibrain_out = brain_classifier_head(unibrain_out)
            criterion = nn.CrossEntropyLoss(weight=class_weights)
            brain_loss = criterion(unibrain_out, brain_labels)

            running_loss += brain_loss.item()
            loss_epoch.append(brain_loss.item())
            # Evaluate AUC and other metrics
            label_real += [i for i in decollate_batch(brain_labels)]
            label_pred += [post_pred(i).detach().cpu().numpy().argmax() for i in decollate_batch(unibrain_out)]
            AUC(y_pred=[post_pred(i) for i in decollate_batch(unibrain_out)],
                y=[post_label(i) for i in decollate_batch(brain_labels, detach=False)])

    loss_results = loss_epoch.aggregate()
    cm_val = ConfusionMatrix(actual_vector=label_real, predict_vector=label_pred)
    logger.print_message(
        f"Validating    - Loss:{float(loss_results):.4f} "
        f"ACC:{float(cm_val.Overall_ACC):.4f} "
        f"SEN:{float(list(cm_val.TPR.values())[1]):.4f} "
        f"SPE:{float(list(cm_val.TNR.values())[1]):.4f} "
        f"F1:{float(list(cm_val.F1.values())[1]):.4f} "
        f"AUC:{AUC.aggregate():.4f}")
    return float(AUC.aggregate()), loss_results, cm_val


def main():
    set_seed(42)
    best_test_auc = 0.75

    save_dir = os.path.join(args.output, args.save_file_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    train_t_dir = os.path.join(save_dir, 'train')
    val_t_dir = os.path.join(save_dir, 'val')

    if not os.path.exists(train_t_dir):
        os.makedirs(train_t_dir)
    if not os.path.exists(val_t_dir):
        os.makedirs(val_t_dir)
    if not os.path.exists(args.output):
        os.makedirs(args.output)

    training_acc, training_loss, training_sen, training_spe, training_f1, training_auc = [], [], [], [], [], []
    validation_acc, validation_loss, validation_sen, validation_spe, validation_f1, validation_auc = [], [], [], [], [], []
    best_brain_pet_hierarchical_model_path = os.path.join(save_dir, f"best_model_{args.save_file_name}.ckpt")
    folder_name = ("fold " + str(args.fold_index) + ",embed_dim " + str(args.embed_dim) +
                   ",depth " + str(args.depths) + ",num_heads " + str(args.num_heads) +
                   ",window_size " + str(args.window_size) + ",patch_size " + str(args.patch_size) +
                   ",batch_size " + str(args.batch_size) + ",mlp_ratio " + str(args.mlp_ratio) +
                   ",drop_rate " + str(args.drop_rate) + ",lr " + str(args.lr) +
                   ",weight_decay " + str(args.wd) + ",enhance_p " + str(args.enhance_ps))

    transform = transforms.Compose([transforms.RandomApply([transforms.RandomAffine(degrees=20, translate=(0.1, 0.1), scale=(0.95, 1.05))], p=args.enhance_ps)])
    train_dataset = GeneralDataset(args.feats_path, args.fold_path_brain, args.fold_path_bhg, args.fold_index, "train", transform=transform)
    dataloader_train = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, collate_fn=custom_collate_fn)
    val_dataset = GeneralDataset_brain(args.feats_path, args.fold_path_brain, args.fold_index, "val", transform=None)
    dataloader_val = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    del train_dataset, val_dataset
    gc.collect()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    bhg_hierarchical_model = Multi_organ_PanSwin(
        in_chans=args.in_chans,
        embed_dim=args.embed_dim,
        window_size=args.window_size,
        patch_size=args.patch_size,
        depths=args.depths,
        num_heads=args.num_heads,
        mlp_ratio=args.mlp_ratio,
        qkv_bias=args.qkv_bias,
        drop_rate=args.drop_rate,
        attn_drop_rate=args.attn_drop_rate,
        drop_path_rate=args.drop_path_rate,
        patch_norm=args.patch_norm,
        use_checkpoint=args.use_checkpoint,
        spatial_dims=args.spatial_dims,
        brain_size=args.brain_size,
        heart_size=args.heart_size,
        gut_size=args.gut_size).to(device)

    multi_classifier_head = TransformerClassifier(
        brain_num_features=768 * 3,
        transformer_encoder_head=8,
        transformer_encoder_dim_feedforward=128,
        transformer_encoder_dropout=0.1,
        transformer_encoder_activation='relu',
        transformer_encoder_num_layers=1,
        num_classes=2
    ).to(device)

    brain_model = PanSwin(
        in_chans=args.in_chans,
        embed_dim=args.embed_dim,
        window_size=args.window_size,
        patch_size=args.patch_size,
        depths=args.depths,
        num_heads=args.num_heads,
        mlp_ratio=args.mlp_ratio,
        qkv_bias=args.qkv_bias,
        drop_rate=args.drop_rate,
        attn_drop_rate=args.attn_drop_rate,
        drop_path_rate=args.drop_path_rate,
        patch_norm=args.patch_norm,
        use_checkpoint=args.use_checkpoint,
        spatial_dims=args.spatial_dims,
    ).to(device)
    brain_classifier_head = MLP(768, 128, 2).to(device)

    logger = Logger(save_dir, 'log.txt')

    checkpoint = torch.load(args.bhg_checkpoint_path, map_location=device)
    bhg_hierarchical_model.load_state_dict(checkpoint['bhg_hierarchical_model'])
    multi_classifier_head.load_state_dict(checkpoint['multi_classifier_head'])
    for param in bhg_hierarchical_model.parameters():
        param.requires_grad = False
    for param in multi_classifier_head.parameters():
        param.requires_grad = False
    checkpoint = torch.load(args.brain_checkpoint_path, map_location=device)
    brain_model.load_state_dict(checkpoint['brain_swin_transformer'])
    brain_classifier_head.load_state_dict(checkpoint['brain_classifier_head'])

    params = list(brain_model.parameters()) + list(brain_classifier_head.parameters())

    brain_size = [dim // 4 for dim in args.brain_size]
    heart_size = [dim // 4 for dim in args.heart_size]
    gut_size = [dim // 4 for dim in args.gut_size]
    raw_brain_shape = [torch.empty((args.batch_size, *([size // (2 ** i) for size in brain_size]), args.embed_dim * (2 ** i))) for i in range(len(args.depths))]
    raw_heart_shape = [torch.empty((args.batch_size, *([size // (2 ** i) for size in heart_size]), args.embed_dim * (2 ** i))) for i in range(len(args.depths))]
    raw_gut_shape = [torch.empty((args.batch_size, *([size // (2 ** i) for size in gut_size]), args.embed_dim * (2 ** i))) for i in range(len(args.depths))]
    brain_packer = TGICBranches(raw_brain_shape, num_heads=8, scale_factor=2).to(device)
    heart_packer = TGICBranches(raw_heart_shape, num_heads=8, scale_factor=2).to(device)
    gut_packer = TGICBranches(raw_gut_shape, num_heads=8, scale_factor=2).to(device)
    for token_packer in brain_packer.token_packers:
        params += list(token_packer.parameters())
    for token_packer in heart_packer.token_packers:
        params += list(token_packer.parameters())
    for token_packer in gut_packer.token_packers:
        params += list(token_packer.parameters())
    qkv = ComputeQKV(raw_brain_shape).to(device)
    params += list(qkv.q_linear_list.parameters())
    params += list(qkv.k_linear_list.parameters())
    params += list(qkv.v_linear_list.parameters())
    multihead_atten = [nn.MultiheadAttention(embed_dim=args.embed_dim * (2 ** i), num_heads=1).to(device) for i in range(len(args.depths))]
    for multihead in multihead_atten:
        params += list(multihead.parameters())
    head_mlp = [MLP(args.embed_dim * (2 ** i), args.embed_dim * (2 ** i), args.embed_dim * (2 ** i)).to(device) for i in range(len(args.depths))]
    for head in head_mlp:
        params += list(head.parameters())
    optimizer = optim.AdamW(params=params, lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=args.wd)
    del raw_brain_shape, raw_heart_shape, raw_gut_shape
    gc.collect()

    for epoch in range(100):
        train_auc, train_loss_results, cm_train = train_epoch(epoch, bhg_hierarchical_model, brain_model, brain_classifier_head, dataloader_train, optimizer, brain_packer, heart_packer, gut_packer, qkv, multihead_atten, head_mlp, device, logger, args.patch_size, args.embed_dim, args.depths)
        val_auc, val_loss_results, cm_val = validate_epoch(brain_model, brain_classifier_head, dataloader_val, device, logger)
        training_loss.append(float(train_loss_results))
        training_acc.append(float(cm_train.Overall_ACC))
        training_sen.append(float(list(cm_train.TPR.values())[1]))
        training_spe.append(float(list(cm_train.TNR.values())[1]))
        training_f1.append(float(list(cm_train.F1.values())[1]))
        training_auc.append(train_auc)
        validation_loss.append(float(val_loss_results))
        validation_acc.append(float(cm_val.Overall_ACC))
        validation_sen.append(float(list(cm_val.TPR.values())[1]))
        validation_spe.append(float(list(cm_val.TNR.values())[1]))
        validation_f1.append(float(list(cm_val.F1.values())[1]))
        validation_auc.append(val_auc)
        if train_auc > 0.75 and val_auc > 0.75 and training_f1[-1] > 0.2 and validation_f1[-1] > 0.2:
            save_checkpoint({
                'brain_model': brain_model.state_dict(),
                'brain_classifier_head': brain_classifier_head.state_dict(),
                'brain_packer': brain_packer.state_dict(),
                'heart_packer': heart_packer.state_dict(),
                'gut_packer': gut_packer.state_dict(),
                'qkv': qkv.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_test_auc': best_test_auc
            }, best_brain_pet_hierarchical_model_path)
            logger.print_message(f"Saved best model with validation AUC: {val_auc:.4f} at epoch {epoch}")
        if (epoch + 1) % 2 == 0:
            plot(training_acc, train_t_dir, folder_name, "training_acc")
            plot(training_loss, train_t_dir, folder_name, "training_loss")
            plot(training_sen, train_t_dir, folder_name, "training_sen")
            plot(training_spe, train_t_dir, folder_name, "training_spe")
            plot(training_f1, train_t_dir, folder_name, "training_f1")
            plot(training_auc, train_t_dir, folder_name, "training_auc")
            plot(validation_acc, val_t_dir, folder_name, "val_acc")
            plot(validation_loss, val_t_dir, folder_name, "val_loss")
            plot(validation_sen, val_t_dir, folder_name, "val_sen")
            plot(validation_spe, val_t_dir, folder_name, "val_spe")
            plot(validation_f1, val_t_dir, folder_name, "val_f1")
            plot(validation_auc, val_t_dir, folder_name, "val_auc")

if __name__ == '__main__':
    main()