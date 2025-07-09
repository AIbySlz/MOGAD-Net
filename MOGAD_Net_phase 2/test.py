import argparse
import datetime
import random
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib

matplotlib.use('Agg')
from model.PanSwin import PanSwin
from dataloader.GeneralDataset_brain_data import GeneralDataset_brain
from pycm import ConfusionMatrix
from monai.data import decollate_batch
from monai.metrics import CumulativeAverage, ROCAUCMetric
from monai.transforms import Compose, Activations, AsDiscrete
from torch.cuda.amp import GradScaler, autocast
from model.TGIC import TGICBranches, ComputeQKV
from model.module import MLP, Logger

# Argument parser
parser = argparse.ArgumentParser(description='Validation Only')
parser.add_argument("--feats_path", default=r"D:\DATA\MNI\FDG_crop1.5\crop_pth", type=str, help="dataset directory")
parser.add_argument("--fold_path_brain", default=r"D:\DATA\MNI\FDG_fold_partition\Brain_Labeled\5fold\42", type=str, help="dataset pdf directory")
parser.add_argument("--fold_path_bhg", default=r"D:\DATA\MNI\FDG_fold_partition\Whole_Body_Labeled\5fold\42", type=str, help="dataset pdf directory")
parser.add_argument("--output", type=str, default=r'D:\DATA\Output\Alignment_Classification_BHG', help="Output directory")
parser.add_argument("--save_file_name", default=datetime.datetime.now().strftime("%Y-%m%d-%H%M"), type=str, help="folder name to save subject")
parser.add_argument("--brain_checkpoint_path", type=str, default=r'D:\DATA\Output\Brain_Pretrain\FDG_SwinToken_2025-0309-1625\fold_1_lr_4.9999999999999996e-06_wd_0.001_eh_0.9\best_brain_classifer_pretrain_model_fold1_2025-0309-1625.ckpt', help="Path to brain model checkpoint")
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
        f"Validation - Loss:{float(loss_results):.4f} "
        f"ACC:{float(cm_val.Overall_ACC):.4f} "
        f"SEN:{float(list(cm_val.TPR.values())[1]):.4f} "
        f"SPE:{float(list(cm_val.TNR.values())[1]):.4f} "
        f"F1:{float(list(cm_val.F1.values())[1]):.4f} "
        f"AUC:{AUC.aggregate():.4f}")
    return float(AUC.aggregate()), loss_results, cm_val


def main():
    set_seed(42)

    save_dir = os.path.join(args.output, args.save_file_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    val_t_dir = os.path.join(save_dir, 'val')
    if not os.path.exists(val_t_dir):
        os.makedirs(val_t_dir)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger = Logger(save_dir, 'log.txt')

    # Load validation dataset
    val_dataset = GeneralDataset_brain(args.feats_path, args.fold_path_brain, args.fold_index, "val", transform=None)
    dataloader_val = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    # Initialize models
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

    # Load checkpoints
    brain_checkpoint = torch.load(args.brain_checkpoint_path, map_location=device)
    brain_model.load_state_dict(brain_checkpoint['brain_swin_transformer'])
    brain_classifier_head.load_state_dict(brain_checkpoint['brain_classifier_head'])
    logger.print_message(f"Successfully loaded checkpoints from:")
    logger.print_message(f"Brain model: {args.brain_checkpoint_path}")

    # Run validation
    val_auc, val_loss_results, cm_val = validate_epoch(brain_model, brain_classifier_head, dataloader_val, device, logger)
    logger.print_message(f"Final Validation Results - AUC: {val_auc:.4f}, Loss: {val_loss_results:.4f}")

if __name__ == '__main__':
    main()