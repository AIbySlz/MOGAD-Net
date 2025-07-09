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
from dataloader.GeneralDataset_whole_body import GeneralDataset_whole_body
from pycm import ConfusionMatrix
from monai.data import decollate_batch
from monai.metrics import CumulativeAverage, ROCAUCMetric
from monai.transforms import Compose, Activations, AsDiscrete
from model.module import Multi_organ_PanSwin, TransformerClassifier, feature_transformation, Logger
import textwrap

# Argument parser
parser = argparse.ArgumentParser(description='Semi-supervised multi-organ collaboration')
parser.add_argument("--labeled_feats_path", default=r"D:\DATA\MNI\FDG_crop1.5\crop_pth", type=str, help="dataset directory")
parser.add_argument("--unlabeled_feats_path", default=r"D:\DATA\MNI\FDG_crop1.5\crop_pth\Hangzhou", type=str, help="dataset directory")
parser.add_argument("--labeled_fold_path_bhg", default=r"D:\DATA\MNI\FDG_fold_partition\Whole_Body_Labeled\5fold\42", type=str, help="dataset pdf directory")
parser.add_argument("--unlabeled_fold_path_bhg", default=r"D:\DATA\MNI\FDG_fold_partition\Hangzhou_Unlabeled\5fold\42", type=str, help="dataset pdf directory")
parser.add_argument("--brain_fold_path", default=r"D:\DATA\MNI\MRI_fold_partition\5fold\42", type=str, help="dataset pdf directory")
parser.add_argument("--output", type=str, default=r'D:\DATA\Output\Alignment_Classification_BHG', help="Use gradient checkpointing for reduced memory usage")
parser.add_argument("--checkpoint_path", type=str, default=r'D:\DATA\Output\Alignment_Classification_BHG\best_model_3028762.ckpt', help="Path to model checkpoint")
parser.add_argument("--lr", default=5e-6, type=float, help="learning rate")
parser.add_argument("--wd", default=1e-3, type=float, help="weight decay")
parser.add_argument("--enhance_p", default=0.7, type=float, help="enhance probability")
parser.add_argument("--save_file_name", default=datetime.datetime.now().strftime("%Y-%m%d-%H%M"), type=str, help="folder name to save subject")
parser.add_argument("--fold_index", default=1, type=int, help="current fold_index")
parser.add_argument("--batch_size", default=8, type=int, help="number of batch size")
parser.add_argument("--brain_size", default=[128, 128, 128], type=int, nargs='+', help="window_size")
parser.add_argument("--heart_size", default=[128, 96, 96], type=int, nargs='+', help="window_size")
parser.add_argument("--gut_size", default=[224, 160, 352], type=int, nargs='+', help="window_size")
parser.add_argument("--workers", default=2, type=int, help="number of workers")
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

def test_epoch(model, multi_classifier_head, dataloader, device, logger):
    model.eval()
    running_loss = 0.0
    label_real = []
    label_pet_pred = []
    loss_epoch = CumulativeAverage()
    AUC = ROCAUCMetric(average='macro')
    pet_post_pred = Compose([Activations(softmax=True)])
    pet_post_label = Compose([AsDiscrete(to_onehot=2)])

    with torch.no_grad():
        for i, (pet_brain, pet_heart, pet_gut, labels) in enumerate(dataloader):
            pet_brain, pet_heart, pet_gut, labels = pet_brain.to(device), pet_heart.to(device), pet_gut.to(
                device), labels.to(device)

            brain_pet_out, heart_pet_out, gut_pet_out = model(pet_brain, pet_heart, pet_gut)
            avgpool = nn.AdaptiveAvgPool1d(1)
            brain_pet_out = [feature_transformation(b, avgpool) for b in brain_pet_out]
            heart_pet_out = [feature_transformation(b, avgpool) for b in heart_pet_out]
            gut_pet_out = [feature_transformation(b, avgpool) for b in gut_pet_out]
            labeled_outputs = multi_classifier_head(brain_pet_out[-1], heart_pet_out[-1], gut_pet_out[-1])

            ncCount = torch.sum(labels == 0).item()
            mciCount = torch.sum(labels == 1).item()
            weights = [mciCount / (mciCount + ncCount), ncCount / (mciCount + ncCount)]
            weights = [w if w != 0.0 else 1e-4 for w in weights]
            class_weights = torch.FloatTensor(weights).to(device)
            criterion = nn.CrossEntropyLoss(weight=class_weights)
            clas_loss = criterion(labeled_outputs, labels)

            running_loss += clas_loss.item()
            loss_epoch.append(clas_loss.item())

            label_real += [i for i in decollate_batch(labels)]
            label_pet_pred += [pet_post_pred(i).detach().cpu().numpy().argmax() for i in decollate_batch(labeled_outputs)]
            AUC(y_pred=[pet_post_pred(i) for i in decollate_batch(labeled_outputs)],
                y=[pet_post_label(i) for i in decollate_batch(labels, detach=False)])

    brain_loss = loss_epoch.aggregate()
    cm_test_pet = ConfusionMatrix(actual_vector=label_real, predict_vector=label_pet_pred)
    logger.print_message(
        f"Testing  - Loss:{float(brain_loss):.4f} "
        f"ACC:{float(cm_test_pet.Overall_ACC):.4f} "
        f"SEN:{float(list(cm_test_pet.TPR.values())[1]):.4f} "
        f"SPE:{float(list(cm_test_pet.TNR.values())[1]):.4f} "
        f"F1:{float(list(cm_test_pet.F1.values())[1]):.4f} "
        f"AUC:{AUC.aggregate():.4f}")
    return brain_loss, float(AUC.aggregate()), cm_test_pet


def main():
    set_seed(42)
    save_dir = os.path.join(args.output, args.save_file_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    test_t_dir = os.path.join(save_dir, 'test')
    if not os.path.exists(test_t_dir):
        os.makedirs(test_t_dir)

    test_dataset = GeneralDataset_whole_body(args.labeled_feats_path, args.labeled_fold_path_bhg, args.fold_index, "test", transform=None)
    dataloader_test = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=None, num_workers=args.workers, pin_memory=True)

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
        transformer_encoder_head=2,
        transformer_encoder_dim_feedforward=128,
        transformer_encoder_dropout=0.1,
        transformer_encoder_activation='relu',
        transformer_encoder_num_layers=1,
        num_classes=2
    ).to(device)

    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    bhg_hierarchical_model.load_state_dict(checkpoint['bhg_hierarchical_model'])
    multi_classifier_head.load_state_dict(checkpoint['multi_classifier_head'])

    logger = Logger(save_dir, 'log.txt')

    # Run testing
    test_loss_results, test_auc, cm_test = test_epoch(bhg_hierarchical_model, multi_classifier_head, dataloader_test, device, logger)


if __name__ == '__main__':
    args = parser.parse_args()
    main()
