import argparse
import datetime
import random
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import matplotlib

matplotlib.use('Agg')
from matplotlib import pyplot as plt
from model.PanSwin import PanSwin
from dataloader.GeneralDataset_whole_body import GeneralDataset_whole_body
from dataloader.GeneralDataset_whole_body_and_brain import GeneralDatasetBHG, custom_collate_fn_bhg
from pycm import ConfusionMatrix
from monai.data import decollate_batch
from monai.metrics import CumulativeAverage, ROCAUCMetric
from monai.transforms import Compose, Activations, AsDiscrete
from model.module import MLP, ClipLoss, ClipLossLabel, ClipLossLabelSingal, Multi_organ_PanSwin, TransformerClassifier, \
    PatchFlatten, TokenFlatten, process_patches, feature_transformation, find_minimal_bounding_cube, Logger
import textwrap

# Argument parser
parser = argparse.ArgumentParser(description='Semi-supervised multi-organ collaboration')
parser.add_argument("--labeled_feats_path", default=r"D:\DATA\MNI\FDG_crop1.5\crop_pth", type=str, help="dataset directory")
parser.add_argument("--unlabeled_feats_path", default=r"D:\DATA\MNI\FDG_crop1.5\crop_pth\Hangzhou", type=str, help="dataset directory")
parser.add_argument("--labeled_fold_path_bhg", default=r"D:\DATA\MNI\FDG_fold_partition\Whole_Body_Labeled\5fold\42", type=str, help="dataset pdf directory")
parser.add_argument("--unlabeled_fold_path_bhg", default=r"D:\DATA\MNI\FDG_fold_partition\Hangzhou_Unlabeled\5fold\42", type=str, help="dataset pdf directory")
parser.add_argument("--brain_fold_path", default=r"D:\DATA\MNI\MRI_fold_partition\5fold\42", type=str, help="dataset pdf directory")
parser.add_argument("--output", type=str, default=r'D:\DATA\Output\Alignment_Classification_BHG', help="Use gradient checkpointing for reduced memory usage")
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
        best_val_auc = checkpoint['best_val_auc']
        return best_val_auc
    else:
        print(f"No checkpoint found at '{filepath}'")
        return None

def plot(data, dir, organ_name_type, image_name):
    try:
        x = list(range(len(data)))
        plt.plot(x, data, label=organ_name_type)
        plt.xlabel('epoch')
        title = organ_name_type
        plt.title(textwrap.fill(title, width=70))
        plt.xlim(0, len(data))
        plt.legend(loc='best')
        plt.savefig(os.path.join(dir, image_name + '.png'))
        plt.close()
    except Exception as e:
        print(e)

class EarlyStopping:
    def __init__(self, patience=10, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

# Training for an epoch
def train_epoch(epoch, model, multi_classifier_head, brain_classifier_head, heart_classifier_head, gut_classifier_head, num_pseudo, dataloader, optimizer, clip_criterion, labeled_clip_criterion, device, logger):
    model.train()
    running_loss = 0.0
    label_pred = []
    label_real = []
    loss_epoch = CumulativeAverage()
    AUC = ROCAUCMetric(average='macro')
    post_pred = Compose([Activations(softmax=True)])
    post_label = Compose([AsDiscrete(to_onehot=2)])

    for batch_data in dataloader:
        unlabeled_multi_feats_brain = [data['brain'] for data in batch_data if not data['is_labeled']]
        unlabeled_multi_feats_heart = [data['heart'] for data in batch_data if not data['is_labeled']]
        unlabeled_multi_feats_gut = [data['gut'] for data in batch_data if not data['is_labeled']]
        labeled_multi_feats_brain = [data['brain'] for data in batch_data if data['is_labeled']]
        labeled_multi_feats_heart = [data['heart'] for data in batch_data if data['is_labeled']]
        labeled_multi_feats_gut = [data['gut'] for data in batch_data if data['is_labeled']]
        labeled_multi_labels = [data['label'] for data in batch_data if data['is_labeled']]
        unlabeled_num = len(unlabeled_multi_feats_brain)
        if unlabeled_num < num_pseudo:
            num_pseudo = unlabeled_num
        del batch_data
        torch.cuda.empty_cache()
        unlabeled_multi_feats_brain = torch.stack(unlabeled_multi_feats_brain).to(device)
        unlabeled_multi_feats_heart = torch.stack(unlabeled_multi_feats_heart).to(device)
        unlabeled_multi_feats_gut = torch.stack(unlabeled_multi_feats_gut).to(device)
        labeled_multi_feats_brain = torch.stack(labeled_multi_feats_brain).to(device)
        labeled_multi_feats_heart = torch.stack(labeled_multi_feats_heart).to(device)
        labeled_multi_feats_gut = torch.stack(labeled_multi_feats_gut).to(device)
        labeled_multi_labels = torch.stack(labeled_multi_labels).to(device)

        # Brain-Heart-Gut
        labeled_brain_out, labeled_heart_out, labeled_gut_out = model(labeled_multi_feats_brain, labeled_multi_feats_heart, labeled_multi_feats_gut)
        avgpool = nn.AdaptiveAvgPool1d(1)
        labeled_brain_out = [feature_transformation(b, avgpool) for b in labeled_brain_out]
        labeled_heart_out = [feature_transformation(b, avgpool) for b in labeled_heart_out]
        labeled_gut_out = [feature_transformation(b, avgpool) for b in labeled_gut_out]
        labeled_outputs = multi_classifier_head(labeled_brain_out[-1], labeled_heart_out[-1], labeled_gut_out[-1])
        labeled_clip_loss = [labeled_clip_criterion[i](b_out, h_out, g_out, labeled_multi_labels, device) for i, (b_out, h_out, g_out) in enumerate(zip(labeled_brain_out, labeled_heart_out, labeled_gut_out))]
        multipliers = [0.1, 0.2, 0.3, 0.4]
        labeled_clip_loss = [loss * factor for loss, factor in zip(labeled_clip_loss, multipliers)]
        labeled_clip_loss = sum(labeled_clip_loss)
        torch.cuda.empty_cache()

        ncCount = torch.sum(labeled_multi_labels == 0).item()
        mciCount = torch.sum(labeled_multi_labels == 1).item()
        weights = [mciCount / (mciCount + ncCount), ncCount / (mciCount + ncCount)]
        weights = [w if w != 0.0 else 1e-4 for w in weights]
        class_weights = torch.FloatTensor(weights).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        labeled_clf_loss = criterion(labeled_outputs, labeled_multi_labels)

        unlabeled_brain_out, unlabeled_heart_out, unlabeled_gut_out = model(unlabeled_multi_feats_brain, unlabeled_multi_feats_heart, unlabeled_multi_feats_gut)
        unlabeled_brain_out = [feature_transformation(b, avgpool) for b in unlabeled_brain_out]
        unlabeled_heart_out = [feature_transformation(b, avgpool) for b in unlabeled_heart_out]
        unlabeled_gut_out = [feature_transformation(b, avgpool) for b in unlabeled_gut_out]
        pseudo_brain_out = brain_classifier_head(unlabeled_brain_out[-1][-unlabeled_num:, :])
        pseudo_heart_out = heart_classifier_head(unlabeled_heart_out[-1][-unlabeled_num:, :])
        pseudo_gut_out = gut_classifier_head(unlabeled_gut_out[-1][-unlabeled_num:, :])
        pseudo_brain_softmax = F.softmax(pseudo_brain_out, dim=1)
        _, top_indices = torch.topk(pseudo_brain_softmax[:, 1], k=num_pseudo, largest=True)
        pseudo_heart_out = pseudo_heart_out[top_indices]
        pseudo_gut_out = pseudo_gut_out[top_indices]
        pseudo_labels = torch.argmax(pseudo_brain_softmax[top_indices], dim=1)
        unlabeled_criterion = nn.CrossEntropyLoss()
        pseudo_loss = 0.5 * unlabeled_criterion(pseudo_heart_out, pseudo_labels) + 0.5 * unlabeled_criterion(pseudo_gut_out, pseudo_labels)
        unlabeled_brain_out = [torch.cat((labeled, unlabeled), dim=0) for labeled, unlabeled in zip(labeled_brain_out, unlabeled_brain_out)]
        unlabeled_heart_out = [torch.cat((labeled, unlabeled), dim=0) for labeled, unlabeled in zip(labeled_heart_out, unlabeled_heart_out)]
        unlabeled_gut_out = [torch.cat((labeled, unlabeled), dim=0) for labeled, unlabeled in zip(labeled_gut_out, unlabeled_gut_out)]
        unlabeled_align_loss = [clip_criterion[i](b_out, h_out, g_out, device) for i, (b_out, h_out, g_out) in enumerate(zip(unlabeled_brain_out, unlabeled_heart_out, unlabeled_gut_out))]
        multipliers = [0.1, 0.2, 0.3, 0.4]
        unlabeled_align_loss = [loss * factor for loss, factor in zip(unlabeled_align_loss, multipliers)]
        unlabeled_align_loss = sum(unlabeled_align_loss)
        del unlabeled_multi_feats_brain, unlabeled_multi_feats_heart, unlabeled_multi_feats_gut
        torch.cuda.empty_cache()

        labeled_clf_loss = labeled_clf_loss / (labeled_clf_loss.detach() + 1e-8)
        pseudo_loss = pseudo_loss / (pseudo_loss.detach() + 1e-8)
        unlabeled_align_loss = unlabeled_align_loss / (unlabeled_align_loss.detach() + 1e-8)
        labeled_clip_loss = labeled_clip_loss / (labeled_clip_loss.detach() + 1e-8)

        multi_loss = labeled_clf_loss + pseudo_loss + unlabeled_align_loss + labeled_clip_loss

        optimizer.zero_grad()
        multi_loss.backward()
        optimizer.step()
        running_loss += multi_loss.item()
        loss_epoch.append(multi_loss.item())

        # Evaluate AUC and other metrics
        label_real += [i for i in decollate_batch(labeled_multi_labels)]
        label_pred += [post_pred(i).detach().cpu().numpy().argmax() for i in decollate_batch(labeled_outputs)]
        AUC(y_pred=[post_pred(i) for i in decollate_batch(labeled_outputs)],
            y=[post_label(i) for i in decollate_batch(labeled_multi_labels, detach=False)])
        del labeled_multi_labels
        torch.cuda.empty_cache()

    loss_results = loss_epoch.aggregate()
    cm_train = ConfusionMatrix(actual_vector=label_real, predict_vector=label_pred)
    logger.print_message(f"Epoch {epoch}")
    logger.print_message(
        f"Training    - labeled_clf_loss:{labeled_clf_loss:.4f} "
        f"pseudo_loss:{pseudo_loss:.4f} "
        f"unlabeled_align_loss:{unlabeled_align_loss:.4f} "
        f"labeled_clip_loss:{labeled_clip_loss:.4f}")
    logger.print_message(
        f"Training    - Loss:{float(loss_results):.4f} "
        f"ACC:{float(cm_train.Overall_ACC):.4f} "
        f"SEN:{float(list(cm_train.TPR.values())[1]):.4f} "
        f"SPE:{float(list(cm_train.TNR.values())[1]):.4f} "
        f"F1:{float(list(cm_train.F1.values())[1]):.4f} "
        f"AUC:{AUC.aggregate():.4f}")

    return loss_results, float(AUC.aggregate()), cm_train


def validate_epoch(model, multi_classifier_head, dataloader, device, logger):
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

            # Evaluate AUC and other metrics for PET
            label_real += [i for i in decollate_batch(labels)]
            label_pet_pred += [pet_post_pred(i).detach().cpu().numpy().argmax() for i in decollate_batch(labeled_outputs)]
            AUC(y_pred=[pet_post_pred(i) for i in decollate_batch(labeled_outputs)], y=[pet_post_label(i) for i in decollate_batch(labels, detach=False)])

    brain_loss = loss_epoch.aggregate()
    cm_val_pet = ConfusionMatrix(actual_vector=label_real, predict_vector=label_pet_pred)
    logger.print_message(
        f"Validation  - Loss:{float(brain_loss):.4f} "
        f"ACC:{float(cm_val_pet.Overall_ACC):.4f} "
        f"SEN:{float(list(cm_val_pet.TPR.values())[1]):.4f} "
        f"SPE:{float(list(cm_val_pet.TNR.values())[1]):.4f} "
        f"F1:{float(list(cm_val_pet.F1.values())[1]):.4f} "
        f"AUC:{AUC.aggregate():.4f}")

    return brain_loss, float(AUC.aggregate()), cm_val_pet


def main():
    set_seed(42)
    hid_dim = 128
    best_val_auc = 0.0
    num_pseudo = 4
    best_val_loss = float('inf')
    best_epoch = -1

    # Get parameters from args
    enhance_p = args.enhance_p
    fold_idx = args.fold_index

    save_dir = os.path.join(args.output, args.save_file_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    train_t_dir = os.path.join(save_dir, 'train')
    val_t_dir = os.path.join(save_dir, 'val')
    if not os.path.exists(train_t_dir):
        os.makedirs(train_t_dir)
    if not os.path.exists(val_t_dir):
        os.makedirs(val_t_dir)

    training_acc, training_loss, training_sen, training_spe, training_f1, training_auc = [], [], [], [], [], []
    validation_acc, validation_loss, validation_sen, validation_spe, validation_f1, validation_auc = [], [], [], [], [], []
    best_bhg_hierarchical_model_path = os.path.join(save_dir, f"best_model_{args.save_file_name}.ckpt")
    folder_name = ("fold " + str(fold_idx) + ",embed_dim " + str(args.embed_dim) +
                   ",depth " + str(args.depths) + ",num_heads " + str(args.num_heads) +
                   ",window_size " + str(args.window_size) + ",patch_size " + str(args.patch_size) +
                   ",batch_size " + str(args.batch_size) + ",mlp_ratio " + str(args.mlp_ratio) +
                   ",drop_rate " + str(args.drop_rate) + ",lr " + str(args.lr) +
                   ",weight_decay " + str(args.wd) + ",enhance_p " + str(args.enhance_p))
    if not os.path.exists(args.output):
        os.makedirs(args.output)

    transform = transforms.Compose([transforms.RandomApply(
        [transforms.RandomAffine(degrees=20, translate=(0.1, 0.1), scale=(0.95, 1.05))], p=enhance_p)])

    train_dataset = GeneralDatasetBHG(args.unlabeled_feats_path, args.labeled_feats_path, args.unlabeled_fold_path_bhg, args.labeled_fold_path_bhg, fold_idx, "train", transform=transform)
    dataloader_train = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=custom_collate_fn_bhg, num_workers=args.workers, pin_memory=True)

    val_dataset = GeneralDataset_whole_body(args.labeled_feats_path, args.labeled_fold_path_bhg, fold_idx, "val", transform=None)
    dataloader_val = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=None, num_workers=args.workers, pin_memory=True)

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
    brain_classifier_head = MLP(768, 128, 2).to(device)
    heart_classifier_head = MLP(768, 128, 2).to(device)
    gut_classifier_head = MLP(768, 128, 2).to(device)

    checkpoint_path = r'D:\DATA\Output\Alignment_Classification_BHG\best_brain_classifer_pretrain_model_fold1_2025-0309-1625.ckpt'

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path)
    try:
        bhg_hierarchical_model.brain_swin_transformer.load_state_dict(checkpoint['brain_swin_transformer'])
        brain_classifier_head.load_state_dict(checkpoint['brain_classifier_head'])
        print("Parameters loaded successfully：brain_swin_transformer and brain_classifier_head")
    except KeyError:
        print("Parameter loading failed：brain_swin_transformer or brain_classifier_head")

    for param in bhg_hierarchical_model.brain_swin_transformer.parameters():
        param.requires_grad = False
    for param in brain_classifier_head.parameters():
        param.requires_grad = False

    clip_criterion = [ClipLoss(args.embed_dim * (2 ** i), hid_dim=hid_dim, temperature=0.1).to(device) for i in range(len(args.depths))]
    labeled_clip_criterion = [ClipLossLabel(args.embed_dim * (2 ** i), hid_dim=hid_dim, temperature=0.1).to(device) for i in range(len(args.depths))]
    params = list(bhg_hierarchical_model.parameters()) + list(multi_classifier_head.parameters()) + list(heart_classifier_head.parameters()) + list(gut_classifier_head.parameters())
    for criterion_sub in clip_criterion:
        params += list(criterion_sub.parameters())
    for criterion_sub in labeled_clip_criterion:
        params += list(criterion_sub.parameters())
    optimizer = optim.AdamW(params=params, lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=args.wd)
    logger = Logger(save_dir, 'log.txt')

    for epoch in range(50):
        train_loss_results, train_auc, cm_train = train_epoch(epoch, bhg_hierarchical_model, multi_classifier_head,
                                                              brain_classifier_head, heart_classifier_head,
                                                              gut_classifier_head, num_pseudo, dataloader_train,
                                                              optimizer, clip_criterion, labeled_clip_criterion,
                                                              device, logger)
        val_loss_results, val_auc, cm_val = validate_epoch(bhg_hierarchical_model, multi_classifier_head,
                                                           dataloader_val, device, logger)

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

        # Save best model based on validation AUC
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_epoch = epoch
            save_checkpoint({
                'bhg_hierarchical_model': bhg_hierarchical_model.state_dict(),
                'multi_classifier_head': multi_classifier_head.state_dict(),
                'brain_classifier_head': brain_classifier_head.state_dict(),
                'heart_classifier_head': heart_classifier_head.state_dict(),
                'gut_classifier_head': gut_classifier_head.state_dict(),
                'optimizer': optimizer.state_dict(),
                'clip_criterion': [criterion.state_dict() for criterion in clip_criterion],
                'best_val_auc': best_val_auc,
                'epoch': epoch
            }, best_bhg_hierarchical_model_path)
            logger.print_message(f"Saved best model with validation AUC: {best_val_auc:.4f} at epoch {epoch}")

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

    logger.print_message(f"Best validation AUC: {float(best_val_auc):.4f} at epoch {best_epoch}")
    return best_val_auc


if __name__ == '__main__':
    args = parser.parse_args()
    main()
