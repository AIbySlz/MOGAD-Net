import argparse
import datetime
import random
import os
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
from dataloader.GeneralDataset_brain_all_data import GeneralDataset_brain
from pycm import ConfusionMatrix
from monai.data import decollate_batch
from monai.metrics import CumulativeAverage, ROCAUCMetric
from monai.transforms import Compose, Activations, AsDiscrete
from model.module import MLP, Logger
import textwrap

# Argument parser
parser = argparse.ArgumentParser(description='Pretraining brain PET-based model')
parser.add_argument("--brain_pth_path", default=r"D:\DATA\MNI\Har_FDG_crop1.5\crop_pth", type=str, help="dataset directory")
parser.add_argument("--brain_fold_path", default=r"D:\DATA\MNI\MRI_fold_partition\5fold\42", type=str, help="dataset pdf directory")
parser.add_argument("--rest_brain_fold_path", default=r"D:\DATA\MNI\FDG-MRI_brain_fold_partition\5fold\42", type=str, help="dataset pdf directory")
parser.add_argument("--output", type=str, default=r'D:\DATA\Output\Brain_Pretrain', help="Use gradient checkpointing for reduced memory usage")
parser.add_argument("--save_file_name", default=datetime.datetime.now().strftime("%Y-%m%d-%H%M"), type=str, help="folder name to save subject")
parser.add_argument("--lr", default=5e-6, type=float, help="learning rate")
parser.add_argument("--wd", default=1e-3, type=float, help="weight decay")
parser.add_argument("--enhance_p", default=0.7, type=float, help="enhance probability")
parser.add_argument("--brain_size", default=[128, 128, 128], type=int, nargs='+', help="window_size")
parser.add_argument("--fold_index", default=1, type=int, help="current fold_index")
parser.add_argument("--batch_size", default=16, type=int, help="number of batch size")
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

def train_epoch(epoch, brain_swin_transformer, brain_classifier_head, dataloader, optimizer, device, logger):
    brain_swin_transformer.train()
    running_loss = 0.0
    label_real = []
    label_pet_pred = []
    loss_epoch = CumulativeAverage()
    AUC = ROCAUCMetric(average='macro')
    pet_post_pred = Compose([Activations(softmax=True)])
    pet_post_label = Compose([AsDiscrete(to_onehot=2)])

    for i, (pet_brain, brain_labels) in enumerate(dataloader):
        pet_brain, brain_labels = pet_brain.to(device), brain_labels.to(device)
        mciCount = torch.sum(brain_labels == 1).item()
        ncCount = torch.sum(brain_labels == 0).item()
        mciweight = ncCount / (mciCount + ncCount)
        ncweight = mciCount / (mciCount + ncCount)
        weights = [ncweight, mciweight]
        weights = [w if w != 0.0 else 1e-4 for w in weights]
        class_weights = torch.FloatTensor(weights).to(device)

        brain_pet_out = brain_swin_transformer(pet_brain)
        brain_pet_out = brain_pet_out[-1]
        brain_pet_out = brain_pet_out.permute(0, 2, 1)
        adaptive_pool = nn.AdaptiveAvgPool1d(1)
        brain_pet_out = adaptive_pool(brain_pet_out)
        brain_pet_out = torch.flatten(brain_pet_out, 1)
        brain_pet_out = brain_classifier_head(brain_pet_out)

        optimizer.zero_grad()
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        brain_loss = criterion(brain_pet_out, brain_labels)
        brain_loss.backward()
        optimizer.step()

        running_loss += brain_loss.item()
        loss_epoch.append(brain_loss.item())

        label_real += [i for i in decollate_batch(brain_labels)]
        label_pet_pred += [pet_post_pred(i).detach().cpu().numpy().argmax() for i in decollate_batch(brain_pet_out)]
        AUC(y_pred=[pet_post_pred(i) for i in decollate_batch(brain_pet_out)], y=[pet_post_label(i) for i in decollate_batch(brain_labels, detach=False)])

    brain_pet_loss = loss_epoch.aggregate()
    cm_train_pet = ConfusionMatrix(actual_vector=label_real, predict_vector=label_pet_pred)
    logger.print_message(f"Epoch {epoch}")
    logger.print_message(
        f"Training    - Loss:{float(brain_pet_loss):.4f} "
        f"ACC:{float(cm_train_pet.Overall_ACC):.4f} "
        f"SEN:{float(list(cm_train_pet.TPR.values())[1]):.4f} "
        f"SPE:{float(list(cm_train_pet.TNR.values())[1]):.4f} "
        f"F1:{float(list(cm_train_pet.F1.values())[1]):.4f} "
        f"AUC:{AUC.aggregate():.4f}")

    return brain_pet_loss, float(AUC.aggregate()), cm_train_pet


def validate_epoch(brain_swin_transformer, brain_classifier_head, dataloader, device, logger):
    brain_swin_transformer.eval()
    running_loss = 0.0
    label_real = []
    label_pet_pred = []
    loss_epoch = CumulativeAverage()
    AUC = ROCAUCMetric(average='macro')
    pet_post_pred = Compose([Activations(softmax=True)])
    pet_post_label = Compose([AsDiscrete(to_onehot=2)])

    with torch.no_grad():
        for i, (pet_brain, brain_labels) in enumerate(dataloader):
            pet_brain, brain_labels = pet_brain.to(device), brain_labels.to(device)

            brain_pet_out = brain_swin_transformer(pet_brain)
            brain_pet_out = brain_pet_out[-1]
            brain_pet_out = brain_pet_out.permute(0, 2, 1)
            adaptive_pool = nn.AdaptiveAvgPool1d(1)
            brain_pet_out = adaptive_pool(brain_pet_out)
            brain_pet_out = torch.flatten(brain_pet_out, 1)
            brain_pet_out = brain_classifier_head(brain_pet_out)

            criterion = nn.CrossEntropyLoss()
            brain_loss = criterion(brain_pet_out, brain_labels)

            running_loss += brain_loss.item()
            loss_epoch.append(brain_loss.item())

            label_real += [i for i in decollate_batch(brain_labels)]
            label_pet_pred += [pet_post_pred(i).detach().cpu().numpy().argmax() for i in decollate_batch(brain_pet_out)]
            AUC(y_pred=[pet_post_pred(i) for i in decollate_batch(brain_pet_out)], y=[pet_post_label(i) for i in decollate_batch(brain_labels, detach=False)])

    brain_pet_loss = loss_epoch.aggregate()
    cm_val_pet = ConfusionMatrix(actual_vector=label_real, predict_vector=label_pet_pred)
    logger.print_message(
        f"Validation  - Loss:{float(brain_pet_loss):.4f} "
        f"ACC:{float(cm_val_pet.Overall_ACC):.4f} "
        f"SEN:{float(list(cm_val_pet.TPR.values())[1]):.4f} "
        f"SPE:{float(list(cm_val_pet.TNR.values())[1]):.4f} "
        f"F1:{float(list(cm_val_pet.F1.values())[1]):.4f} "
        f"AUC:{AUC.aggregate():.4f}")

    return brain_pet_loss, float(AUC.aggregate()), cm_val_pet


def main():
    set_seed(42)
    best_val_auc = 0.0

    para_name = f"fold_{args.fold_index}_lr_{args.lr}_wd_{args.wd}_eh_{args.enhance_p}"
    save_dir = os.path.join(args.output, 'New_SwinGT' + args.save_file_name, para_name)
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

    best_model_path = os.path.join(save_dir, f"best_model_{args.save_file_name}.ckpt")
    folder_name = ("fold " + str(args.fold_index) + ",embed_dim " + str(args.embed_dim) +
                   ",depth " + str(args.depths) + ",num_heads " + str(args.num_heads) +
                   ",window_size " + str(args.window_size) + ",patch_size " + str(args.patch_size) +
                   ",batch_size " + str(args.batch_size) + ",mlp_ratio " + str(args.mlp_ratio) +
                   ",drop_rate " + str(args.drop_rate) + ",lr " + str(args.lr) +
                   ",weight_decay " + str(args.wd) + ",enhance_p " + str(args.enhance_p))

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    transform = transforms.Compose([transforms.RandomApply([transforms.RandomAffine(degrees=30, translate=(0.1, 0.1), scale=(0.9, 1.1))], p=args.enhance_p)])

    train_dataset = GeneralDataset_brain(args.brain_pth_path, args.brain_fold_path, args.rest_brain_fold_path, args.fold_index, "train", transform=transform)
    dataloader_train = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)

    val_dataset = GeneralDataset_brain(args.brain_pth_path, args.brain_fold_path, args.rest_brain_fold_path, args.fold_index, "val", transform=None)
    dataloader_val = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    brain_panswin = PanSwin(
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
        image_size=args.brain_size
    ).to(device)
    brain_classifier_head = MLP(768, 128, 2).to(device)
    brain_panswin = nn.DataParallel(brain_panswin)
    brain_classifier_head = nn.DataParallel(brain_classifier_head)
    params = list(brain_panswin.parameters()) + list(brain_classifier_head.parameters())
    optimizer = optim.AdamW(params=params, lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=args.wd)

    logger = Logger(save_dir, 'log.txt')

    for epoch in range(400):
        train_loss_results, train_auc, cm_train = train_epoch(epoch, brain_panswin, brain_classifier_head, dataloader_train, optimizer, device, logger)
        val_loss_results, val_auc, cm_val = validate_epoch(brain_panswin, brain_classifier_head, dataloader_val, device, logger)

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

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            save_checkpoint({
                'epoch': epoch,
                'brain_panswin': brain_panswin.module.state_dict(),
                'brain_classifier_head': brain_classifier_head.module.state_dict(),
                'optimizer': optimizer.state_dict(),
                'val_auc': val_auc,
                'train_auc': train_auc
            }, best_model_path)
            logger.print_message(f"Saved best model with validation AUC: {best_val_auc:.4f} at epoch {epoch}")

        if (epoch + 1) % 20 == 0:
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
    args = parser.parse_args()
    main()