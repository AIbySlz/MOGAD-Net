import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from collections import Counter
from itertools import chain
import random

class GeneralDataset(Dataset):
    def __init__(self, feats_path, unlabeled_fold_path, labeled_fold_path, fold_index=1, split='train', transform=None):
        self.labeled_feats_path_brain = os.path.join(feats_path, "Brain")
        self.labeled_feats_path_heart = os.path.join(feats_path, "Heart")
        self.labeled_feats_path_gut = os.path.join(feats_path, "Gut")

        self.unlabeled_fold_path = os.path.join(unlabeled_fold_path, str(fold_index), split)
        self.labeled_fold_path = os.path.join(labeled_fold_path, str(fold_index), split)

        self.transform = transform
        self.unlabeled_samples_whole_body = self._load_unlabeled_samples()
        self.labeled_samples_whole_body = self._load_labeled_samples()
        self.used_unlabeled_indices = set()

    def _load_unlabeled_samples(self):
        sample_list = []
        for class_folder in os.listdir(self.unlabeled_fold_path):
            class_folder_path = os.path.join(self.unlabeled_fold_path, class_folder)
            for subnamedir in os.listdir(class_folder_path):
                feats_brain_path = os.path.join(self.labeled_feats_path_brain, subnamedir + ".pth")
                sample_list.append((feats_brain_path, int(class_folder)))
        return sample_list

    def _load_labeled_samples(self):
        sample_list = []
        for class_folder in os.listdir(self.labeled_fold_path):
            class_folder_path = os.path.join(self.labeled_fold_path, class_folder)
            for subnamedir in os.listdir(class_folder_path):
                feats_brain_path = os.path.join(self.labeled_feats_path_brain, subnamedir + ".pth")
                feats_heart_path = os.path.join(self.labeled_feats_path_heart, subnamedir + ".pth")
                feats_gut_path = os.path.join(self.labeled_feats_path_gut, subnamedir + ".pth")
                sample_list.append((feats_brain_path, feats_heart_path, feats_gut_path, int(class_folder)))
        return sample_list

    def __len__(self):
        return len(self.unlabeled_samples_whole_body)

    def __getitem__(self, index):
        sub_brain, label = self.unlabeled_samples_whole_body[index]
        feats_brain = torch.load(sub_brain)
        if self.transform:
            feats_brain = self.transform(feats_brain)
        feats_brain = feats_brain.unsqueeze(0).float()
        label_tensor = torch.tensor(label)
        return {'brain': feats_brain, 'label': label_tensor, 'is_multimodal': False, 'dataset': self}

    def __getitem_whole_body__(self, index):
        sub_brain, sub_heart, sub_gut, label = self.labeled_samples_whole_body[index]
        feats_brain = torch.load(sub_brain)
        feats_heart = torch.load(sub_heart)
        feats_gut = torch.load(sub_gut)
        if self.transform:
            feats_brain = self.transform(feats_brain)
            feats_heart = self.transform(feats_heart)
            feats_gut = self.transform(feats_gut)
        feats_brain = feats_brain.unsqueeze(0).float()
        feats_heart = feats_heart.unsqueeze(0).float()
        feats_gut = feats_gut.unsqueeze(0).float()
        label_tensor = torch.tensor(label)
        return {'brain': feats_brain, 'heart': feats_heart, 'gut': feats_gut, 'label': label_tensor,
                'is_multimodal': True, 'dataset': self}

    def get_unique_sample(self, label):
        indices = [i for i in range(len(self.labeled_samples_whole_body)) if i not in self.used_unlabeled_indices]
        while indices:
            index = random.choice(indices)
            indices.remove(index)
            data = self.__getitem_whole_body__(index)
            if data['label'].item() == label:
                self.used_unlabeled_indices.add(index)
                return data
        self.used_unlabeled_indices.clear()
        return self.get_unique_sample(label)

def custom_collate_fn(batch):
    single_modal_data = [item for item in batch if not item['is_multimodal']]

    dataset = batch[0]['dataset']

    # 确保每个标签的多模态数据至少有2个
    multi_modal_data_0 = [dataset.get_unique_sample(0) for _ in range(1)]
    multi_modal_data_1 = [dataset.get_unique_sample(1) for _ in range(1)]

    sampled_multi_modal = multi_modal_data_0 + multi_modal_data_1

    combined_batch = sampled_multi_modal + single_modal_data
    random.shuffle(combined_batch)

    return combined_batch


class GeneralDatasetBHG(Dataset):
    def __init__(self, unlabeled_feats_path, labeled_feats_path, unlabeled_fold_path, labeled_fold_path, fold_index=1, split='train', transform=None):
        self.unlabeled_feats_path_brain = os.path.join(unlabeled_feats_path, "Brain")
        self.unlabeled_feats_path_heart = os.path.join(unlabeled_feats_path, "Heart")
        self.unlabeled_feats_path_gut = os.path.join(unlabeled_feats_path, "Gut")

        self.labeled_feats_path_brain = os.path.join(labeled_feats_path, "Brain")
        self.labeled_feats_path_heart = os.path.join(labeled_feats_path, "Heart")
        self.labeled_feats_path_gut = os.path.join(labeled_feats_path, "Gut")

        self.unlabeled_fold_path = os.path.join(unlabeled_fold_path, str(fold_index), split)
        self.labeled_fold_path = os.path.join(labeled_fold_path, str(fold_index), split)

        self.transform = transform
        self.unlabeled_samples_whole_body = self._load_unlabeled_samples()
        self.labeled_samples_whole_body = self._load_labeled_samples()
        self.used_unlabeled_indices = set()

    def _load_unlabeled_samples(self):
        sample_list = []
        for class_folder in os.listdir(self.unlabeled_fold_path):
            class_folder_path = os.path.join(self.unlabeled_fold_path, class_folder)
            for subnamedir in os.listdir(class_folder_path):
                feats_brain_path = os.path.join(self.unlabeled_feats_path_brain, subnamedir + ".pth")
                feats_heart_path = os.path.join(self.unlabeled_feats_path_heart, subnamedir + ".pth")
                feats_gut_path = os.path.join(self.unlabeled_feats_path_gut, subnamedir + ".pth")
                sample_list.append((feats_brain_path, feats_heart_path, feats_gut_path, int(class_folder)))
        return sample_list

    def _load_labeled_samples(self):
        sample_list = []
        for class_folder in os.listdir(self.labeled_fold_path):
            class_folder_path = os.path.join(self.labeled_fold_path, class_folder)
            for subnamedir in os.listdir(class_folder_path):
                feats_brain_path = os.path.join(self.labeled_feats_path_brain, subnamedir + ".pth")
                feats_heart_path = os.path.join(self.labeled_feats_path_heart, subnamedir + ".pth")
                feats_gut_path = os.path.join(self.labeled_feats_path_gut, subnamedir + ".pth")
                sample_list.append((feats_brain_path, feats_heart_path, feats_gut_path, int(class_folder)))
        return sample_list

    def __len__(self):
        return len(self.unlabeled_samples_whole_body)

    def __getitem__(self, index):
        sub_brain, sub_heart, sub_gut, label = self.unlabeled_samples_whole_body[index]
        feats_brain = torch.load(sub_brain)
        feats_heart = torch.load(sub_heart)
        feats_gut = torch.load(sub_gut)
        if self.transform:
            feats_brain = self.transform(feats_brain)
            feats_heart = self.transform(feats_heart)
            feats_gut = self.transform(feats_gut)
        feats_brain = feats_brain.unsqueeze(0).float()
        feats_heart = feats_heart.unsqueeze(0).float()
        feats_gut = feats_gut.unsqueeze(0).float()
        label_tensor = torch.tensor(label)
        return {'brain': feats_brain, 'heart': feats_heart, 'gut': feats_gut, 'label': label_tensor,
                'is_labeled': False, 'dataset': self}

    def __getitem_whole_body__(self, index):
        sub_brain, sub_heart, sub_gut, label = self.labeled_samples_whole_body[index]
        feats_brain = torch.load(sub_brain)
        feats_heart = torch.load(sub_heart)
        feats_gut = torch.load(sub_gut)
        if self.transform:
            feats_brain = self.transform(feats_brain)
            feats_heart = self.transform(feats_heart)
            feats_gut = self.transform(feats_gut)
        feats_brain = feats_brain.unsqueeze(0).float()
        feats_heart = feats_heart.unsqueeze(0).float()
        feats_gut = feats_gut.unsqueeze(0).float()
        label_tensor = torch.tensor(label)
        return {'brain': feats_brain, 'heart': feats_heart, 'gut': feats_gut, 'label': label_tensor,
                'is_labeled': True, 'dataset': self}

    def get_unique_sample(self, label):
        indices = [i for i in range(len(self.labeled_samples_whole_body)) if i not in self.used_unlabeled_indices]
        while indices:
            index = random.choice(indices)
            indices.remove(index)
            data = self.__getitem_whole_body__(index)
            if data['label'].item() == label:
                self.used_unlabeled_indices.add(index)
                return data
        self.used_unlabeled_indices.clear()
        return self.get_unique_sample(label)

def custom_collate_fn_bhg(batch):
    unlabeled_data = [item for item in batch if not item['is_labeled']]
    dataset = batch[0]['dataset']

    # 确保每个标签的数据至少有2个
    labeled_data_0 = [dataset.get_unique_sample(0) for _ in range(1)]
    labeled_data_1 = [dataset.get_unique_sample(1) for _ in range(1)]

    sampled_labeled_data = labeled_data_0 + labeled_data_1

    combined_batch = sampled_labeled_data + unlabeled_data
    random.shuffle(combined_batch)

    return combined_batch

