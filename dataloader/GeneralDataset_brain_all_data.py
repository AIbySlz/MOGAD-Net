import os
import torch
from torch.utils.data import Dataset
class GeneralDataset_brain(Dataset):
    def __init__(self, feats_path, fold_path, rest_fold_path, fold_index=1, split='train', transform=None):
        self.feats_path = os.path.join(feats_path, "Brain")
        self.fold_path = os.path.join(fold_path, str(fold_index), split)
        self.other_fold_path = os.path.join(rest_fold_path, str(fold_index), split)
        self.transform = transform
        self.samples = self._load_samples()

    def _load_samples(self):
        sample_list = []
        for class_folder in os.listdir(self.fold_path):
            class_folder_path = os.path.join(self.fold_path, class_folder)
            for subnamedir in os.listdir(class_folder_path):
                feats_brain_path = os.path.join(self.feats_path, subnamedir + ".pth")
                sample_list.append((feats_brain_path, int(class_folder)))

        for class_folder in os.listdir(self.other_fold_path):
            class_folder_path = os.path.join(self.other_fold_path, class_folder)
            for subnamedir in os.listdir(class_folder_path):
                feats_brain_path = os.path.join(self.feats_path, subnamedir + ".pth")
                sample_list.append((feats_brain_path, int(class_folder)))
        return sample_list

    def _get_sample_weights(self):
        class_counts = {}
        for _, label in self.samples:
            class_counts[label] = class_counts.get(label, 0) + 1
        return [1.0 / class_counts[label] for _, label in self.samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sub_sample, label = self.samples[index]
        feats_brain = torch.load(sub_sample)
        if self.transform:
            feats_brain = self.transform(feats_brain)
        feats_brain = feats_brain.unsqueeze(0).float()
        return feats_brain, torch.tensor(label)


class GeneralDataset_brain_mri(Dataset):
    def __init__(self, feats_path, fold_path, fold_index=1, split='train', transform=None):
        self.feats_path = feats_path
        self.fold_path = os.path.join(fold_path, str(fold_index), split)
        self.transform = transform
        self.samples = self._load_samples()

    def _load_samples(self):
        sample_list = []
        for class_folder in os.listdir(self.fold_path):
            class_folder_path = os.path.join(self.fold_path, class_folder)
            for subnamedir in os.listdir(class_folder_path):
                feats_brain_path = os.path.join(self.feats_path, subnamedir + ".pth")
                sample_list.append((feats_brain_path, int(class_folder)))
        return sample_list

    def _get_sample_weights(self):
        class_counts = {}
        for _, label in self.samples:
            class_counts[label] = class_counts.get(label, 0) + 1
        return [1.0 / class_counts[label] for _, label in self.samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sub_sample, label = self.samples[index]
        feats_brain = torch.load(sub_sample)
        if self.transform:
            feats_brain = self.transform(feats_brain)
        feats_brain = feats_brain.unsqueeze(0).float()
        return feats_brain, torch.tensor(label)


# if __name__ == '__main__':
#     feats_path = r'D:\DATA\MNI\FDG_crop1.5\crop_pth'
#     fold_path_brain = r'D:\DATA\MNI\MRI_fold_partition\5fold\42'
#     other_fold_path_brain = r'D:\DATA\MNI\MRI_fold_partition\5fold\42'
#
#
#     split = "train"
#     fold_index = 1
#
#     dataset_brain = GeneralDataset_brain(feats_path, fold_path_brain, other_fold_path_brain, fold_index, split, transform=None)
#
#     dataloader_brain = DataLoader(dataset_brain, batch_size=5, shuffle=True, num_workers=2)
#
#     for batch_idx, (feats_brain, labels) in enumerate(dataloader_brain):
#         print(f'brain Features shape: {feats_brain.shape}', flush=True)


