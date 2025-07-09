import os
import torch
from torch.utils.data import Dataset, DataLoader

class GeneralDataset_whole_body(Dataset):
    def __init__(self, feats_path, fold_path, fold_index=1, split='train', transform=None):
        self.feats_path_brain = os.path.join(feats_path, "Brain")
        self.feats_path_heart = os.path.join(feats_path, "Heart")
        self.feats_path_gut = os.path.join(feats_path, "Gut")
        self.fold_path = os.path.join(fold_path, str(fold_index), split)
        self.transform = transform
        self.samples = self._load_samples()

    def _load_samples(self):
        sample_list = []
        for class_folder in os.listdir(self.fold_path):
            class_folder_path = os.path.join(self.fold_path, class_folder)
            for subnamedir in os.listdir(class_folder_path):
                feats_brain = os.path.join(self.feats_path_brain, f"{subnamedir}.pth")
                feats_heart = os.path.join(self.feats_path_heart, f"{subnamedir}.pth")
                feats_gut = os.path.join(self.feats_path_gut, f"{subnamedir}.pth")
                sample_list.append((feats_brain, feats_heart, feats_gut, int(class_folder)))
        return sample_list

    def _get_sample_weights(self):
        class_counts = {}
        for *_, label in self.samples:
            class_counts[label] = class_counts.get(label, 0) + 1
        return [1.0 / class_counts[label] for *_, label in self.samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sub_brain, sub_heart, sub_gut, label = self.samples[index]

        feats = [
            torch.load(sub_brain),
            torch.load(sub_heart),
            torch.load(sub_gut)
        ]

        if self.transform:
            feats = [self.transform(f) for f in feats]

        feats = [f.unsqueeze(0).float() for f in feats]
        return (*feats, torch.tensor(label))


if __name__ == '__main__':
    feats_path = r'D:\DATA\3_crop_pth'
    fold_path = r'D:\DATA\fold_partition\wholebody\5fold\42'

    dataset = GeneralDataset_whole_body(
        feats_path=feats_path,
        fold_path=fold_path,
        fold_index=1,
        split="train"
    )

    dataloader = DataLoader(dataset, batch_size=5, shuffle=True, num_workers=2)

    for batch_idx, (brain, heart, gut, labels) in enumerate(dataloader):
        print(f'Brain Features shape: {brain.shape}', flush=True)
        print(f'Heart Features shape: {heart.shape}', flush=True)
        print(f'Gut Features shape: {gut.shape}', flush=True)
        print(f'Labels: {labels}', flush=True)