# 🧠 MOGAD-Net: Multi-organ Guided Alzheimer's Diagnosis Network

**Multi-organ guided early diagnosis of Alzheimer’s disease via hierarchical alignment and knowledge distillation**

---

## 📜 Overview

**MOGAD-Net** is a deep learning framework designed to enable early detection of Alzheimer's disease by integrating pathological features from the brain, heart, and gut using whole-body PET imaging. The architecture consists of three progressive phases:

- **Pretraining phase:** Trains a baseline classifier on brain PET images to distinguish cognitively normal (NC) individuals from those with mild cognitive impairment (MCI). The outputs are used to generate pseudo-labels for unlabeled samples.
- **Phase 1:** Establishes a semi-supervised multi-organ collaboration framework. The brain branch’s predictions guide the training of heart and gut branches, enabling inter-organ alignment and feature fusion.
- **Phase 2:** Transfers diagnostic knowledge from the multi-organ network to a brain-only model (using PET or MRI), enhancing clinical applicability when peripheral organ data are unavailable.
![Framework of the proposed MOGAD-Net for training and inference phases.](https://github.com/AIbySlz/MOGAD-Net/blob/main/figure/Framework.jpg)
---

## 🛠️ Installation

Install PyTorch and torchvision from http://pytorch.org and other dependencies. You can install all the dependencies by
```
pip install -r requirements.txt
```

---

## 📁 Folder Structure
```
MOGAD-Net
  ├─ dataloader 
  │   ├─ GeneralDataset_brain_all_data.py <pretraining phase> 
  │   ├─ GeneralDataset_brain_data.py <phase 2>
  │   ├─ GeneralDataset_whole_body.py <phase 1>
  │   └─ GeneralDataset_whole_body_and_brain.py <phase 1，phase 2>
  ├─ model
  │   ├─ module.py 
  │   ├─ PanSwin.py
  │   └─ TGIC.py
  ├─ MOGAD_Net_phase 1
  │   ├─ test.py 
  │   └─ train.py 
  ├─ MOGAD_Net_phase 2
  │   ├─ test.py
  │   └─ train.py 
  ├─ MOGAD_Net_pretraining
  │   ├─ test.py
  │   └─ train.py
  ├─README.md
  └─requirements.txt
```
