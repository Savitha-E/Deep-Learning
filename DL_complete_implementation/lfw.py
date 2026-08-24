import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader

lfw_dataset = datasets.LFWPeople(root="/home/ibab/PycharmProjects/Sem3_DL/vision_datasets", split="train",
                                 transform=transforms.ToTensor())

print(np.unique(lfw_dataset.targets))

labels = [target for _, target in lfw_dataset]
print(len(torch.unique(torch.tensor(labels))))

lfw_loader = DataLoader(dataset=lfw_dataset, batch_size=100, shuffle=True)

for X, y in lfw_loader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break
