import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, random_split

image_transform = transforms.Compose([transforms.Grayscale(num_output_channels=3),
                                      transforms.Resize((224, 224)),
                                      transforms.ToTensor()])

caltech_dataset = datasets.Caltech101(root="/home/ibab/PycharmProjects/Sem3_DL/vision_datasets",
                                      transform=image_transform)

# Does not have in-built train test split. So we use Random Split

train_and_val_size = int(0.8 * len(caltech_dataset))
test_size = int(len(caltech_dataset) - train_and_val_size)

caltech_train_and_val, caltech_test = random_split(caltech_dataset, [train_and_val_size, test_size])

tran_size = int(0.7 * len(caltech_train_and_val))
val_size = int(len(caltech_train_and_val) - tran_size)
caltech_train, caltech_val = random_split(caltech_train_and_val, [tran_size, val_size])

# Basic dataset Information

print(caltech_train)
print(caltech_val)
print(caltech_test)
labels = [target for _, target in caltech_train]
print(len(np.unique(labels)))

train_loader = DataLoader(caltech_train, batch_size=100, shuffle=True)
val_loader = DataLoader(caltech_val, batch_size=100, shuffle=True)
test_loader = DataLoader(caltech_test, batch_size=100, shuffle=True)

for x, y in train_loader:
    print(f"Shape of Image[N, C, H, W]: {x.shape}")
    break
