import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.optim import Adam
import numpy as np

# fer_train = datasets.FER2013(root="/home/ibab/PycharmProjects/Sem3_DL/vision_datasets")
# print(fer_train)
# fer_test = datasets.FER2013(root="/home/ibab/PycharmProjects/Sem3_DL/vision_datasets", split="test")

lfw = datasets.ImageFolder(root="/home/ibab/data/lfw-py/lfw_funneled")
print(lfw.classes)

labels = [target for _, target in lfw]
print(len(np.unique(labels)))







