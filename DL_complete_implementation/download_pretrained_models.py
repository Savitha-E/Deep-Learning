import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, Dataset, random_split
from torch.optim import Adam
weights = models.ResNet18_Weights
resnet = models.resnet18(weights)
weights_vgg = models.VGG11_Weights
vgg11 = models.vgg11(weights_vgg)
inception_weights = models.Inception_V3_Weights
inception_net = models.inception_v3(inception_weights)

print(resnet)
print(vgg11)
print(inception_net)
