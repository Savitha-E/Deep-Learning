# from torchvision.datasets import LFWPeople
# from torchvision.datasets import FashionMNIST
from torchvision import models
import torch
from torchvision.datasets import Flickr8k

Flickr8k(root="/home/ibab/PycharmProjects/Sem3_DL/vision_datasets", ann_file="/home/ibab/PycharmProjects/Sem3_DL/vision_datasets/flickr8k_ann_file.txt")

# LFWPeople(root="/home/ibab/PycharmProjects/Sem3_DL/vision_datasets", download=True)
# FashionMNIST(root="/home/ibab/PycharmProjects/Sem3_DL/vision_datasets", download=True)

# Load the model with pre-trained weights (this requires internet the first time)
# model = models.resnet18(pretrained=True)

# Save the model weights to a local file
# torch.save(model.state_dict(), '/home/ibab/PycharmProjects/Sem3_DL/vision_datasets/resnet18_pretrained.pth')

