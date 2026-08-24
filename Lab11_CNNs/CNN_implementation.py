import torch # torch is used for neural netwroks ,cnn ,automatic differentiation, dl and GPU computing
import torch
from torchvision import datasets, transforms


transform = transforms.Compose([
    transforms.ToTensor()
])


train_dataset = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)


test_dataset = datasets.MNIST(
    root="./data",
    train=False,
    download=True,
    transform=transform
)


image, label = train_dataset[0]

print("Image shape:", image.shape)
print("Label:", label)
