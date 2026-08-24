import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using Device {device}')

# define Transforms for dataset (because models trained on ImageNet require 224x224 input)

train_transform = transforms.Compose([  # transforms.Grayscale(num_output_channels=3),  # Convert 1-channel to 3-channel

    transforms.Resize((224, 224)),  # Resizing to 224x224 for ImageNet models
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

test_and_val_transform = transforms.Compose(
    [  # transforms.Grayscale(num_output_channels=3),  # Convert 1-channel to 3-channel

        transforms.Resize((224, 224)),  # Resizing to 224x224 for ImageNet models
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])



# print(model)


flowers_train = datasets.Flowers102(root="/home/ibab/PycharmProjects/Sem3_DL/vision_datasets/", split="train",
                                    transform=train_transform)

flowers_val = datasets.Flowers102(root="/home/ibab/PycharmProjects/Sem3_DL/vision_datasets/", split="val",
                                  transform=test_and_val_transform)

flowers_test = datasets.Flowers102(root="/home/ibab/PycharmProjects/Sem3_DL/vision_datasets/", split="test",
                                   transform=test_and_val_transform)

print("Basic Dataset Information:")
print(flowers_train)
print(len(flowers_train))
print(flowers_test)
print(len(flowers_test))
print("Number of Classes: ")
# print(len(torch.unique(flowers_train._labels)))
labels = [target for _, target in flowers_train]
print(len(torch.unique(torch.tensor(labels))))

train_loader = DataLoader(dataset=flowers_train, batch_size=64, shuffle=True)
val_loader = DataLoader(dataset=flowers_val, batch_size=64, shuffle=True)
test_loader = DataLoader(dataset=flowers_test, batch_size=64, shuffle=True)

# initialize model from locally saved model

model = models.resnet18(pretrained=False)

# Load the saved weights from the local file
model.load_state_dict(torch.load("/home/ibab/PycharmProjects/Sem3_DL/vision_datasets/resnet18_pretrained.pth"))
print(model)

for param in model.parameters():
    param.requires_grad = False  # This freezes the layers, so they won't be updated during training

# Replace the final layer (ResNet has a fully connected layer called `fc`)
num_classes = len(torch.unique(torch.tensor(labels)))
model.fc = nn.Linear(in_features=model.fc.in_features, out_features=num_classes)

model = model.to(device)
print("Modified Arch:")
print(model)


# if you want to finetune model:
# for name, param in model.named_parameters():
#     if "layer4" in name or "fc" in name:  # Fine-tune last block (layer4) and fully connected layer (fc)
#         param.requires_grad = True
#     else:
#         param.requires_grad = False
#
# # Now, create the optimizer for only those layers that require gradients
# optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0001)

# else go ahead with the model as normal training and testing

