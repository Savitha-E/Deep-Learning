# ================================================================
# B1. Implement a CNN model using PyTorch for image classification
#     using the CIFAR-10 dataset.
#     Use at least three convolutional layers.
#     Train for 5 epochs, print loss and training accuracy
#     periodically, calculate test accuracy, and compare with
#     a pre-trained ResNet18 model.
# ================================================================


# ================================================================
# IMPORT LIBRARIES
# ================================================================

# Import PyTorch
import torch

# Import neural network modules
import torch.nn as nn

# Import optimization algorithms
import torch.optim as optim

# Import CIFAR-10 dataset and image transformations
from torchvision import datasets, transforms, models

# Import DataLoader for creating batches
from torch.utils.data import DataLoader


# ================================================================
# SELECT DEVICE
# ================================================================

# Use GPU if available, otherwise use CPU
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ================================================================
# IMAGE TRANSFORMATION
# ================================================================

# Convert images into PyTorch tensors
# and normalize the pixel values
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        (0.5, 0.5, 0.5),
        (0.5, 0.5, 0.5)
    )
])


# ================================================================
# LOAD CIFAR-10 TRAINING DATASET
# ================================================================

train_dataset = datasets.CIFAR10(
    root="/home/ibab/Dl_datasets/DL_data/data",
    train=True,
    download=False,
    transform=transform
)


# ================================================================
# LOAD CIFAR-10 TESTING DATASET
# ================================================================

test_dataset = datasets.CIFAR10(
    root="/home/ibab/Dl_datasets/DL_data/data",
    train=False,
    download=False,
    transform=transform
)


# ================================================================
# CREATE DATALOADERS
# ================================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=128,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=128,
    shuffle=False
)


# ================================================================
# DEFINE CNN MODEL
# ================================================================

# Create our CNN class
class CNN(nn.Module):

    # ------------------------------------------------------------
    # CONSTRUCTOR
    # ------------------------------------------------------------

    def __init__(self):

        # Initialize parent class
        super(CNN, self).__init__()


        # --------------------------------------------------------
        # CONVOLUTIONAL LAYER 1
        # --------------------------------------------------------

        # CIFAR-10 images have 3 channels:
        # Red, Green and Blue
        #
        # 3 input channels → 32 feature maps
        self.conv1 = nn.Conv2d(
            3,
            32,
            kernel_size=3,
            padding=1
        )


        # --------------------------------------------------------
        # CONVOLUTIONAL LAYER 2
        # --------------------------------------------------------

        # 32 feature maps → 64 feature maps
        self.conv2 = nn.Conv2d(
            32,
            64,
            kernel_size=3,
            padding=1
        )


        # --------------------------------------------------------
        # CONVOLUTIONAL LAYER 3
        # --------------------------------------------------------

        # 64 feature maps → 128 feature maps
        self.conv3 = nn.Conv2d(
            64,
            128,
            kernel_size=3,
            padding=1
        )


        # --------------------------------------------------------
        # MAX POOLING
        # --------------------------------------------------------

        # Reduce the spatial dimensions
        self.pool = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )


        # --------------------------------------------------------
        # FULLY CONNECTED LAYER
        # --------------------------------------------------------

        # After three convolution + pooling stages:
        #
        # 32 × 32
        # ↓ pool
        # 16 × 16
        # ↓ pool
        # 8 × 8
        #
        # Final feature maps = 128
        #
        # Therefore:
        #
        # 128 × 8 × 8 = 8192
        self.fc1 = nn.Linear(
            128 * 4 * 4,
            256
        )


        # --------------------------------------------------------
        # OUTPUT LAYER
        # --------------------------------------------------------

        # CIFAR-10 has 10 classes
        self.fc2 = nn.Linear(
            256,
            10
        )


        # ReLU activation
        self.relu = nn.ReLU()


    # ============================================================
    # FORWARD PASS
    # ============================================================

    def forward(self, x):

        # First convolution
        x = self.relu(
            self.conv1(x)
        )

        # First pooling
        x = self.pool(x)


        # Second convolution
        x = self.relu(
            self.conv2(x)
        )

        # Second pooling
        x = self.pool(x)


        # Third convolution
        x = self.relu(
            self.conv3(x)
        )

        # Third pooling
        x = self.pool(x)


        # Flatten the feature maps
        x = x.view(
            x.size(0),
            -1
        )


        # Fully connected layer
        x = self.relu(
            self.fc1(x)
        )


        # Output layer
        x = self.fc2(x)

        return x


# ================================================================
# CREATE CNN MODEL
# ================================================================

model = CNN().to(device)


# ================================================================
# LOSS FUNCTION
# ================================================================

# CrossEntropyLoss is suitable for multi-class classification
criterion = nn.CrossEntropyLoss()


# ================================================================
# OPTIMIZER
# ================================================================

# Adam optimizer updates the network weights
optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)


# ================================================================
# TRAIN CNN
# ================================================================

# The question specifically asks for 5 epochs
num_epochs = 5


for epoch in range(num_epochs):

    # Put model into training mode
    model.train()

    # Track total loss
    running_loss = 0.0

    # Track correct predictions
    correct = 0

    # Track total training samples
    total = 0


    # ------------------------------------------------------------
    # TRAINING LOOP
    # ------------------------------------------------------------

    for images, labels in train_loader:

        # Move images to CPU/GPU
        images = images.to(device)

        # Move labels to CPU/GPU
        labels = labels.to(device)


        # Forward pass
        outputs = model(images)


        # Calculate loss
        loss = criterion(
            outputs,
            labels
        )


        # Clear previous gradients
        optimizer.zero_grad()


        # Backpropagation
        loss.backward()


        # Update weights
        optimizer.step()


        # Add batch loss
        running_loss += loss.item()


        # Find predicted class
        _, predicted = torch.max(
            outputs,
            1
        )


        # Count total samples
        total += labels.size(0)


        # Count correct predictions
        correct += (
            predicted == labels
        ).sum().item()


    # Calculate training accuracy
    train_accuracy = (
        100 * correct / total
    )


    # Calculate average loss
    average_loss = (
        running_loss / len(train_loader)
    )


    # Print results after every epoch
    print(
        f"Epoch [{epoch+1}/{num_epochs}] "
        f"Loss: {average_loss:.4f} "
        f"Training Accuracy: {train_accuracy:.2f}%"
    )


# ================================================================
# TEST CNN
# ================================================================

# Put model into evaluation mode
model.eval()

correct = 0
total = 0


# Disable gradient calculation
with torch.no_grad():

    for images, labels in test_loader:

        # Move data to device
        images = images.to(device)
        labels = labels.to(device)


        # Get predictions
        outputs = model(images)


        # Get predicted class
        _, predicted = torch.max(
            outputs,
            1
        )


        # Count samples
        total += labels.size(0)


        # Count correct predictions
        correct += (
            predicted == labels
        ).sum().item()


# Calculate test accuracy
cnn_accuracy = (
    100 * correct / total
)


# Print CNN test accuracy
print(
    f"CNN Test Accuracy: {cnn_accuracy:.2f}%"
)


# ================================================================
# PRE-TRAINED RESNET18
# ================================================================

# Load a pre-trained ResNet18 model
resnet = models.resnet18(
    weights=models.ResNet18_Weights.DEFAULT
)


# ---------------------------------------------------------------
# MODIFY FINAL LAYER
# ---------------------------------------------------------------

# ResNet18 was originally trained for 1000 ImageNet classes.
#
# CIFAR-10 has only 10 classes.
#
# Therefore, replace the final fully-connected layer.
resnet.fc = nn.Linear(
    resnet.fc.in_features,
    10
)


# Move ResNet to selected device
resnet = resnet.to(device)


# Define loss function
criterion_resnet = nn.CrossEntropyLoss()


# Define optimizer
optimizer_resnet = optim.Adam(
    resnet.parameters(),
    lr=0.001
)


# ================================================================
# TRAIN RESNET18
# ================================================================

# Train ResNet18 for 5 epochs
for epoch in range(5):

    # Set model to training mode
    resnet.train()

    running_loss = 0.0

    for images, labels in train_loader:

        # Move images and labels to device
        images = images.to(device)
        labels = labels.to(device)


        # Forward pass
        outputs = resnet(images)


        # Calculate loss
        loss = criterion_resnet(
            outputs,
            labels
        )


        # Clear old gradients
        optimizer_resnet.zero_grad()


        # Calculate gradients
        loss.backward()


        # Update weights
        optimizer_resnet.step()


        # Add loss
        running_loss += loss.item()


    # Print loss every epoch
    print(
        f"ResNet18 Epoch [{epoch+1}/5] "
        f"Loss: {running_loss / len(train_loader):.4f}"
    )


# ================================================================
# TEST RESNET18
# ================================================================

resnet.eval()

correct = 0
total = 0


with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)
        labels = labels.to(device)


        # Generate predictions
        outputs = resnet(images)


        # Get predicted class
        _, predicted = torch.max(
            outputs,
            1
        )


        # Count samples
        total += labels.size(0)


        # Count correct predictions
        correct += (
            predicted == labels
        ).sum().item()


# Calculate ResNet18 test accuracy
resnet_accuracy = (
    100 * correct / total
)


# Print ResNet18 accuracy
print(
    f"ResNet18 Test Accuracy: "
    f"{resnet_accuracy:.2f}%"
)


# ================================================================
# COMPARE MODELS
# ================================================================

print("\nModel Comparison")
print(
    f"CNN Test Accuracy     : "
    f"{cnn_accuracy:.2f}%"
)

print(
    f"ResNet18 Test Accuracy : "
    f"{resnet_accuracy:.2f}%"
)


# ================================================================
# IMPORT REQUIRED LIBRARIES
# ================================================================

# NumPy is used for numerical operations
import numpy as np

# PyTorch
import torch

# PyTorch neural-network module
import torch.nn as nn

# Optimizer
import torch.optim as optim

# Dataset and DataLoader utilities
from torch.utils.data import TensorDataset, DataLoader, random_split
