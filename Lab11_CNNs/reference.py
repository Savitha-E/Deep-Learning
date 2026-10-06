
import torch                                   #provides tensors, device , automatic differentiation , nn functionality , model parameters
import torch.nn as nn                          # used for defining a network
import torch.nn.functional as F                # performs operations such as activation functions on a layer
import torch.optim as optim                    # imports optimization algorithms (adam optimiser)
from torchvision import datasets, transforms   # imports datasets and transforms
from torch.utils.data import DataLoader        # dataloader is used to feed data to the model in batches

# --------------------
# 1. Data preparation
# --------------------
transform = transforms.Compose([
    transforms.ToTensor(),                                                                            # Convert image to tensor
    transforms.Normalize((0.1307,), (0.3081,))                                             # Normalize MNIST tensor values using mean and SD
])

train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)        #trainset , will download and apply transform
test_dataset  = datasets.MNIST(root="./data", train=False, download=True, transform=transform)        # testset , will download and apply transform

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)                                 #loads the dataset and feeds it in batches of 64
test_loader  = DataLoader(test_dataset, batch_size=1000, shuffle=False)                               #loads the dataset and feeds it in batches of 1000

# --------------------
# 2. Define CNN
# --------------------
class BasicCNN(nn.Module):                                                                      #Here the basiccnn is the child class name. The parent class is torch.nn's module
    def __init__(self):                                                                         # init for the child class
        super(BasicCNN, self).__init__()                                                         #init for the parent class
                                                               # Conv layer: in_channels=1 (grayscale), out_channels=32, kernel=3x3
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool  = nn.MaxPool2d(2, 2)
        self.fc1   = nn.Linear(64 * 7 * 7, 128)                                         # After 2 pools, image is 7x7
        self.fc2   = nn.Linear(128, 10)                                       # 10 classes for MNIST

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))                                                               # [batch,32,14,14]  # x = self.conv1(x) ,  x = self.relu(x) ,  x = self.pool(x)
        x = self.pool(F.relu(self.conv2(x)))                                                                   # [batch,64,7,7]
        x = x.view(-1, 64 * 7 * 7)                                                                          # Flatten
        x = F.relu(self.fc1(x))                                                                          # Fully connected layer1
        x = self.fc2(x)                                                                                  # Fully connected layer 2
        return x                                                                                         # return scores of the 10 classes

# --------------------
# 3. Training setup
# --------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")               #This checks if we have gpu  , and use cuda if not use cpu
model = BasicCNN().to(device)                                                      # creates the instance of the class and also attaches it to the device so they use the same parameters.
optimizer = optim.Adam(model.parameters(), lr=0.001)                                # (Adam-optimization algo )optimises by updating the weights during the weights ,
                                                                                   # paremeters are us giving the adam algo all the parameters of the model and lr -stands for learning rate)
criterion = nn.CrossEntropyLoss()                                      #Loss function

# --------------------
# 4. Training loop
# --------------------
for epoch in range(3):                                                          # 3 epochs for demo  epoch - one complete pass through the training dataset
    
    model.train()                                                              #The model is being trained ,calledfor training,Bcz the train_loader said batch size is 64, data will be 64, every time 64 images iwll be sent to train model
    for batch_idx, (data, target) in enumerate(train_loader):              #batch idx tell the currrent batch it is running or training, data - 64 images (1 batch) , 1 batch = [64,1,28,28] 1 bcz it is grayscale image, target is 64 labels  , enumaertaor matches the batch idx with the data and target respective to that batch
        data, target = data.to(device), target.to(device)                  #moves the target and data to gpu is available using cuda device or remian as cpu

        optimizer.zero_grad()                                               # This stores the gradients for the previous batch and then updates the gradients for the current batch
        output = model(data)                                                # forward propogation of one whole batch
        loss = criterion(output, target)                                    #Calculates the loss of the output
        loss.backward()                                                     # Back propogation calculating gradients for that batch with respect to model parameters
        optimizer.step()                                                    # Gradients are calculated and these are updated as model parameters

        if batch_idx % 100 == 0:                                             # for every 100 batches print the training information
            print(f"Epoch {epoch} [{batch_idx*len(data)}/{len(train_loader.dataset)}] Loss: {loss.item():.4f}")
                                                                              #prints the epoch , batch idx, len of the data (64) len of train_loader.dataset - (60000) and loss=
# --------------------
# 5. Evaluation
# --------------------
model.eval()                               # training is complete and now we are evaluating
correct = 0                                # This keeps track of the number of images that it correctly classifies
test_loss = 0                              # the accumulated loss obtained from the test batches

with torch.no_grad():                                                  #  only testing model , no need for calculation of gradients
    for data, target in test_loader:                                   # for the images ( data - 1 batch) and target respective to that batch -
        data, target = data.to(device), target.to(device)              # moves the respective data and target to gpu
        output = model(data)                                           # forward pass
        test_loss += criterion(output, target).item()                  # calculate the loss for this test batch(sum of losses)
        pred = output.argmax(dim=1, keepdim=True)                      # Based on the classes that are there in the output , which is the class that has highest probability, Keep dim- retain the dimesnions instea dof 1000 , [1000,1]
        correct += pred.eq(target.view_as(pred)).sum().item()          #  compares the prediction with the actual label, target view brings the shap e of teh predicted to same as actaul label so they can be compared
                                                                       # correct adds the number of correct predictions to the overall total.

test_loss /= len(test_loader)
accuracy = 100. * correct / len(test_loader.dataset)
                                                                        # accuracy= number of predictions / total number of test sample * 100
print(f"\nTest set: Average loss {test_loss:.4f}, Accuracy {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)\n")
                                                                        #This prints the test loss , number of correct predictions , the total and the accuracy.