
#
# Steps:
# 1. Load 3190 labelled DNA sequences
# 2. Split data into 80% training and 20% testing
# 3. One-hot encode DNA sequences
# 4. Construct a two fully-connected-layer neural network
# 5. Train for at least 5 epochs
# 6. Print training loss at the end of every epoch
# 7. Calculate test accuracy and F1-score
#
# ============================================================



import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import TensorDataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score




df = pd.read_csv(
    "splice.data",
    sep=r"\s+",
    header=None,
    names=["class", "id", "sequence"],
    engine="python"
)


# Extract input and target

X = df["sequence"].astype(str)
y = df["class"].astype(str)


print("Dataframe shape:", df.shape)
print("Input X shape:", X.shape)
print("Target y shape:", y.shape)

print("\nFirst 5 rows:")
print(df.head())

print("\nClasses:")
print(y.unique())



X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))




label_encoder = LabelEncoder()

y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)


print("\nClass mapping:")

for class_name, class_number in zip(
    label_encoder.classes_,
    range(len(label_encoder.classes_))
):
    print(class_name, "->", class_number)



num_classes = len(label_encoder.classes_)



base_to_number = {
    "A": 0,
    "C": 1,
    "G": 2,
    "T": 3
}


def encode_sequences(sequences):

    encoded_sequences = []

    for sequence in sequences:


        sequence = sequence.upper()

        one_hot_sequence = []

        for base in sequence:


            vector = [0, 0, 0, 0]


            if base in base_to_number:

                vector[base_to_number[base]] = 1



            one_hot_sequence.append(vector)


        one_hot_sequence = np.array(
            one_hot_sequence,
            dtype=np.float32
        )

        one_hot_sequence = one_hot_sequence.flatten()

        encoded_sequences.append(one_hot_sequence)

    return np.array(
        encoded_sequences,
        dtype=np.float32
    )


X_train_encoded = encode_sequences(X_train)
X_test_encoded = encode_sequences(X_test)


print("\nEncoded training shape:", X_train_encoded.shape)
print("Encoded testing shape:", X_test_encoded.shape)




X_train_tensor = torch.tensor(
    X_train_encoded,
    dtype=torch.float32
)

X_test_tensor = torch.tensor(
    X_test_encoded,
    dtype=torch.float32
)


y_train_tensor = torch.tensor(
    y_train_encoded,
    dtype=torch.long
)

y_test_tensor = torch.tensor(
    y_test_encoded,
    dtype=torch.long
)




train_dataset = TensorDataset(
    X_train_tensor,
    y_train_tensor
)

test_dataset = TensorDataset(
    X_test_tensor,
    y_test_tensor
)




batch_size = 32


train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True
)


test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False
)


print("\nNumber of input features:", X_train_tensor.shape[1])
print("Training batch size:", batch_size)
print("Number of training batches:", len(train_loader))
print("Number of testing batches:", len(test_loader))




class TwoLayerNetwork(nn.Module):

    def __init__(self, input_features, num_classes):

        super().__init__()


        self.layer1 = nn.Linear(
            input_features,
            128
        )



        self.relu = nn.ReLU()



        self.dropout = nn.Dropout(
            p=0.2
        )



        self.layer2 = nn.Linear(
            128,
            num_classes
        )


    def forward(self, x):


        x = self.layer1(x)


        x = self.relu(x)


        x = self.dropout(x)


        x = self.layer2(x)


        return x




input_features = X_train_tensor.shape[1]


model = TwoLayerNetwork(
    input_features,
    num_classes
)


print("\nModel:")
print(model)



criterion = nn.CrossEntropyLoss()


# Adam optimizer

optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)




epochs = 5

loss_history = []


for epoch in range(epochs):



    model.train()

    running_loss = 0.0


    for batch_X, batch_y in train_loader:


        predictions = model(batch_X)


        loss = criterion(
            predictions