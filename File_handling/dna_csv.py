import pandas as pd
import numpy as np

data = pd.read_csv("/home/ibab/DL_data/File_handling/dna/dna_practice.csv")

print(data)
print(data.shape)
print(data.columns)

sequences = data["sequence"]
y = data["target"]

print(sequences)
print(y)

encoding = {
    "A": [1, 0, 0, 0],
    "T": [0, 1, 0, 0],
    "G": [0, 0, 1, 0],
    "C": [0, 0, 0, 1]
}
def one_hot_encode(sequence):

    encoded = []

    for base in sequence:
        encoded.append(encoding[base])

    return np.array(encoded)

print(one_hot_encode("ATGC"))

#Encode all sequences
X = np.array([
    one_hot_encode(sequence)
    for sequence in sequences
])

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("X_train:", X_train.shape)
print("X_test:", X_test.shape)

print("y_train:", y_train.shape)
print("y_test:", y_test.shape)