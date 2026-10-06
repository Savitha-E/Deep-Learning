import numpy as np

data = np.load("/home/ibab/DL_data/File_handling/npy/pract.npy")

print(data.shape)
print(data[0])

#print first 5 smaple
print(data[:5])

#print a column
print(data[:, 3])

#remove a column
data_without_waste = np.delete(data, 3, axis=1)

print(data_without_waste.shape)
X = data_without_waste[:, 0:4]

y = data_without_waste[:, 4]
print("X:", X.shape)
print("y:", y.shape)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("X_train:", X_train.shape)
print("y_train:", y_train.shape)

print("X_test:", X_test.shape)
print("y_test:", y_test.shape)