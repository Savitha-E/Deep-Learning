import tarfile

import numpy as np


data = np.load("/home/ibab/DL_data/File_handling/npz/pract.npz")

print(data.files)
print(data["data"].shape)
print(data["column_names"])
print(data["target_column"])
print(data["waste_column"])

X = data["data"]
print(X.shape)
print(X[0]) # first row
print(X[:5]) # first 5 rows
print(X[:, 3]) # look at a column


#Drop column

X_without_waste = np.delete(X, 3, axis=1) #4th column , axis 1 means column
print(X_without_waste.shape)
print(X_without_waste[0])


print("Before:")
print(X[0])

print("After:")
print(X_without_waste[0])

#only features
X_features = X_without_waste[:, 0:4]
print(X_features.shape)
print(X_features[0])

#only target
X_target = X_without_waste[:, 4 ]
print(X_target.shape)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X_features,
    X_target,
    test_size=0.2,
    random_state=42
)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)
