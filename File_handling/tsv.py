import pandas as pd

# ============================================================
# 1. Load TSV file
# ============================================================

data = pd.read_csv(
    "/home/ibab/DL_data/File_handling/tsv/pract.tsv",
    sep="\t"
)

print(data)


# ============================================================
# 2. Inspect the dataset
# ============================================================

print(data.shape)

print(data.columns)

print(data.head())


# ============================================================
# 3. Look at individual columns
# ============================================================

print(data["chromosome"])

print(data["start"])

print(data["gene"])

print(data["target"])


# ============================================================
# 4. Drop columns
# ============================================================

# Drop text columns that we are not using as numerical
# features in this exercise

data_without_text = data.drop(
    columns=["chromosome", "gene"]
)

print(data_without_text)


# ============================================================
# 5. Drop the waste column
# ============================================================

data_without_waste = data_without_text.drop(
    columns=["waste_column"]
)

print(data_without_waste)

print(data_without_waste.shape)


# ============================================================
# 6. Separate features
# ============================================================

X_features = data_without_waste[
    ["start", "end", "score"]
]

print("X shape:")
print(X_features.shape)

print(X_features.head())


# ============================================================
# 7. Separate target
# ============================================================

y = data_without_waste["target"]

print("y shape:")
print(y.shape)

print(y.head())


# ============================================================
# 8. Convert X from pandas DataFrame to NumPy array
# ============================================================

X_features = X_features.to_numpy()

y = y.to_numpy()

print("X NumPy shape:")
print(X_features.shape)

print("y NumPy shape:")
print(y.shape)


# ============================================================
# 9. Train/test split
# ============================================================

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X_features,
    y,
    test_size=0.2,
    random_state=42
)


# ============================================================
# 10. Check final shapes
# ============================================================

print("X_train:", X_train.shape)
print("y_train:", y_train.shape)

print("X_test:", X_test.shape)
print("y_test:", y_test.shape)