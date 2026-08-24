import pandas as pd
import numpy as np
import string
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from torch.optim import Adam

# load data:
movie_df = pd.read_csv('movie_reviews_data.csv')
print(movie_df)

# add start and end token to all reviews
for idx in range(len(movie_df)):
    movie_df.iloc[idx, 1] = "START" + movie_df.iloc[idx, 1] + "END"
print(movie_df)
max_len = len(max(movie_df["Review"], key=len)) - 6
movie_df["Label"] = np.where(movie_df["Label"] == "positive", 1, 0)
print(movie_df)
# create vocab
all_characters = string.ascii_letters + string.punctuation + string.whitespace
print(all_characters)

vocab = {all_characters[i]: i for i in range(len(all_characters))}
vocab["START"] = len(vocab)
vocab["END"] = len(vocab)
print(vocab)


def tokenize(text):
    tokens = []
    i = 0
    while i < len(text):
        if i + 5 <= len(text) and text[i:i + 5] == "START":
            tokens.append("START")
            i += 5
        elif i + 3 <= len(text) and text[i:i + 3] == "END":
            tokens.append("END")
            i += 3
        else:
            tokens.append(text[i])
            i += 1

    return tokens


def text2idxarray(text):
    tokens = tokenize(text)
    idx_array = np.zeros(shape=max_len)
    for i in range(len(tokens)):
        idx_array[i,] = vocab[tokens[i]]
    return idx_array


rev = "STARTThe movie was uninspired and lacks depth.END"
print(text2idxarray(rev))


# def positional_encoding(seq_len, d):
#         P = np.zeros(shape=(seq_len, d))
#         for k in range(seq_len):
#             for i in range(math.floor(d/2)):
#                 denominator = 10000 ** (2 * i/d)
#                 P[k, 2 * i] = np.sin(k / denominator)
#                 P[k, 2 * i + 1] = np.cos(k / denominator)
#         P_tensor = torch.tensor(P, dtype=torch.float64)
#         return P_tensor
#
#
# print(positional_encoding(10, 5))

class ReviewDataset(Dataset):
    def __init__(self, df):
        self.reviews = df["Review"].apply(func=text2idxarray)
        self.sentiment = df["Label"]

    def __len__(self):
        return len(self.reviews)

    def __getitem__(self, idx):
        review_tensor = torch.tensor(self.reviews[idx], dtype=torch.long)
        sentiment_tensor = torch.tensor(self.sentiment[idx], dtype=torch.long)

        return review_tensor, sentiment_tensor


movies_dataset = ReviewDataset(movie_df)
# dataloader = DataLoader(movies_dataset, batch_size=50, shuffle=True)
# for review, sentiment in dataloader:
#     t1 = review
#     t2 = sentiment
# print(len(movies_dataset))

train_size = int(0.7 * len(movies_dataset))
test_size = int(len(movies_dataset) - train_size)

train_dataset, test_dataset = random_split(movies_dataset, lengths=[train_size, test_size])

# print(len(train_dataset))
# print(len(test_dataset))

train_loader = DataLoader(train_dataset, batch_size=70, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=60, shuffle=False)


def positional_encoding(seq_len, d):
        P = np.zeros(shape=(seq_len, d))
        for k in range(seq_len):
            for i in range(d // 2):
                denominator = 10000 ** (2 * i/d)
                P[k, 2 * i] = np.sin(k / denominator)
                P[k, 2 * i + 1] = np.cos(k / denominator)
        P_tensor = torch.tensor(P, dtype=torch.float32)
        return P_tensor


class MyTransformer(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, max_seq_len, output_dim, dropout):
        super(MyTransformer, self).__init__()
        self.max_seq_len = max_seq_len
        self.d = embed_dim
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.encoding_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads)
        self.encoder = nn.TransformerEncoder(encoder_layer=self.encoding_layer, num_layers=num_layers)
        self.fc = nn.Linear(in_features=embed_dim, out_features=output_dim)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x_embed = self.embedding(x)
        x_pos_encoding = positional_encoding(self.max_seq_len, self.d)
        x_together = x_embed + x_pos_encoding.unsqueeze(
            0)  # match batch size. Initially pos_encoding is of shape (seq_len, embed_dim). now it is of shape (1, seq_len, embed_dim) and after addition, my vector will be of shape (batch_size, seq_len, embed_dim) which was is expected by my transformer
        x_together = self.dropout(x_together)  # shape : (batch_size, seq_len, embed_dim)
        x_encoded = self.encoder(x_together.permute(1, 0, 2))  # after permute shape is (seq_len, batch_size, embed_dim). This is what transformer needs
        x_encoded = x_encoded.mean(dim=0)  # Global Average Pooling: This step reduces the sequence to a single vector for each sample in the batch. shape : (batch_size, embed_dim)
        output = self.fc(x_encoded)

        return output


# initialise parameters, model, optimiser and loss function
num_heads = 4
vocab_size = len(vocab)
embed_dim = 512
num_layers = 4
num_classes = 2
dropout = 0.3
max_seq_len = max_len

transformer_model = MyTransformer(vocab_size=vocab_size, embed_dim=embed_dim, num_heads=num_heads, num_layers=num_layers,
                                  max_seq_len=max_seq_len, output_dim=num_classes, dropout=dropout)

optimizer = Adam(params=transformer_model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()

# training loop:
epochs = 10
for epoch in range(epochs):
    transformer_model.train()
    train_loss = 0.0
    for batch, (review, sentiment) in enumerate(train_loader):
        optimizer.zero_grad()
        pred = transformer_model(review)
        loss = loss_fn(pred, sentiment)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch + 1}/{epochs}: Training Loss = {train_loss / len(train_loader)}")




