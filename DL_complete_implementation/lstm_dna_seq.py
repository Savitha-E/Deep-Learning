import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from torch.optim import Adam
import pandas as pd
import numpy as np

#
# # create dataset
# seq1 = "ACGTAGCTAGCT"
# seq2 = "GCTAGCTAGGCA"
# seq3 = "CGTACGTAGCTA"
# seq4 = "TGCATGCATGCA"
#
# seq_list = [seq1, seq2, seq3, seq4]
# seq_dict_list = []
# for seq in seq_list:
#     for _ in range(100):
#         seq_dict = {"Sequence": seq, "Label": 0 if seq == seq1 or seq == seq3 else 1}
#         seq_dict_list.append(seq_dict)
# print(len(seq_dict_list))
#
# seq_df = pd.DataFrame(seq_dict_list)
# seq_df.to_csv("seq_rnn_data.csv")

# create vocab
# vocab = {"A": 0,
#          "T": 1,
#          "G": 2,
#          "C": 3}
vocab = {"START": 0,
         "A": 1,
         "T": 2,
         "G": 3,
         "C": 4,
         "END": 5}


def tokenize(seq):
    i = 0
    tokens = []
    while i < len(seq):
        if i + 5 <= len(seq) and seq[i:i + 5] == "START":
            tokens.append("START")
            i += 5
        elif i + 3 <= len(seq) and seq[i:i + 3] == "END":
            tokens.append("END")
            i += 3
        else:
            tokens.append(seq[i])
            i += 1
    return tokens


def seq2idxarray(seq):
    tokens = tokenize(seq)
    idx_list = []
    for base in tokens:
        idx_list.append(vocab[base])
    idx_array = np.array(idx_list)
    return idx_array


# def seq2idxarray(seq):
#     idx_list = []
#     for i in seq:
#         idx_list.append(vocab[i])
#     idx_array = np.array(idx_list)
#     return idx_array


# print(seq2idxarray("STARTTGCATGCATGCAEND"))

# append "START" and "END" to each of the seqs
seq_df = pd.read_csv("seq_rnn_data.csv")
seq_df = seq_df.sample(frac=1, random_state=16).reset_index(drop=True)

for i in range(len(seq_df)):
    seq_df.iloc[i, 1] = "START" + seq_df.iloc[i, 1] + "END"

print(seq_df)

torch.manual_seed(16)


class SeqDataset(Dataset):
    def __init__(self, df):
        self.seqs = df["Sequence"].apply(func=seq2idxarray).values
        self.labels = torch.tensor(df["Label"].values, dtype=torch.int64)

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        self.sequence_array = self.seqs[idx]
        self.sequence_tensor = torch.tensor(self.sequence_array, dtype=torch.int64)
        self.label = self.labels[idx]
        return self.sequence_tensor, self.label


seq_dataset = SeqDataset(seq_df)

train_length = int(0.8 * len(seq_df["Sequence"]))
test_length = int(seq_dataset.__len__() - train_length)

train_dataset, test_dataset = random_split(seq_dataset, lengths=[train_length, test_length])

train_loader = DataLoader(train_dataset, shuffle=True, batch_size=40)
test_loader = DataLoader(test_dataset, shuffle=False, batch_size=40)


class myLSTM(nn.Module):
    def __init__(self, vocab_size, embed_size, output_size, hidden_size, num_layers, dropout):
        super(myLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(input_size=embed_size, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout,
                            batch_first=True)
        self.fc = nn.Linear(in_features=hidden_size, out_features=output_size)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x_embed = self.embedding(x)
        out, h = self.lstm(x_embed)
        out = out[:, -1,
              :]  # [:, -1, :] represents all seqs: last time step: entire output vector. We are basically using the entire output vector of the last time step and passing it to FC layer
        out = self.dropout(out)
        out = self.fc(out)

        return out


# initialize parameters

vocab_size = len(vocab)
embed_size = 16
output_size = 2
hidden_size = 64
num_layers = 1
dropout = 0.5

# initialize lstm

lstm = myLSTM(vocab_size=vocab_size, embed_size=embed_size, output_size=output_size, hidden_size=hidden_size,
              num_layers=num_layers, dropout=dropout)

# define criterion and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = Adam(params=lstm.parameters(), lr=0.001)

# training loop
num_epochs = 50

for epoch in range(num_epochs):
    lstm.train()
    total_loss = 0
    for sequence, label in train_loader:
        optimizer.zero_grad()
        pred = lstm(sequence)
        loss = criterion(pred, label)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
    if epoch % 5 == 0:
        print(f"[{epoch + 1}/{num_epochs}]: Training Loss = {total_loss / len(train_loader)}")

# testing loop:
lstm.eval()
test_loss, correct_preds = 0.0, 0.0
num_batches = len(test_loader)
num_dp = len(test_loader.dataset)

with torch.no_grad():
    for batch, (sequence, label) in enumerate(test_loader):
        preds = lstm(sequence)
        loss = criterion(preds, label)
        test_loss += loss.item()
        correct_preds += (preds.argmax(1) == label).type(torch.float).sum().item()
    avg_test_loss = test_loss / num_batches
    model_acc = correct_preds / num_dp
    print(f"Average Test Loss = {avg_test_loss}")
    print(f"Model Accuracy = {model_acc * 100}%")
