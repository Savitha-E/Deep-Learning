import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import spacy
import string


class CharDataset(Dataset):
    def __init__(self, text, char_to_idx,sequence_length=100):
        self.sequence_length = sequence_length
        self.text = text
        self.char_to_idx = char_to_idx
        self.encoded_text = [char_to_idx.get(c, 0) for c in text]  # Encode text with indices

    def __len__(self):
        return len(self.encoded_text) - self.sequence_length

    def __getitem__(self, idx):
        x = self.encoded_text[idx:idx + self.sequence_length]
        y = self.encoded_text[idx + 1:idx + self.sequence_length + 1]
        return torch.tensor(x), torch.tensor(y)
class CharLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, n_layers):
        super(CharLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        embedded = self.embedding(x)
        output, hidden = self.lstm(embedded, hidden)
        output = self.fc(output)
        return output, hidden
def main():
    # Load spaCy's English tokenizer and vocabulary
    nlp = spacy.blank("en")

    # Step 1: Create a character-level vocabulary
    # Start with lowercase letters, numbers, and punctuation
    characters = list(string.digits + string.punctuation + ' ')
    # Add unique characters from spaCy's vocab words

    # vocab = set()
    #
    # # Add characters from the `characters` list
    # for char in characters:
    #     vocab.add(char)
    #
    # #Add each character from words in `nlp.vocab` that are alphabetic
    # for word in nlp.vocab:
    #     if word.is_alpha:
    #         for char in word.text.lower():
    #             vocab.add(char)

    vocab = sorted(set(characters + [char for word in nlp.vocab if word.is_alpha for char in word.text.lower()]))
    vocab_size = len(vocab)

    # Character to index mappings
    char_to_idx = {ch: i for i, ch in enumerate(vocab)}
    idx_to_char = {i: ch for i, ch in enumerate(vocab)}

    with open('pizza.txt', 'r', encoding='utf-8') as file:  # Ensure utf-8 for compatibility with special characters
        text_data = file.read()
    dataset = CharDataset(text_data,char_to_idx)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    embedding_dim = 64
    hidden_dim = 128
    n_layers = 2

    # Initialize model, loss, and optimizer
    model = CharLSTM(vocab_size=vocab_size, embedding_dim=embedding_dim, hidden_dim=hidden_dim, n_layers=n_layers)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    def train():
        n_epochs = 10
        # for epoch in range(n_epochs):
        #     model.train()
        #     total_loss = 0
        #     hidden = None
        #     for inputs, targets in dataloader:
        #         optimizer.zero_grad()
        #
        #         # Forward pass
        #         outputs, hidden = model(inputs, hidden)
        #         hidden = tuple(
        #             [h.detach() for h in hidden])  # Detach hidden state to prevent backprop through entire history
        #
        #         # Reshape outputs and targets for CrossEntropyLoss
        #         outputs = outputs.view(-1, vocab_size)
        #         targets = targets.view(-1)
        #
        #         # Calculate loss and backpropagate
        #         loss = criterion(outputs, targets)
        #         loss.backward()
        #         optimizer.step()
        #
        #         total_loss += loss.item()
        #     print(f"Epoch {epoch + 1}, Loss: {total_loss / len(dataloader):.4f}")
        for epoch in range(n_epochs):
            model.train()
            total_loss = 0
            for inputs, targets in dataloader:
                optimizer.zero_grad()

                # Initialize hidden state based on the current batch size
                batch_size = inputs.size(0)
                hidden = (
                    torch.zeros(n_layers, batch_size, hidden_dim),
                    torch.zeros(n_layers, batch_size, hidden_dim)
                )

                # Forward pass
                outputs, hidden = model(inputs, hidden)

                # Detach hidden state to avoid backprop through entire sequence history
                hidden = tuple([h.detach() for h in hidden])

                # Reshape outputs and targets for CrossEntropyLoss
                outputs = outputs.view(-1, vocab_size)
                targets = targets.view(-1)

                # Calculate loss and backpropagate
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            print(f"Epoch {epoch + 1}, Loss: {total_loss / len(dataloader):.4f}")
        # Save the model's state_dict
        torch.save(model.state_dict(), "char_lstm_model.pth")

    # Sampling function for generating text from the model
    def generate_text(model, start_text="The", length=100):
        model.eval()
        chars = [char_to_idx[c] for c in start_text.lower()]
        hidden = None
        for _ in range(length):
            x = torch.tensor([chars[-1]]).unsqueeze(0)  # Create input tensor
            output, hidden = model(x, hidden)
            last_char = output.argmax(dim=2).item()
            chars.append(last_char)
        return ''.join(idx_to_char[idx] for idx in chars)

    train()


    # model.load_state_dict(torch.load("char_lstm_model.pth"))
    # model.eval()
    # Set the model to evaluation mode (this disables dropout and batch normalization, if present)

    # Generate and print sample text
    print("Generated text:", generate_text(model, start_text="temperature", length=280))


if __name__ == "__main__":
    main()

