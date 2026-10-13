import os
import re
import random
from collections import Counter

import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

# 1. SETTINGS
DATA_DIR = r"/home/ibab/DL_data/RNN/Flickr 8k Dataset"
IMAGE_DIR = os.path.join(DATA_DIR, "Images")
EPOCHS = 5
BATCH_SIZE = 16
MAX_IMAGES = 1000   # Set to 0 to use all images
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 2. LOAD CAPTIONS
caption_file = os.path.join(DATA_DIR, "captions.txt")
if not os.path.exists(caption_file):
    caption_file = os.path.join(DATA_DIR, "Flickr8k.token.txt")

rows = []
with open(caption_file, encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if "\t" in line:                       # Flickr8k.token.txt format
            name, caption = line.split("\t", 1)
            name = name.split("#")[0]
        elif "," in line:                      # captions.csv format
            name, caption = line.split(",", 1)
            name = name.strip().strip('"')
            caption = caption.strip().strip('"')
            if name.lower() in ("image", "filename"):
                continue
        else:
            continue

        if os.path.isfile(os.path.join(IMAGE_DIR, name)):
            rows.append([name, caption])

data = pd.DataFrame(rows, columns=["image", "caption"])
if data.empty:
    raise ValueError("No captions found. Check DATA_DIR and caption file.")

# 3. USE A SMALL SUBSET FIRST
image_names = data.image.unique().tolist()
random.seed(42)
random.shuffle(image_names)
if MAX_IMAGES:
    data = data[data.image.isin(image_names[:MAX_IMAGES])].copy()

# 4. TEXT PREPROCESSING
def tokenize(text):
    return re.findall(r"[a-z0-9]+", text.lower())

counts = Counter()
for caption in data.caption:
    counts.update(tokenize(caption))

words = ["<PAD>", "<START>", "<END>", "<UNK>"]
words += [w for w, n in counts.items() if n >= 3]
word_to_id = {w: i for i, w in enumerate(words)}
id_to_word = {i: w for w, i in word_to_id.items()}
PAD, START, END, UNK = 0, 1, 2, 3
MAX_LEN = 22

def encode(caption):
    tokens = tokenize(caption)[:MAX_LEN - 2]
    ids = [START] + [word_to_id.get(w, UNK) for w in tokens] + [END]
    ids = ids[:MAX_LEN]
    return ids + [PAD] * (MAX_LEN - len(ids))

data["ids"] = data.caption.apply(encode)
print("Images:", data.image.nunique(), "| Captions:", len(data), "| Vocabulary:", len(words))

# 5. SPLIT BY IMAGE (so one image's captions don't appear in both sets)
names = data.image.unique().tolist()
random.shuffle(names)
cut = max(1, int(0.8 * len(names)))
train_names, test_names = set(names[:cut]), set(names[cut:])
train_data = data[data.image.isin(train_names)].reset_index(drop=True)
test_data = data[data.image.isin(test_names)].reset_index(drop=True)

# 6. PRETRAINED CNN: FROZEN FEATURE EXTRACTOR, NOT TRAINED
weights = models.ResNet18_Weights.DEFAULT
cnn = models.resnet18(weights=weights)
cnn.fc = nn.Identity()  # image becomes a 512-number vector
cnn = cnn.to(DEVICE).eval()
for p in cnn.parameters():
    p.requires_grad = False
image_transform = weights.transforms()

@torch.no_grad()
def get_features(path):
    image = Image.open(path).convert("RGB")
    image = image_transform(image).unsqueeze(0).to(DEVICE)
    return cnn(image).squeeze(0).cpu()

# 7. DATASET
class FlickrDataset(Dataset):
    def __init__(self, frame):
        self.frame = frame

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, i):
        row = self.frame.iloc[i]
        image_path = os.path.join(IMAGE_DIR, row.image)
        feature = get_features(image_path)
        caption = torch.tensor(row.ids, dtype=torch.long)
        return feature, caption

train_loader = DataLoader(FlickrDataset(train_data), batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(FlickrDataset(test_data), batch_size=BATCH_SIZE)

# 8. BASIC RNN MODEL
class CaptionRNN(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, 128, padding_idx=PAD)
        self.image_to_hidden = nn.Linear(512, 256)
        self.rnn = nn.RNN(128, 256, batch_first=True)
        self.fc = nn.Linear(256, vocab_size)

    def forward(self, image_features, caption_input):
        hidden = torch.tanh(self.image_to_hidden(image_features)).unsqueeze(0)
        x = self.embedding(caption_input)
        output, _ = self.rnn(x, hidden)
        return self.fc(output)

model = CaptionRNN(len(words)).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss(ignore_index=PAD)

# 9. TRAINING EPOCHS
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for features, captions in train_loader:
        features, captions = features.to(DEVICE), captions.to(DEVICE)
        inputs = captions[:, :-1]      # <START> and words so far
        targets = captions[:, 1:]      # next word to predict

        predictions = model(features, inputs)
        loss = loss_fn(predictions.reshape(-1, len(words)), targets.reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch + 1}/{EPOCHS}, Loss: {total_loss / len(train_loader):.4f}")

# 10. TEST LOSS
model.eval()
test_loss = 0
with torch.no_grad():
    for features, captions in test_loader:
        features, captions = features.to(DEVICE), captions.to(DEVICE)
        predictions = model(features, captions[:, :-1])
        loss = loss_fn(predictions.reshape(-1, len(words)), captions[:, 1:].reshape(-1))
        test_loss += loss.item()

if len(test_loader):
    print(f"Test Loss: {test_loss / len(test_loader):.4f}")
else:
    print("Test set is empty. Increase MAX_IMAGES to at least 2.")

# 11. GENERATE A CAPTION FOR A TEST IMAGE
@torch.no_grad()
def caption_image(path):
    model.eval()
    feature = get_features(path).unsqueeze(0).to(DEVICE)
    generated = [START]

    for _ in range(MAX_LEN):
        caption_input = torch.tensor([generated], device=DEVICE)
        scores = model(feature, caption_input)
        next_word = int(scores[0, -1].argmax())

        if next_word in (END, PAD):
            break
        generated.append(next_word)

    return " ".join(id_to_word[i] for i in generated[1:])

if len(test_data):
    example = test_data.iloc[0]
    example_path = os.path.join(IMAGE_DIR, example.image)
    print("\nTest image:", example.image)
    print("Generated caption:", caption_image(example_path))
    print("Actual caption:", example.caption)
else:
    print("No test image available. Increase MAX_IMAGES.")