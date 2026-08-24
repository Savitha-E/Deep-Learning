import string
# Define special tokens and characters
special_tokens = ["START", "STOP"]
all_characters = string.ascii_letters + string.punctuation + " "  # Including letters and punctuation
vocab = {token: idx for idx, token in enumerate(special_tokens)}  # Start with special tokens

# Add the characters to the vocabulary starting after special tokens
vocab.update({char: idx + len(special_tokens) for idx, char in enumerate(all_characters)})

print(vocab)
