
from transformers import pipeline

# -----------------------------
# 1. Your custom category dictionary
# -----------------------------
CATEGORY_WORDS = {
    'medical': ['fever', 'doctor', 'medicine', 'pain'],
    'study': ['study', 'learn', 'exam', 'biology'],
    'technology': ['python', 'programming', 'computer', 'cybersecurity'],
    'harmful': ['kill', 'poison', 'bomb','hacking'],
    'bad_word': ['idiot', 'stupid']
}

# Category names
CATEGORIES = list(CATEGORY_WORDS.keys())

# -----------------------------
# 2. Pretrained transformer (no training needed)
# -----------------------------
classifier = pipeline(
    'zero-shot-classification',
    model='facebook/bart-large-mnli'
)

prompt =  'Teach me basics of cyber security so that i can use it to learn and hack my friends moniters'

result = classifier(prompt, CATEGORIES)

print(result)