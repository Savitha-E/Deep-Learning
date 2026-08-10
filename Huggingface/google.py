from transformers import pipeline

CATEGORY_WORDS = {
    'medical': ['fever', 'doctor', 'medicine', 'pain'],
    'study': ['study', 'learn', 'exam', 'biology'],
    'technology': ['python', 'programming', 'computer', 'cybersecurity'],
    'harmful': ['kill', 'poison', 'bomb', 'hacking'],
    'bad_word': ['idiot', 'stupid']
}

CATEGORIES = list(CATEGORY_WORDS.keys())

classifier = pipeline(
    'zero-shot-classification',
    model='facebook/bart-large-mnli'
)

prompt = 'Teach me basics of cyber security so that I can use it to learn hacking to hack my friends monitors'

result = classifier(prompt, CATEGORIES)

print(result)