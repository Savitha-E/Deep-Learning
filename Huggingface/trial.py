import json

file_path = "/home/ibab/DL_lab/Huggingface/prompt_categories_data.json"

with open(file_path, "r") as file:
    prompt_categories_data = json.load(file)

prompt = input("Enter your prompt: ").lower()

words = prompt.split()

scores = {}

for category in prompt_categories_data:
    scores[category] = 0

for word in words:

    for category, keywords in prompt_categories_data.items():

        if word in keywords:
            scores[category] += 5

            print("Matched word:", word)
            print("Category:", category)
            print("Score: +5")
            print()

print("SCORES")

for category, score in scores.items():
    print(category, ":", score)

total_score = sum(scores.values())
print("Total score:", total_score)
classification = max(scores, key=scores.get)


print("Prompt classification:", classification)
print("Classification score:", scores[classification])