import os
from huggingface_hub import InferenceClient


client = InferenceClient(
    provider="hf-inference",
    api_key=os.environ["HF_TOKEN"]
)


prompt = "What are the side effects?"


labels = [
    "valid",
    "ambiguous",
    "invalid"
]


result = client.zero_shot_classification(
    prompt,
    candidate_labels=labels,
    model="facebook/bart-large-mnli"
)


print(result)