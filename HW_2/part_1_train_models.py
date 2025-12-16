from datasets import load_dataset
import re
import string
import time
from gensim.models import Word2Vec


dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")

def preprocess_hf_data(hf_dataset):
    cleaned_sentences = []

    for item in hf_dataset:
        text = item['text']

        if not text.strip():
            continue

        new_sentence = []
        words = text.split()

        for word in words:
            word = word.lower()
            if re.search(r'\d', word):
                new_sentence.append("<NUM>")
                continue

            word = word.translate(str.maketrans('', '', string.punctuation))
            if len(word) > 0:
                new_sentence.append(word)

        if len(new_sentence) > 1:
            cleaned_sentences.append(new_sentence)

    return cleaned_sentences

cleaned_data = preprocess_hf_data(dataset)

architectures = ["CBOW", "Skip-gram"]
dimensions = [50, 100, 300]
window_sizes = [5, 10]

EPOCHS = 5
NEGATIVE = 5
MIN_COUNT = 5
WORKERS = 4

print(f"Starting training for 12 models...")
total_start = time.time()

for arch_name in architectures:
    sg_type = 1 if arch_name == "Skip-gram" else 0

    for dim in dimensions:
        for win in window_sizes:
            print(f"Training -> Arch: {arch_name} | Dim: {dim} | Win: {win} ...", end=" ")
            loop_start = time.time()

            model = Word2Vec(
                sentences=cleaned_data,
                vector_size=dim,
                window=win,
                sg=sg_type,
                negative=NEGATIVE,
                epochs=EPOCHS,
                min_count=MIN_COUNT,
                workers=WORKERS
            )

            filename = f"models/w2v_{arch_name}_{dim}_{win}.model"
            model.save(filename)

            duration = time.time() - loop_start
            print(f"[Done in {duration:.1f}s]")
total_end = time.time()

print(f"\nAll 12 models trained in {int(total_end - total_start)} seconds and were saved in 'models/' folder!")
