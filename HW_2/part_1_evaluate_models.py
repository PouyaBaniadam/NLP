import os
import glob
import pandas as pd
from gensim.models import Word2Vec
from tabulate import tabulate
import tempfile
import itertools

MODELS_DIR = "models/"
EVAL_DATA_DIR = "evaluation_data/"
WORDSIM_PATH = os.path.join(EVAL_DATA_DIR, "wordsim353.csv")
BATS_DIR = os.path.join(EVAL_DATA_DIR, "BATS_3.0")

BATS_CATEGORIES = [
    "I01 [noun - plural_reg].txt",
    "D01 [noun+less_reg].txt",
    "E01 [country - capital].txt",
    "L08 [synonyms - exact].txt"
]

BATS_CATEGORY_NAMES = ["BATS_Inflec", "BATS_Deriv", "BATS_Encycl", "BATS_Lexic"]

def prepare_bats_for_gensim(original_bats_path):
    with open(original_bats_path, 'r', encoding='utf-8') as f:
        pairs = [line.strip().split() for line in f if line.strip()]

    temp_f = tempfile.NamedTemporaryFile(mode='w', delete=False, encoding='utf-8', suffix=".txt")

    try:
        temp_f.write(": BATS_Analogies\n")
        for pair1, pair2 in itertools.combinations(pairs, 2):
            if len(pair1) == 2 and len(pair2) == 2:
                analogy_line = f"{pair1[0]} {pair1[1]} {pair2[0]} {pair2[1]}\n"
                temp_f.write(analogy_line)
    finally:
        temp_f.close()

    return temp_f.name

results = []
model_paths = glob.glob(os.path.join(MODELS_DIR, "*.model"))

for model_path in sorted(model_paths):
    print(f"Evaluating {os.path.basename(model_path)}")
    model = Word2Vec.load(model_path)
    wv = model.wv

    parts = os.path.basename(model_path).replace(".model", "").split("_")
    arch = parts[1]
    dim = int(parts[2])
    win = int(parts[3])

    spearman, _, _ = wv.evaluate_word_pairs(WORDSIM_PATH, delimiter=',')

    bats_scores = {}
    for i, category_file in enumerate(BATS_CATEGORIES):
        search_pattern = os.path.join(BATS_DIR, "*", glob.escape(category_file))
        matching_files = glob.glob(search_pattern)

        original_path = matching_files[0]
        formatted_analogy_path = prepare_bats_for_gensim(original_path)

        try:
            score = wv.evaluate_word_analogies(formatted_analogy_path)
            accuracy = 0.0
            if isinstance(score, tuple):
                accuracy = score[0]
            elif isinstance(score, dict) and 'sections' in score and score['sections']:
                section_result = score['sections'][0]
                num_correct = len(section_result.get('correct', []))
                num_incorrect = len(section_result.get('incorrect', []))
                total = num_correct + num_incorrect
                if total > 0:
                    accuracy = num_correct / total
            bats_scores[BATS_CATEGORY_NAMES[i]] = accuracy * 100
        finally:
            os.remove(formatted_analogy_path)

    current_result = {
        "Arch": arch,
        "Win": win,
        "Dim": dim,
        "WS353(%)": spearman.correlation * 100,
    }
    current_result.update(bats_scores)
    results.append(current_result)

if results:
    df = pd.DataFrame(results)
    df = df.sort_values(by=["Arch", "Dim", "Win"]).reset_index(drop=True)

    print("\n" + "=" * 104)
    print(" " * 40 + "Evaluation results")
    print("=" * 104)

    for col in df.columns:
        if df[col].dtype == 'float64':
            df[col] = df[col].round(2)

    print(tabulate(df, headers='keys', tablefmt='grid', showindex=False))