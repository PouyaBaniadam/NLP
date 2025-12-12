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

def run_qualitative_analysis(results_df):
    print("\n" + "=" * 77)
    print("Qualitative analysis (Best vs Worst & Window Size Effect)")
    print("=" * 77)

    if results_df.empty:
        return

    best_row = results_df.loc[results_df['WS353(%)'].idxmax()]
    worst_row = results_df.loc[results_df['WS353(%)'].idxmin()]

    def get_model_path(row):
        return os.path.join(MODELS_DIR, f"w2v_{row['Arch']}_{int(row['Dim'])}_{int(row['Win'])}.model")

    best_model_path = get_model_path(best_row)
    worst_model_path = get_model_path(worst_row)

    print(f"Best Model:  {os.path.basename(best_model_path)} (Score: {best_row['WS353(%)']:.2f})")
    print(f"Worst Model: {os.path.basename(worst_model_path)} (Score: {worst_row['WS353(%)']:.2f})")

    best_wv = Word2Vec.load(best_model_path).wv
    worst_wv = Word2Vec.load(worst_model_path).wv

    target_words = ["bank", "apple", "run"]
    print("\n[Part A] Polysemous Words Analysis (Top 5 Neighbors)")

    for word in target_words:
        print(f"\nTarget Word: '{word}'")
        try:
            best_neighbors = [w[0] for w in best_wv.most_similar(word, topn=5)]
            worst_neighbors = [w[0] for w in worst_wv.most_similar(word, topn=5)]

            print(f"  Best Model:  {', '.join(best_neighbors)}")
            print(f"  Worst Model: {', '.join(worst_neighbors)}")
        except KeyError:
            print(f"  Word '{word}' not in vocabulary.")

    print("\n[Part B] Window Size Effect Analysis (Syntactic vs Semantic)")

    path_win5 = os.path.join(MODELS_DIR, "w2v_Skip-gram_100_5.model")
    path_win10 = os.path.join(MODELS_DIR, "w2v_Skip-gram_100_10.model")

    if os.path.exists(path_win5) and os.path.exists(path_win10):
        try:
            wv_win5 = Word2Vec.load(path_win5).wv
            wv_win10 = Word2Vec.load(path_win10).wv

            test_words = ["run", "driving", "play"]

            for word in test_words:
                print(f"\nAnalyzing word: '{word}'")
                try:
                    n_win5 = [w[0] for w in wv_win5.most_similar(word, topn=5)]
                    n_win10 = [w[0] for w in wv_win10.most_similar(word, topn=5)]

                    print(f"  Small Window (5):  {', '.join(n_win5)}")
                    print(f"  Large Window (10): {', '.join(n_win10)}")
                except KeyError:
                    print(f"  Word '{word}' not in vocabulary.")
        except Exception as e:
            print(f"Error loading window size models: {e}")
    else:
        print("Skipping Window Size analysis: Specific Skip-gram models (100d, 5/10) not found.")

results = []
model_paths = glob.glob(os.path.join(MODELS_DIR, "*.model"))

if not model_paths:
    print("No models found in 'models/' directory. Please run training script first.")
else:
    for model_path in sorted(model_paths):
        print(f"Evaluating {os.path.basename(model_path)}...")
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

            if not matching_files:
                bats_scores[BATS_CATEGORY_NAMES[i]] = 0.0
                continue

            original_path = matching_files[0]
            formatted_analogy_path = prepare_bats_for_gensim(original_path)

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
        print(" " * 40 + "PARAMETER SEARCH RESULTS")
        print("=" * 104)

        for col in df.columns:
            if df[col].dtype == 'float64':
                df[col] = df[col].round(2)

        print(tabulate(df, headers='keys', tablefmt='grid', showindex=False))

        run_qualitative_analysis(df)