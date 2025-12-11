import os
import re
import string
import numpy as np
import pandas as pd
import gensim.downloader as api
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from tabulate import tabulate

DATASET_PATH = "AG_News_Subset.csv"
BEST_MODEL_PATH = "models/w2v_Skip-gram_100_10.model"
GLOVE_MODEL_NAME = "glove-wiki-gigaword-100"

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '<NUM>', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = text.split()
    return tokens

def get_document_vector(tokens, model):
    vectors = [model[word] for word in tokens if word in model]
    if not vectors:
        return np.zeros(model.vector_size)
    return np.mean(vectors, axis=0)

df = pd.read_csv(DATASET_PATH)
df['text'] = df['Title'] + ' ' + df['Description']
df['tokens'] = df['text'].apply(preprocess_text)
df['label_id'] = df['Class Index'] - 1
label_names = ['World', 'Sports', 'Business', 'Sci/Tech']

X_train, X_test, y_train, y_test = train_test_split(
    df['tokens'], df['label_id'], test_size=0.2, random_state=42, stratify=df['label_id']
)

results = []

# TF-IDF
X_train_text = [' '.join(tokens) for tokens in X_train]
X_test_text = [' '.join(tokens) for tokens in X_test]
vectorizer = TfidfVectorizer(max_features=10000)
X_train_tfidf = vectorizer.fit_transform(X_train_text)
X_test_tfidf = vectorizer.transform(X_test_text)
clf_tfidf = LogisticRegression(max_iter=1000)
clf_tfidf.fit(X_train_tfidf, y_train)
y_pred_tfidf = clf_tfidf.predict(X_test_tfidf)
report_tfidf = classification_report(y_test, y_pred_tfidf, output_dict=True)
results.append({
    "Method": "TF-IDF (Baseline)", "Accuracy": report_tfidf['accuracy'],
    "Precision": report_tfidf['macro avg']['precision'], "Recall": report_tfidf['macro avg']['recall'],
    "Macro-F1": report_tfidf['macro avg']['f1-score'],
})

# Word2Vec
w2v_model = Word2Vec.load(BEST_MODEL_PATH).wv
X_train_w2v = np.array([get_document_vector(tokens, w2v_model) for tokens in X_train])
X_test_w2v = np.array([get_document_vector(tokens, w2v_model) for tokens in X_test])
clf_w2v = LogisticRegression(max_iter=1000)
clf_w2v.fit(X_train_w2v, y_train)
y_pred_w2v = clf_w2v.predict(X_test_w2v)
report_w2v = classification_report(y_test, y_pred_w2v, output_dict=True)
results.append({
    "Method": "Your Word2Vec", "Accuracy": report_w2v['accuracy'],
    "Precision": report_w2v['macro avg']['precision'], "Recall": report_w2v['macro avg']['recall'],
    "Macro-F1": report_w2v['macro avg']['f1-score'],
})

# GloVe
print("Loading GloVe model")
glove_model = api.load(GLOVE_MODEL_NAME)
X_train_glove = np.array([get_document_vector(tokens, glove_model) for tokens in X_train])
X_test_glove = np.array([get_document_vector(tokens, glove_model) for tokens in X_test])
clf_glove = LogisticRegression(max_iter=1000)
clf_glove.fit(X_train_glove, y_train)
y_pred_glove = clf_glove.predict(X_test_glove)
report_glove = classification_report(y_test, y_pred_glove, output_dict=True)
results.append({
    "Method": "Pre-trained GloVe", "Accuracy": report_glove['accuracy'],
    "Precision": report_glove['macro avg']['precision'], "Recall": report_glove['macro avg']['recall'],
    "Macro-F1": report_glove['macro avg']['f1-score'],
})

df_results = pd.DataFrame(results)
for col in ["Accuracy", "Precision", "Recall", "Macro-F1"]:
    df_results[col] = df_results[col].apply(lambda x: f"{x:.4f}")

print("\n" + "=" * 72)
print(" " * 17 + "Classification Model Comparison Results")
print("=" * 72)
print(tabulate(df_results, headers='keys', tablefmt='grid', showindex=False))

cm = confusion_matrix(y_test, y_pred_tfidf)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_names, yticklabels=label_names)
plt.title('Confusion Matrix - TF-IDF Model')
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.show()