# 3-1: Data Preprocessing
import nltk
from nltk.corpus import movie_reviews, stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer

# 1. Downloads
try:
    nltk.data.find('corpora/movie_reviews')
except LookupError:
    nltk.download('movie_reviews')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

# 2. Preprocessing Function
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text: str) -> str:
    # A. Lowercasing
    text = text.lower()
    # B. Tokenization
    tokens = word_tokenize(text)
    # C. Removing Punctuation, Stopwords & Lemmatization
    clean_tokens = []
    for token in tokens:
        if token.isalpha() and token not in stop_words:
            lemma = lemmatizer.lemmatize(token)
            clean_tokens.append(lemma)
    return " ".join(clean_tokens)

# 3. Loading and Cleaning Data
print("Loading and Preprocessing data...")
documents = []
for category in movie_reviews.categories():
    for fileid in movie_reviews.fileids(category):
        raw_text = movie_reviews.raw(fileid)
        cleaned_text = preprocess_text(raw_text)
        documents.append((cleaned_text, category))

# Separating text and labels
X_text = [doc for doc, category in documents]
y_labels = [category for doc, category in documents]

# 4. Vectorization (Bag of Words)
# This converts text to a matrix of token counts
vectorizer = CountVectorizer()
X_vectorized = vectorizer.fit_transform(X_text)

print("-" * 30)
print("DELIVERABLE 3-1 COMPLETED")
print(f"Number of documents processed: {len(X_text)}")
print(f"Shape of Data Matrix (Rows, Features): {X_vectorized.shape}")
print(f"Sample Features (Words): {vectorizer.get_feature_names_out()[:10]}")
print("-" * 30)


# 3-2: Model Implementation
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# 1. Split the dataset (80/20 split)
print("Splitting data into Train (80%) and Test (20%)...")
X_train, X_test, y_train, y_test = train_test_split(
    X_vectorized, y_labels, test_size=0.2, random_state=42
)

# 2. Train Multinomial Naive Bayes model
print("Training Naive Bayes Model...")
model = MultinomialNB()
model.fit(X_train, y_train)

# 3. Evaluate the model's performance
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, pos_label='pos')
recall = recall_score(y_test, y_pred, pos_label='pos')
f1 = f1_score(y_test, y_pred, pos_label='pos')

print("\n" + "="*30)
print("DELIVERABLE 3-2 RESULTS")
print("="*30)
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-Score:  {f1:.4f}")
print("\nDetailed Classification Report:")
print(classification_report(y_test, y_pred))

# 4. Visualize Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Negative', 'Positive'],
            yticklabels=['Negative', 'Positive'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - Naive Bayes')
plt.show()