# test/lab5_test.py
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sklearn.model_selection import train_test_split
from src.preprocessing.regex_tokenizer import RegexTokenizer
from src.representations.count_vectorizer import CountVectorizer
from src.models.text_classifier import TextClassifier

# 1) Dataset
texts = [
    "This movie is fantastic and I love it!",
    "I hate this film, it's terrible.",
    "The acting was superb, a truly great experience.",
    "What a waste of time, absolutely boring.",
    "Highly recommend this, a masterpiece.",
    "Could not finish watching, so bad."
]
labels = [1, 0, 1, 0, 1, 0]  # 1 = positive, 0 = negative

# 2) Split 80/20
X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.2, random_state=42, stratify=labels
)

# 3) Tokenizer + CountVectorizer (Lab 2)
tokenizer = RegexTokenizer()
vectorizer = CountVectorizer(tokenizer)

# 4) Classifier
clf = TextClassifier(vectorizer)

# 5) Train
clf.fit(X_train, y_train)

# 6) Predict
y_pred = clf.predict(X_test)

# 7) Evaluate
metrics = clf.evaluate(y_test, y_pred)

# 8) Print results
print("Test texts:", X_test)
print("True labels :", y_test)
print("Pred labels :", y_pred)
print("\nMetrics:")
for k, v in metrics.items():
    print(f"- {k}: {v:.4f}")
