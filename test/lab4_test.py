import sys
import os
from sklearn.model_selection import train_test_split

# Đảm bảo Python tìm thấy thư mục 'src'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# --- SỬA DÒNG NÀY ---
from src.preprocessing.regex_tokenizer import RegexTokenizer     # Đã đổi từ .tokenizer sang .regex_tokenizer
# --------------------

from src.representations.tfidf_vectorizer import TfidfVectorizer 
from src.models.text_classifier import TextClassifier           

def main():
    # Dữ liệu mẫu cho Text Classification
    texts = [
        "This movie is fantastic and I love it!",
        "I hate this film, it's terrible.",
        "The acting was superb, a truly great experience.",
        "What a waste of time, absolutely boring.",
        "Highly recommend this, a masterpiece.",
        "Could not finish watching, so bad.",
        "An amazing piece of art, loved every second.",
        "Disappointing and slow, would not recommend."
    ]
    labels = [1, 0, 1, 0, 1, 0, 1, 0] # 1: Positive, 0: Negative

    # Chia dữ liệu: 80% Train, 20% Test
    X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)

    # Khởi tạo Pipeline
    tokenizer = RegexTokenizer()
    vectorizer = TfidfVectorizer(tokenizer=tokenizer)
    classifier = TextClassifier(vectorizer=vectorizer)

    # Huấn luyện
    print("--- Training Text Classifier ---")
    classifier.fit(X_train, y_train)

    # Dự đoán và Đánh giá
    y_pred = classifier.predict(X_test)
    metrics = classifier.evaluate(y_test, y_pred)

    print("\nEvaluation Metrics:")
    for key, value in metrics.items():
        print(f" - {key}: {value:.4f}")

    # Thử nghiệm thực tế
    sample = ["I love the fantastic acting"]
    pred = classifier.predict(sample)
    print(f"\nResult: '{sample[0]}' -> {'Positive' if pred[0] == 1 else 'Negative'}")

if __name__ == "__main__":
    main()