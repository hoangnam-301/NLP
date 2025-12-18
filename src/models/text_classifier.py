from typing import List, Dict
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class TextClassifier:
    def __init__(self, vectorizer):
        """
        Khởi tạo với một instance vectorizer (Tfidf hoặc CountVectorizer)
        """
        self.vectorizer = vectorizer
        # Sử dụng liblinear cho tập dữ liệu nhỏ
        self._model = LogisticRegression(solver='liblinear')

    def fit(self, texts: List[str], labels: List[int]):
        """
        Huấn luyện mô hình
        """
        # Bước 1: Chuyển văn bản thành ma trận đặc trưng
        X = self.vectorizer.fit_transform(texts)
        # Bước 2: Huấn luyện Logistic Regression
        self._model.fit(X, labels)

    def predict(self, texts: List[str]) -> List[int]:
        """
        Dự đoán nhãn cho văn bản mới
        """
        # Bước 1: Chuyển văn bản mới thành đặc trưng (chỉ transform, không fit lại)
        X = self.vectorizer.transform(texts)
        # Bước 2: Dự đoán
        return self._model.predict(X)

    def evaluate(self, y_true: List[int], y_pred: List[int]) -> Dict[str, float]:
        """
        Tính toán các chỉ số đánh giá
        """
        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred),
            "recall": recall_score(y_true, y_pred),
            "f1_score": f1_score(y_true, y_pred)
        }