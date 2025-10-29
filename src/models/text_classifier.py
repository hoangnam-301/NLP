# src/models/text_classifier.py
from __future__ import annotations
from typing import List, Dict
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Vectorizer: dùng interface của bạn (Lab 2/3)
# expect: vectorizer.fit_transform(texts) -> list[list[float|int]]
#         vectorizer.transform(texts) -> list[list[float|int]]

class TextClassifier:
    def __init__(self, vectorizer):
        """
        vectorizer: instance tuân theo interface Vectorizer (Lab 2/3)
        """
        self.vectorizer = vectorizer
        self._model: LogisticRegression | None = None

    def fit(self, texts: List[str], labels: List[int]) -> None:
        """
        Huấn luyện LogisticRegression trên đặc trưng rút ra bởi vectorizer.
        """
        X_list = self.vectorizer.fit_transform(texts)   # list[list[...]]
        X = np.array(X_list, dtype=float)               # về numpy cho sklearn
        y = np.array(labels, dtype=int)

        # liblinear phù hợp dataset nhỏ; có thể đổi 'lbfgs' khi dữ liệu lớn hơn
        clf = LogisticRegression(solver="liblinear", max_iter=1000)
        clf.fit(X, y)
        self._model = clf

    def predict(self, texts: List[str]) -> List[int]:
        """
        Dự đoán nhãn cho văn bản mới.
        """
        if self._model is None:
            raise ValueError("Model chưa được huấn luyện. Hãy gọi fit() trước.")
        X_list = self.vectorizer.transform(texts)
        X = np.array(X_list, dtype=float)
        y_pred = self._model.predict(X)
        return y_pred.tolist()

    def evaluate(self, y_true: List[int], y_pred: List[int]) -> Dict[str, float]:
        """
        Tính các chỉ số: accuracy, precision, recall, f1 (macro cho nhị phân nhỏ).
        """
        acc = accuracy_score(y_true, y_pred)
        pre = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1  = f1_score(y_true, y_pred, zero_division=0)
        return {"accuracy": acc, "precision": pre, "recall": rec, "f1": f1}
