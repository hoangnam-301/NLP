from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report

class IntentClassifier:
    def __init__(self, max_features=5000, max_iter=1000):
        """Khởi tạo pipeline TF-IDF + Logistic Regression"""
        self.pipeline = make_pipeline(
            TfidfVectorizer(max_features=max_features),
            LogisticRegression(max_iter=max_iter)
        )

    def train(self, X_train, y_train):
        """Huấn luyện mô hình"""
        self.pipeline.fit(X_train, y_train)

    def evaluate(self, X_test, y_test):
        """Đánh giá mô hình"""
        y_pred = self.pipeline.predict(X_test)
        print(classification_report(y_test, y_pred))
        return y_pred

    def predict(self, texts):
        """Dự đoán intent cho văn bản mới"""
        return self.pipeline.predict(texts)
