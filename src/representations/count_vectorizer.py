# src/representations/count_vectorizer.py
from src.core.interfaces import Vectorizer
from typing import List
import numpy as np  # chỉ dùng nếu bạn muốn tiện xử lý vector, không bắt buộc

class CountVectorizer(Vectorizer):
    def __init__(self, tokenizer):
        """
        tokenizer: một instance của Tokenizer (từ Lab 1)
        """
        self.tokenizer = tokenizer
        self.vocabulary_ = {}

    def fit(self, corpus: List[str]):
        """Học từ vựng (vocabulary) từ danh sách các văn bản."""
        unique_tokens = set()

        # duyệt qua từng văn bản trong corpus
        for doc in corpus:
            tokens = self.tokenizer.tokenize(doc)
            unique_tokens.update(tokens)

        # sắp xếp để có thứ tự cố định và gán index
        self.vocabulary_ = {token: idx for idx, token in enumerate(sorted(unique_tokens))}

    def transform(self, documents: List[str]) -> List[List[int]]:
        """Chuyển danh sách văn bản thành danh sách vector đếm."""
        if not self.vocabulary_:
            raise ValueError("Vocabulary is empty. You must call fit() before transform().")

        vectors = []

        for doc in documents:
            tokens = self.tokenizer.tokenize(doc)
            vector = [0] * len(self.vocabulary_)

            for token in tokens:
                if token in self.vocabulary_:
                    index = self.vocabulary_[token]
                    vector[index] += 1

            vectors.append(vector)

        return vectors

    def fit_transform(self, corpus: List[str]) -> List[List[int]]:
        """Thực hiện fit rồi transform trên cùng dữ liệu."""
        self.fit(corpus)
        return self.transform(corpus)
