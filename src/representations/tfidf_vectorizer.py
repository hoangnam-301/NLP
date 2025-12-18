import math
from typing import List
import numpy as np

class TfidfVectorizer:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.vocabulary = {}
        self.idf = {}

    def _calculate_tf(self, tokens: List[str]) -> dict:
        tf = {}
        for token in tokens:
            tf[token] = tf.get(token, 0) + 1
        # Normalization: tf = count / total_tokens
        total = len(tokens)
        for token in tf:
            tf[token] = tf[token] / total
        return tf

    def fit(self, texts: List[str]):
        # 1. Build Vocabulary
        all_tokens_docs = [self.tokenizer.tokenize(text) for text in texts]
        unique_tokens = sorted(list(set([token for doc in all_tokens_docs for token in doc])))
        self.vocabulary = {token: i for i, token in enumerate(unique_tokens)}

        # 2. Calculate IDF
        num_docs = len(texts)
        for token in self.vocabulary:
            # Count how many documents contain this token
            containing_docs = sum(1 for doc in all_tokens_docs if token in doc)
            # idf = log(total_docs / docs_with_token)
            self.idf[token] = math.log(num_docs / (1 + containing_docs))

    def transform(self, texts: List[str]) -> np.ndarray:
        X = np.zeros((len(texts), len(self.vocabulary)))
        for i, text in enumerate(texts):
            tokens = self.tokenizer.tokenize(text)
            tf = self._calculate_tf(tokens)
            for token, tf_val in tf.items():
                if token in self.vocabulary:
                    col_idx = self.vocabulary[token]
                    X[i, col_idx] = tf_val * self.idf[token]
        return X

    def fit_transform(self, texts: List[str]) -> np.ndarray:
        self.fit(texts)
        return self.transform(texts)