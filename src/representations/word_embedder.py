import gensim.downloader as api
import numpy as np
import sys
import os

# Giả sử bạn đã có Tokenizer từ Lab 1, nếu chưa có hãy dùng tạm hàm split()
# hoặc import từ project của bạn
try:
    from src.preprocessing.tokenizer import Tokenizer
except ImportError:
    class Tokenizer:
        def tokenize(self, text):
            return text.lower().split()

class WordEmbedder:
    def __init__(self, model_name: str = 'glove-wiki-gigaword-50'):
        print(f"Loading model '{model_name}'... This may take a while.")
        self.model = api.load(model_name)
        self.vector_size = self.model.vector_size
        self.tokenizer = Tokenizer()

    def get_vector(self, word: str):
        """Trả về vector embedding của một từ. Xử lý OOV."""
        try:
            return self.model[word.lower()]
        except KeyError:
            return None

    def get_similarity(self, word1: str, word2: str):
        """Tính cosine similarity giữa 2 từ."""
        try:
            return self.model.similarity(word1.lower(), word2.lower())
        except KeyError:
            return 0.0

    def get_most_similar(self, word: str, top_n: int = 10):
        """Tìm các từ tương đồng nhất."""
        try:
            return self.model.most_similar(word.lower(), topn=top_n)
        except KeyError:
            return []

    def embed_document(self, document: str):
        """Tạo document vector bằng cách trung bình cộng word vectors."""
        tokens = self.tokenizer.tokenize(document)
        vectors = []
        
        for token in tokens:
            v = self.get_vector(token)
            if v is not None:
                vectors.append(v)
        
        if not vectors:
            return np.zeros(self.vector_size)
        
        return np.mean(vectors, axis=0)