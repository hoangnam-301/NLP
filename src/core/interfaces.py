# src/core/interfaces.py
from abc import ABC, abstractmethod

class Tokenizer(ABC):
    @abstractmethod
    def tokenize(self, text: str) -> list[str]:
        """Tách văn bản thành danh sách token."""
        pass


class Vectorizer(ABC):
    @abstractmethod
    def fit(self, corpus: list[str]):
        """Học vocabulary từ danh sách các văn bản."""
        pass

    @abstractmethod
    def transform(self, documents: list[str]) -> list[list[int]]:
        """Biến các văn bản thành vector đếm dựa trên vocabulary đã học."""
        pass

    @abstractmethod
    def fit_transform(self, corpus: list[str]) -> list[list[int]]:
        """Thực hiện fit rồi transform trên cùng dữ liệu."""
        pass
