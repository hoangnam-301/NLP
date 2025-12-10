# src/core/interfaces.py (Updated)
import abc
from typing import List, Dict, Any, Union

# --- Tokenizer Interface (From Lab 1) ---
class Tokenizer(abc.ABC):
    @abc.abstractmethod
    def tokenize(self, text: str) -> List[str]:
        pass

# --- Vectorizer Interface (For Lab 2) ---
class Vectorizer(abc.ABC):
    """
    Abstract Base Class for a Vectorizer.
    Defines the standard interface for transforming text into numerical vectors.
    """
    
    @abc.abstractmethod
    def fit(self, corpus: List[str]) -> None:
        """
        Learns the vocabulary from a list of documents (corpus).
        """
        pass

    @abc.abstractmethod
    def transform(self, documents: List[str]) -> List[List[int]]:
        """
        Transforms a list of documents into a list of count vectors 
        based on the learned vocabulary.
        """
        pass

    def fit_transform(self, corpus: List[str]) -> List[List[int]]:
        """
        A convenience method that performs fit and then transform on the same data.
        """
        self.fit(corpus)
        return self.transform(corpus)