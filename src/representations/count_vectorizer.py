# src/representations/count_vectorizer.py
from typing import List, Dict, Set, Union
from src.core.interfaces import Vectorizer, Tokenizer # Import interfaces


class CountVectorizer(Vectorizer):
    """
    Implements a CountVectorizer (Bag-of-Words model).
    Transforms documents into count vectors based on a learned vocabulary.
    """
    
    def __init__(self, tokenizer: Tokenizer):
        """
        Initializes the vectorizer with a tokenizer instance.
        
        Args:
            tokenizer: An instance of a class inheriting from Tokenizer.
        """
        self.tokenizer = tokenizer
        # Stores the word-to-index mapping: {'word': index}
        self.vocabulary_: Dict[str, int] = {}

    def fit(self, corpus: List[str]) -> None:
        """
        Learns the vocabulary from the corpus.
        
        Args:
            corpus: A list of documents (strings).
        """
        unique_tokens: Set[str] = set()
        
        # 1. Collect all unique tokens
        for document in corpus:
            tokens = self.tokenizer.tokenize(document)
            unique_tokens.update(tokens)
            
        # 2. Create the vocabulary_ dictionary (word-to-index mapping)
        # Sort the tokens for a deterministic index assignment
        sorted_tokens = sorted(list(unique_tokens))
        
        self.vocabulary_ = {
            token: index 
            for index, token in enumerate(sorted_tokens)
        }
        
        print(f"INFO: Vocabulary learned. Size: {len(self.vocabulary_)}")

    def transform(self, documents: List[str]) -> List[List[int]]:
        """
        Transforms documents into count vectors based on the learned vocabulary.
        
        Args:
            documents: A list of documents (strings) to transform.
            
        Returns:
            A list of count vectors (List[List[int]]).
        """
        vector_size = len(self.vocabulary_)
        document_term_matrix: List[List[int]] = []
        
        for document in documents:
            # Create a zero vector with length equal to vocabulary size
            vector = [0] * vector_size
            tokens = self.tokenizer.tokenize(document)
            
            # Count the frequency of each token
            for token in tokens:
                # Check if the token exists in the learned vocabulary
                if token in self.vocabulary_:
                    # Get the index of the token
                    index = self.vocabulary_[token]
                    # Increment the count at the corresponding index
                    vector[index] += 1
            
            document_term_matrix.append(vector)
            
        return document_term_matrix