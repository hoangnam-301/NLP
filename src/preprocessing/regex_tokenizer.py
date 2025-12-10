import re
from typing import List
from src.core.interfaces import Tokenizer
class RegexTokenizer(Tokenizer):
    """
    A more advanced tokenizer that uses a regular expression to extract 
    tokens (words or single non-word, non-whitespace characters).
    """
    # Regex: \w+ (words/numbers/underscore) OR [^\w\s] (any single punctuation/symbol)
    TOKEN_PATTERN = r'\w+|[^\w\s]'

    def tokenize(self, text: str) -> List[str]:
        # 1. Convert the text to lowercase
        text = text.lower()
        
        # 2. Use re.findall to extract all matching tokens based on the pattern
        tokens = re.findall(self.TOKEN_PATTERN, text)
        
        return tokens