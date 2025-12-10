import re
from typing import List
from src.core.interfaces import Tokenizer
class SimpleTokenizer(Tokenizer):
    """
    A basic tokenizer that converts to lowercase, splits on whitespace, 
    và chỉ xử lý các dấu câu cơ bản.
    """
    def tokenize(self, text: str) -> List[str]:
        # 1. Convert the text to lowercase
        text = text.lower()
        
        # 2. Add spaces before and after common punctuation (KHÔNG xử lý dấu ')
        text = text.replace('.', ' . ')
        text = text.replace(',', ' , ')
        text = text.replace('?', ' ? ')
        text = text.replace('!', ' ! ')
        
        # Xử lý dấu chấm lửng
        text = text.replace('...', ' ... ')
        
        # 3. Clean up and split
        text = re.sub(r'\s+', ' ', text).strip()
        tokens = text.split(' ')
        
        return [token for token in tokens if token]