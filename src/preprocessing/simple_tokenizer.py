# src/preprocessing/simple_tokenizer.py
import re
from src.core.interfaces import Tokenizer

class SimpleTokenizer(Tokenizer):
    def tokenize(self, text: str) -> list[str]:
        # Đưa về chữ thường
        text = text.lower()

        # Tách dấu câu (.,!?...) ra khỏi từ bằng cách chèn khoảng trắng
        text = re.sub(r'([.,!?])', r' \1 ', text)

        # Xóa khoảng trắng thừa và tách từ theo dấu cách
        tokens = text.split()

        return tokens
