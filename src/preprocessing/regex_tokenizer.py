# src/preprocessing/regex_tokenizer.py
import re
from src.core.interfaces import Tokenizer

class RegexTokenizer(Tokenizer):
    def __init__(self, pattern: str = r"\w+|[^\w\s]"):
        """
        pattern: biểu thức chính quy để tách token.
        Mặc định: \w+|[^\w\s]
        - \w+  : khớp với chuỗi ký tự chữ và số (từ).
        - [^\w\s] : khớp với các ký tự không phải chữ/số và không phải khoảng trắng (như dấu chấm, dấu phẩy,...)
        """
        self.pattern = pattern

    def tokenize(self, text: str) -> list[str]:
        """
        Tách văn bản thành danh sách token (từ hoặc dấu câu).
        """
        tokens = re.findall(self.pattern, text.lower())  # chuyển về chữ thường để đồng nhất
        return tokens
