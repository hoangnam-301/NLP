# test/lab1_test.py

import os
import sys
import io 
from typing import List

# =============================================================
# SỬA LỖI UNICODE: Bắt buộc đầu ra chuẩn (stdout) sử dụng mã hóa UTF-8
# =============================================================
# Áp dụng cho Windows để khắc phục lỗi khi in emoji hoặc ký tự Unicode
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


# 1. THIẾT LẬP ĐƯỜNG DẪN (Để tìm thấy thư mục src)
# Lấy đường dẫn của tệp hiện tại (test/lab1_test.py)
CURRENT_FILE_PATH = os.path.abspath(__file__)
TEST_DIR = os.path.dirname(CURRENT_FILE_PATH)
# Lùi lại 1 cấp từ 'test' để đến thư mục gốc 'nlp1'
ROOT_DIR = os.path.dirname(TEST_DIR) 

# Thêm thư mục GỐC vào sys.path. Điều này cho phép import module bắt đầu bằng 'src.'
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
    
# Debug:
print(f"DEBUG: Calculated ROOT_DIR: {ROOT_DIR}")

# 2. IMPORT CÁC MODULE
try:
    # SỬ DỤNG CÚ PHÁP TUYỆT ĐỐI BẮT ĐẦU TỪ SRC
    # Đảm bảo bạn đã có file __init__.py trong src/ và các thư mục con
    from src.preprocessing.simple_tokenizer import SimpleTokenizer
    from src.preprocessing.regex_tokenizer import RegexTokenizer
    from src.core.dataset_loaders import load_raw_text_data
except ImportError as e:
    print(f"Import Error: Could not find modules.")
    print(f"Chi tiết: {e}")
    print("Vui lòng kiểm tra: 1. Đã có file __init__.py trong src/ và các thư mục con chưa?")
    print("2. Các file implementation (tokenizer) đã import bằng cú pháp 'from src.core.interfaces' chưa?")
    sys.exit(1)


def run_tests():
    """Khởi tạo và kiểm tra các tokenizer."""
    
    print("\n## 🚀 Lab 1: Text Tokenization Test")
    print("---")
    
    simple_tokenizer = SimpleTokenizer()
    regex_tokenizer = RegexTokenizer()

    # Thêm test case để so sánh rõ hơn sự khác biệt giữa hai tokenizer
    test_sentences = [
        "Hello, world! This is a test.",
        "NLP is fascinating... isn't it?",
        "Let's see how it handles 123 numbers and punctuation!",
        "This costs $1,234.50, and it's complex."
    ]
    
    for i, sentence in enumerate(test_sentences):
        print(f"\n### Test Case {i+1}: \"{sentence}\"")
        
        # Test SimpleTokenizer
        simple_tokens = simple_tokenizer.tokenize(sentence)
        print(f"**SimpleTokenizer Output:** {simple_tokens}")
        
        # Test RegexTokenizer
        regex_tokens = regex_tokenizer.tokenize(sentence)
        print(f"**RegexTokenizer Output:** {regex_tokens}")

    # --- Task 3: Tokenization with UD_English-EWT Dataset ---
    print("\n\n## 📊 Task 3: Tokenization with UD_English-EWT Dataset")
    print("---")
    
    dataset_path = "/Data/HaritoWork/Teaching/VNU_HUS/Tu_NLP/data/UD_English-EWT/en_ewt-ud-train.txt"
    raw_text = load_raw_text_data(dataset_path)
    
    # Chỉ lấy 300 ký tự đầu tiên
    sample_text = raw_text[:300] 
    
    print("\n--- Tokenizing Sample Text from UD_English-EWT ---")
    print(f"Original Sample (first 100 chars): **{sample_text[:100]}...**")
    
    simple_tokens = simple_tokenizer.tokenize(sample_text)
    print(f"\n**SimpleTokenizer Output (first 25 tokens):** {simple_tokens[:25]}")
    
    regex_tokens = regex_tokenizer.tokenize(sample_text)
    print(f"**RegexTokenizer Output (first 25 tokens):** {regex_tokens[:25]}")
    
    print("\n--- Observation ---")
    print(
        "**Comparison:** RegexTokenizer (using \\w+|[^\\w\\s]) is generally more robust. "
        "SimpleTokenizer often struggles with contractions and leaves compound tokens (like $1,234.50) intact."
    )


if __name__ == "__main__":
    run_tests()