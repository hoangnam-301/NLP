# test/lab2_test.py

import os
import sys
import io

# 1. FIX LỖI UNICODE: Buộc đầu ra chuẩn (stdout) sử dụng mã hóa UTF-8
if sys.platform == "win32":
    # Áp dụng cho Windows để hiển thị emoji và ký tự Unicode
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# 2. SETUP ĐƯỜNG DẪN (Để tìm thấy thư mục src)
# Lấy đường dẫn của tệp hiện tại (test/lab2_test.py)
CURRENT_FILE_PATH = os.path.abspath(__file__)
TEST_DIR = os.path.dirname(CURRENT_FILE_PATH)
# Lùi lại 1 cấp từ 'test' để đến thư mục gốc 'nlp1'
ROOT_DIR = os.path.dirname(TEST_DIR) 

# Thêm thư mục GỐC vào sys.path. Điều này cho phép import module bắt đầu bằng 'src.'
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
    
# Debug:
print(f"DEBUG: Calculated ROOT_DIR: {ROOT_DIR}")

# 3. IMPORT CÁC MODULE (Bằng cú pháp tuyệt đối từ gốc)
try:
    # Nếu thư mục gốc đã được thêm, import phải bắt đầu bằng 'src.'
    from src.representations.count_vectorizer import CountVectorizer
    from src.preprocessing.regex_tokenizer import RegexTokenizer
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Vui lòng kiểm tra: 1. Đã có file __init__.py trong tất cả các thư mục con của src chưa?")
    print("2. Các file bên trong src đã import lẫn nhau bằng cú pháp from src.core.interfaces chưa?")
    sys.exit(1)


if __name__ == '__main__':
    print("\n## 🧪 Lab 2: Count Vectorization Evaluation")
    print("---")
    
    # 1. Define the sample corpus
    corpus = [
        "I love NLP.",
        "I love programming.",
        "NLP is a subfield of AI."
    ]
    
    # 2. Instantiate the tokenizer (from Lab 1)
    tokenizer = RegexTokenizer()
    print("INFO: RegexTokenizer instantiated.")
    
    # 3. Instantiate the CountVectorizer
    vectorizer = CountVectorizer(tokenizer=tokenizer)
    print("INFO: CountVectorizer instantiated.")
    
    print("\n--- Running fit_transform ---")
    
    # 4. Use fit_transform
    dtm = vectorizer.fit_transform(corpus)
    
    # 5. Print the learned vocabulary
    print("\n### Learned Vocabulary (Token: Index)")
    sorted_vocab = sorted(vectorizer.vocabulary_.items(), key=lambda item: item[1])
    vocab_map = {token: index for token, index in sorted_vocab}
    print(vocab_map)
    
    # 6. Print the resulting Document-Term Matrix (DTM)
    print("\n### Resulting Document-Term Matrix (DTM)")
    # Print the column headers (tokens)
    print(f"Columns (Tokens): {list(vocab_map.keys())}")
    
    for i, vector in enumerate(dtm):
        print(f"Document {i+1}: {vector} (\"{corpus[i]}\")")

    print("\n--- Evaluation Complete ---")