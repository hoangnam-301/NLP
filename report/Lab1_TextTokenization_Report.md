# Báo Cáo Lab 1: Text Tokenization (Tách Từ)

link code: https://github.com/hoangnam-301/NLP/blob/main/src/core/interfaces.py
           https://github.com/hoangnam-301/NLP/blob/main/src/preprocessing/simple_tokenizer.py
           https://github.com/hoangnam-301/NLP/blob/main/src/preprocessing/regex_tokenizer.py
           https://github.com/hoangnam-301/NLP/blob/main/src/core/dataset_loaders.py
           https://github.com/hoangnam-301/NLP/blob/main/test/lab1_test.py
---

## I. Mục Tiêu (Objective)

Mục tiêu của Lab 1 là hiểu và triển khai bước tiền xử lý cơ bản trong NLP: **Tokenization (Tách từ)**.  
Lab được chia thành hai nhiệm vụ:

1. Triển khai một **SimpleTokenizer** dựa trên các quy tắc đơn giản.
2. Triển khai một **RegexTokenizer** mạnh mẽ hơn, sử dụng biểu thức chính quy.

---

## II. Triển Khai (Implementation Details) ⊠

### 1. Cấu Trúc Project

Lab này sử dụng cấu trúc thư mục sau, với các tệp implementation được đặt trong thư mục `src`:

| Tệp tin                               | Mô tả triển khai                                                                                                                                                                                                 |
| :------------------------------------ | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `src/core/interfaces.py`              | Định nghĩa lớp trừu tượng `Tokenizer(abc.ABC)` với phương thức duy nhất: `tokenize(text: str) -> list[str]`.                                                                                                    |
| `src/preprocessing/simple_tokenizer.py` | Triển khai `SimpleTokenizer`. Sử dụng phương pháp thay thế chuỗi (`replace`) để chuyển chữ thường, tách từ bằng khoảng trắng, và tách dấu câu cơ bản (`.`, `,`, `!`, `?`).                                     |
| `src/preprocessing/regex_tokenizer.py`  | Triển khai `RegexTokenizer`. Sử dụng thư viện `re` với biểu thức chính quy `\w+|[^\w\s]` để tách các chuỗi từ/số (`\w+`) và các ký tự không phải từ/không phải khoảng trắng (`[^\w\s]`), đảm bảo tách rời punctuation. |

### 2. Các Bước Triển Khai Chính

1. **Định nghĩa Interface:**  
   Tạo lớp `Tokenizer` trừu tượng trong `src/core/interfaces.py`.

2. **SimpleTokenizer:**
   - Chuyển đầu vào sang chữ thường (`text.lower()`).
   - Thay thế dấu câu bằng `dấu_câu + khoảng trắng` (ví dụ: `.` → ` . `).
   - Sử dụng `split()` để tách tokens theo khoảng trắng.

3. **RegexTokenizer:**
   - Sử dụng `re.findall(r'\w+|[^\w\s]', text.lower())` để trích xuất tokens trong một bước.
   - Bắt được cả chuỗi từ/số và các dấu câu / ký hiệu riêng lẻ.

---

## III. Cách Chạy Code và Ghi Log Kết Quả ⊠

### 1. Môi Trường Thực Thi (Execution Environment)

- **Thư mục chạy:** Thư mục gốc dự án  
  `c:\Users\fpt\Downloads\nlp1\`
- **Tệp kiểm tra:** `test/lab1_test.py`
- **Lệnh thực thi:**

```bash
python test/lab1_test.py
```

### 2. Log Kết Quả Thu Được

Dưới đây là toàn bộ log kết quả thực thi:

```text
[Running] python -u "c:\Users\fpt\Downloads\nlp1\test\lab1_test.py"
DEBUG: Calculated ROOT_DIR: c:\Users\fpt\Downloads\nlp1

## 🚀 Lab 1: Text Tokenization Test
---

### Test Case 1: "Hello, world! This is a test."
**SimpleTokenizer Output:** ['hello', ',', 'world', '!', 'this', 'is', 'a', 'test', '.']
**RegexTokenizer Output:** ['hello', ',', 'world', '!', 'this', 'is', 'a', 'test', '.']

### Test Case 2: "NLP is fascinating... isn't it?"
**SimpleTokenizer Output:** ['nlp', 'is', 'fascinating', '.', '.', '.', "isn't", 'it', '?']
**RegexTokenizer Output:** ['nlp', 'is', 'fascinating', '.', '.', '.', 'isn', "'", 't', 'it', '?']

### Test Case 3: "Let's see how it handles 123 numbers and punctuation!"
**SimpleTokenizer Output:** ["let's", 'see', 'how', 'it', 'handles', '123', 'numbers', 'and', 'punctuation', '!']
**RegexTokenizer Output:** ['let', "'", 's', 'see', 'how', 'it', 'handles', '123', 'numbers', 'and', 'punctuation', '!']

### Test Case 4: "This costs $1,234.50, and it's complex."
**SimpleTokenizer Output:** ['this', 'costs', '$1', ',', '234', '.', '50', ',', 'and', "it's", 'complex', '.']
**RegexTokenizer Output:** ['this', 'costs', '$', '1', ',', '234', '.', '50', ',', 'and', 'it', "'", 's', 'complex', '.']


## 📊 Task 3: Tokenization with UD_English-EWT Dataset
---
INFO: Simulating load from /Data/HaritoWork/Teaching/VNU_HUS/Tu_NLP/data/UD_English-EWT/en_ewt-ud-train.txt. Returning placeholder text.

--- Tokenizing Sample Text from UD_English-EWT ---
Original Sample (first 100 chars): **It's a beautiful, new day! What's the plan? I don't know yet. The price is $1,234.50. This is the fi...**

**SimpleTokenizer Output (first 25 tokens):** ["it's", 'a', 'beautiful', ',', 'new', 'day', '!', "what's", 'the', 'plan', '?', 'i', "don't", 'know', 'yet', '.', 'the', 'price', 'is', '$1', ',', '234', '.', '50', '.']
**RegexTokenizer Output (first 25 tokens):** ['it', "'", 's', 'a', 'beautiful', ',', 'new', 'day', '!', 'what', "'", 's', 'the', 'plan', '?', 'i', 'don', "'", 't', 'know', 'yet', '.', 'the', 'price', 'is']

--- Observation ---
**Comparison:** RegexTokenizer (using \w+|[^\w\s]) is generally more robust. SimpleTokenizer often struggles with contractions and leaves compound tokens (like $1,234.50) intact.

[Done] exited with code=0 in 0.099 seconds
```

---

## IV. Giải Thích Các Kết Quả Thu Được

Kết quả cho thấy sự khác biệt rõ rệt về khả năng tách token giữa hai phương pháp:

### 1. SimpleTokenizer (Quy tắc thủ công)

- **Ưu điểm:**
  - Đơn giản, dễ hiểu.
  - Xử lý tốt các dấu câu cơ bản như `,`, `!`, `?`, `.`.

- **Hạn chế:**
  - **Contractions (từ rút gọn):**  
    Giữ nguyên các từ rút gọn như `"isn't"`, `"let's"`, `"it's"`.  
    Không tách dấu nháy đơn (`'`) ra khỏi từ, nên không phân rã được các thành phần ngữ pháp nhỏ hơn (như `isn` + `'` + `t`).
  - **Compound tokens (tokens phức hợp):**  
    Không tách chính xác các cấu trúc như tiền tệ hoặc số có định dạng:  
    `"$1,234.50"` bị xử lý thành `['$1', ',', '234', '.', '50']`, giữ `$` dính vào số.

### 2. RegexTokenizer (Dựa trên `\w+|[^\w\s]`)

- **Ưu điểm (Tính mạnh mẽ):**
  - **Tách contractions:**  
    Biểu thức `[^\w\s]` bắt được dấu nháy đơn (`'`) và tách nó ra:  
    `"isn't"` → `['isn', "'", 't']`.  
    `"let's"` → `['let', "'", 's']`.
  - **Tách symbol / số:**  
    Tách dấu tiền tệ `$` khỏi số:  
    `"$1,234.50"` → `['$', '1', ',', '234', '.', '50']`.
  - **Tokens “nguyên tố”:**  
    Các tokens thu được mang tính “nguyên tố” hơn, phù hợp cho các tác vụ NLP nâng cao như POS tagging hoặc parsing.

- **Kết luận:**  
  RegexTokenizer tạo ra các tokens chất lượng cao hơn bằng cách phân tách các ký hiệu không phải từ/số một cách nhất quán, khắc phục hầu hết các hạn chế của SimpleTokenizer.

---

## V. Nguồn Tham Khảo (References)

- Kiến thức về **Biểu thức chính quy (Regex)**: cú pháp `\w+` và `[^\w\s]` để phân tách token.  
- Kiến thức về Python `sys.path` và **Absolute Imports** để thiết lập môi trường dự án.
