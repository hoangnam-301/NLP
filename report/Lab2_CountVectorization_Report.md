# Báo Cáo Lab 2: Count Vectorization (Bag-of-Words)

link code: https://github.com/hoangnam-301/NLP/blob/main/src/core/interfaces.py
           https://github.com/hoangnam-301/NLP/blob/main/src/representations/count_vectorizer.py
           https://github.com/hoangnam-301/NLP/blob/main/test/lab2_test.py
---

## I. Mục Tiêu (Objective)

Mục tiêu của Lab 2 là triển khai mô hình **Count Vectorization** (mô hình Túi từ – **Bag-of-Words** hay **BoW**) để chuyển đổi các tài liệu văn bản thành các vector số học. Việc triển khai này sử dụng **RegexTokenizer** đã hoàn thành từ Lab 1.

---

## II. Triển Khai (Implementation Details) 

### 1. Cấu Trúc Project

Lab này sử dụng lại cấu trúc package và bổ sung thư mục `src/representations` cho `CountVectorizer`.

| Tệp tin                                | Mô tả triển khai                                                                                                                                                                                                 |
| :------------------------------------- | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `src/core/interfaces.py`              | Định nghĩa lớp trừu tượng `Vectorizer(abc.ABC)` với các phương thức: `fit()`, `transform()`, và `fit_transform()`.                                                                                               |
| `src/representations/count_vectorizer.py` | Triển khai lớp `CountVectorizer(Vectorizer)`. Constructor nhận `tokenizer: Tokenizer`. Phương thức `fit()` thu thập token độc nhất và tạo ánh xạ `self.vocabulary_`. Phương thức `transform()` tạo vector zero và tăng count tại chỉ mục tương ứng cho mỗi token tìm thấy. |

### 2. Các Bước Triển Khai Chính

1. **Giao diện:**  
   Thêm `Vectorizer` Abstract Base Class vào `src/core/interfaces.py`.

2. **Học từ vựng (`fit`):**  
   Duyệt qua `corpus`, sử dụng `tokenizer.tokenize()` để thu thập tất cả các token độc nhất (unique tokens).  
   Sắp xếp các token và gán chỉ mục để tạo `self.vocabulary_`.

3. **Chuyển đổi (`transform`):**  
   Đối với mỗi tài liệu, tạo một vector đếm có kích thước bằng kích thước từ vựng.  
   Duyệt qua các token; nếu token có trong từ vựng, tăng giá trị tại chỉ mục tương ứng lên 1.

---

## III. Cách Chạy Code Và Ghi Log Kết Quả ⊠

### 1. Môi Trường Thực Thi (Execution Environment)

- **Thư mục chạy:** Thư mục gốc dự án  
  `c:\Users\fpt\Downloads\nlp1\`
- **Tệp kiểm tra:** `test/lab2_test.py`
- **Lệnh thực thi:**

```bash
python test/lab2_test.py
```

### 2. Log Kết Quả Thu Được

Dưới đây là toàn bộ log kết quả thực thi:

```text
[Running] python -u "c:\Users\fpt\Downloads\nlp1\test\lab2_test.py"
DEBUG: Calculated ROOT_DIR: c:\Users\fpt\Downloads\nlp1

## Lab 2: Count Vectorization Evaluation
---
INFO: RegexTokenizer instantiated.
INFO: CountVectorizer instantiated.

--- Running fit_transform ---
INFO: Vocabulary learned. Size: 10

### Learned Vocabulary (Token: Index)
{'.': 0, 'a': 1, 'ai': 2, 'i': 3, 'is': 4, 'love': 5, 'nlp': 6, 'of': 7, 'programming': 8, 'subfield': 9}

### Resulting Document-Term Matrix (DTM)
Columns (Tokens): ['.', 'a', 'ai', 'i', 'is', 'love', 'nlp', 'of', 'programming', 'subfield']
Document 1: [1, 0, 0, 1, 0, 1, 1, 0, 0, 0] ("I love NLP.")
Document 2: [1, 0, 0, 1, 0, 1, 0, 0, 1, 0] ("I love programming.")
Document 3: [1, 1, 1, 0, 1, 0, 1, 1, 0, 1] ("NLP is a subfield of AI.")

--- Evaluation Complete ---

[Done] exited with code=0 in 0.09 seconds
```

---

## IV. Giải Thích Các Kết Quả Thu Được

### 1. Phân Tích Từ Vựng (Vocabulary Analysis)

- **Kích thước:** 10 tokens độc nhất đã được học.
- **Quy tắc BoW:** Mỗi token (kể cả dấu chấm `"."`) được gán một chiều (feature) trong không gian vector.
- **Tính nhất quán:** `RegexTokenizer` đã chuẩn hóa từ `I` thành `i`, đảm bảo tính đồng nhất về mặt từ vựng.

### 2. Phân Tích Ma Trận Tài Liệu – Từ (Document-Term Matrix, DTM)

Ma trận DTM có kích thước **3 × 10**. Mỗi vector thể hiện số lần đếm token trong từng tài liệu:

- **Document 1** (`"I love NLP."`):  
  Vector chứa giá trị `1` tại các cột tương ứng với các token: `i`, `love`, `nlp`, và `.`.

- **Document 2** (`"I love programming."`):  
  Vector có giá trị `1` tại các cột: `.`, `i`, `love`, `programming`.

- **Document 3** (`"NLP is a subfield of AI."`):  
  Vector có giá trị `1` tại các chỉ mục tương ứng với: `a`, `ai`, `is`, `nlp`, `of`, `subfield` và `.`;  
  có giá trị `0` tại các chỉ mục tương ứng với `i` và `love` vì các token này không xuất hiện trong câu.

**Kết luận:**  
Quá trình Count Vectorization đã thành công trong việc mã hóa tần suất từ, tạo ra các vector số học chính xác, hoàn thành mục tiêu của Lab 2.

