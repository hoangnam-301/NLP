# Báo cáo Bài Lab 7: Phân tích Cú pháp Phụ thuộc với spaCy

---

## I. Giới thiệu

Bài Lab này tập trung vào việc thực hành kỹ thuật **Phân tích Cú pháp Phụ thuộc (Dependency Parsing)**, một phương pháp nền tảng trong NLP giúp xác định cấu trúc ngữ pháp của câu dưới dạng mối quan hệ **head** và **dependent**.  

Công cụ chính được sử dụng là thư viện **spaCy** của Python.

---

## II. Các Bước Triển khai (Implementation Steps)

Quá trình triển khai được chia thành các bước logic tuần tự:

1. **Cài đặt Môi trường**  
   - Cài đặt thư viện `spaCy`.  
   - Tải mô hình tiếng Anh có chứa thông tin parsing: `en_core_web_md`.

2. **Phân tích và Trực quan hóa**  
   - Sử dụng hàm `nlp()` của spaCy để xử lý văn bản thành đối tượng `Doc`.  
   - Sử dụng `displacy.render(doc, style="dep", jupyter=True)` để hiển thị cây phụ thuộc.

3. **Truy cập Thuộc tính Token**  
   - Duyệt qua từng `Token` trong đối tượng `Doc` để trích xuất các thuộc tính cốt lõi của cây phụ thuộc:  
     - `token.text`  
     - `token.dep_`  
     - `token.head.text`  
     - `token.children`

4. **Trích xuất Thông tin có mục đích**  
   - Xây dựng các đoạn code logic để duyệt cây phụ thuộc nhằm giải quyết các bài toán cụ thể:  
     - Tìm bộ ba **S–V–O (Subject–Verb–Object)**.  
     - Tìm bổ ngữ tính từ (adjectival modifiers).

5. **Phát triển Hàm tùy chỉnh**  
   - Viết các hàm nâng cao để tự động hóa:  
     - Tìm token `ROOT`.  
     - Tự trích xuất cụm danh từ (noun phrases).  
     - Tìm đường đi từ một token bất kỳ đến `ROOT`.

---

## III. Hướng dẫn Thực thi Code (Code Execution Guide)

Toàn bộ bài Lab được thiết kế để chạy trong môi trường **Jupyter Notebook** hoặc **Google Colab** (`.ipynb`), nằm trong thư mục `./notebook/`.

1. **Cài đặt ban đầu**

   ```bash
   pip install -U spacy
   python -m spacy download en_core_web_md
   ```

2. **Trực quan hóa (Phần 2.2)**  
   - Để xem cây phụ thuộc trực tiếp trong Notebook, sử dụng:

   ```python
   from spacy import displacy
   displacy.render(doc, style="dep", jupyter=True)
   ```

3. **Thực thi các phần khác**  
   - Các đoạn code còn lại (Phần 3, 4, 5) được chia thành các cell trong Notebook.  
   - Chạy tuần tự để quan sát kết quả in ra màn hình (cấu trúc head–dependent, triplets, noun phrases, đường đi đến ROOT, ...).

---

## IV. Phân tích Kết quả (Result Analysis)

### 4.1. Phân tích Cây Phụ thuộc (Visual Analysis)

Câu ví dụ:

> **"The quick brown fox jumps over the lazy dog."**

Cây phụ thuộc được spaCy sinh ra cho thấy:

- **Gốc (ROOT):**  
  - Từ `jumps` (VERB) là **ROOT** của câu, là động từ chính điều khiển toàn bộ cấu trúc.

- **Vế Chủ ngữ:**
  - Danh từ `fox` (NOUN) là **chủ ngữ** (`nsubj`) của `jumps`.  
  - `fox` được bổ nghĩa bởi:
    - `The` (`det`)  
    - `quick` (`amod`)  
    - `brown` (`amod`)

- **Vế Tân ngữ/Bổ ngữ Giới từ:**
  - Giới từ `over` (ADP) là **bổ ngữ giới từ** (`prep`) của `jumps`.  
  - `over` chi phối danh từ `dog` (NOUN) với quan hệ `pobj`.  
  - `dog` được bổ nghĩa bởi:
    - `the` (`det`)  
    - `lazy` (`amod`)

### 4.2. Phân tích Thuộc tính Token (Phần 3)

Với câu:

> **"Apple is looking at buying U.K. startup for $1 billion"**

Phân tích cho thấy cấu trúc head–dependent rõ ràng:

- `looking` là **ROOT** (động từ chính).  
- `Apple` là **chủ ngữ** (`nsubj`) của `looking`.  
- `startup` là **tân ngữ trực tiếp** (`dobj`) của hành động `buying`.  
- `billion` là **tân ngữ giới từ** (`pobj`) của giới từ `for`.

Ví dụ bảng trích xuất một số token:

| TEXT    | DEP   | HEAD TEXT | HEAD POS | CHILDREN            |
|--------|-------|-----------|----------|----------------------|
| Apple  | nsubj | looking   | VERB     | `[]`                 |
| startup| dobj  | buying    | VERB     | `['U.K.', 'for']`    |
| billion| pobj  | for       | ADP      | `['$', '1']`         |

### 4.3. Trích xuất Bộ ba và Bổ ngữ (Phần 4)

| Bài toán                | Kỹ thuật sử dụng                                              | Kết quả ví dụ                                  | Ý nghĩa                                                                 |
|-------------------------|---------------------------------------------------------------|------------------------------------------------|-------------------------------------------------------------------------|
| Tìm Triplet (S–V–O)     | Tìm `nsubj` và `dobj` trong `children` của các token VERB.   | `(cat, chased, mouse)`, `(dog, watched, them)` | Trích xuất thành công các sự kiện/hành động chính trong câu ghép.      |
| Tìm Bổ ngữ Tính từ      | Tìm `amod` trong `children` của các NOUN.                    | `cat → ['big', 'fluffy', 'white']`, `mat → ['warm']` | Xác định các mô tả (tính từ bổ nghĩa) đi kèm danh từ.                 |

### 4.4. Phân tích Các Bài tập Nâng cao (Phần 5)

- **Bài 1 (Động từ Chính):**  
  - Xác nhận `jumps` là **ROOT** của câu ví dụ.

- **Bài 2 (Cụm Danh từ):**  
  - Phương pháp tự viết cho kết quả:  
    - `['The quick brown fox', 'the lazy dog']`  
  - Khớp với kết quả cụm danh từ tích hợp sẵn trong spaCy.

- **Bài 3 (Đường đi đến ROOT):**  
  - Truy vết thành công chuỗi phụ thuộc từ một token đến `ROOT`, ví dụ:

  $$
  \text{startup (dobj)} \rightarrow \text{buying (pcomp)} \rightarrow \text{at (prep)} \rightarrow \text{looking (ROOT)}
  $$

---

## V. Tài liệu Tham khảo (Cited References)

- **Thư viện spaCy**  
  - Công cụ chính được sử dụng để thực hiện Dependency Parsing.  
  - Website: https://spacy.io/

- **Hệ thống Nhãn Phụ thuộc (Universal Dependencies - UD)**  
  - Cơ sở cho các nhãn quan hệ như `nsubj`, `dobj`, `amod`, `ROOT`, v.v.  
  - Tham khảo: https://universaldependencies.org/u/dep/index.html
