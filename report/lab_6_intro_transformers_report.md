# BÁO CÁO THỰC HÀNH LAB 6: GIỚI THIỆU VỀ TRANSFORMERS

---

## 1. Giải thích các bước thực hiện

Quy trình thực hiện bài lab được chia thành các giai đoạn chính như sau:

### 1.1. Cài đặt & Khởi tạo môi trường

- Cài đặt các thư viện cần thiết gồm **`transformers`** và **`torch`**.
- Sử dụng hàm **`pipeline()`** của Hugging Face để tự động hóa toàn bộ quy trình:
  - Tokenization (tiền xử lý văn bản)
  - Đưa dữ liệu vào mô hình
  - Sinh ra kết quả dự đoán

Cách tiếp cận này giúp người học nhanh chóng làm quen với Transformer mà không cần cài đặt thủ công từng bước phức tạp.

---

### 1.2. Tác vụ Fill-mask (BERT – Encoder-only)

- Sử dụng mô hình **`bert-base-uncased`**, thuộc kiến trúc **Encoder-only Transformer**.
- Nhiệm vụ của mô hình là dự đoán từ bị che bởi token `[MASK]` dựa trên ngữ cảnh hai chiều (trước và sau).
- BERT tận dụng cơ chế **Self-Attention hai chiều**, cho phép hiểu sâu sắc mối quan hệ ngữ nghĩa giữa các từ trong câu.

---

### 1.3. Tác vụ Text Generation (GPT-2 – Decoder-only)

- Sử dụng mô hình **GPT-2**, thuộc kiến trúc **Decoder-only Transformer**.
- Mô hình sinh văn bản theo cơ chế **tự hồi quy (auto-regressive)**, nghĩa là mỗi từ mới chỉ phụ thuộc vào các từ đứng trước.
- GPT-2 thể hiện khả năng sinh văn bản mạch lạc, tự nhiên và có phong cách giống ngôn ngữ con người.

---

### 1.4. Trích xuất Vector Embedding câu

- Sử dụng **`AutoModel`** để truy cập trực tiếp vào đầu ra của mô hình BERT.
- Lấy tensor **`last_hidden_state`** (kích thước `[batch_size, sequence_length, hidden_size]`).
- Áp dụng kỹ thuật **Mean Pooling** (có sử dụng `attention_mask`) để tổng hợp embedding của các token thành **một vector đại diện cho toàn bộ câu**.

---

## 2. Hướng dẫn chạy code

### Bước 1: Khởi tạo Notebook
- Mở **Google Colab** và tạo một Notebook mới.

### Bước 2: Cài đặt thư viện
Chạy lệnh sau tại cell đầu tiên:
```bash
pip install transformers torch
```

### Bước 3: Thực thi mã nguồn
- Copy lần lượt các đoạn code tương ứng với:
  - **Bài 1:** Fill-mask với BERT
  - **Bài 2:** Text Generation với GPT-2
  - **Bài 3:** Trích xuất Embedding câu
- Dán mỗi phần vào một cell riêng biệt và nhấn **Play** (hoặc `Ctrl + Enter`) để thực thi.

**Lưu ý:**  
- Nếu sử dụng mô hình RoBERTa mặc định, token che phải là `<mask>`.  
- Trong bài lab này sử dụng **`bert-base-uncased`**, do đó token `[MASK]` vẫn được giữ nguyên.

---

## 3. Phân tích kết quả thực nghiệm

### 3.1. Bài 1 – Fill-mask với BERT

- Từ được dự đoán là **"capital"** với độ tin cậy **0.9991 (99.91%)**.
- Kết quả này cho thấy khả năng hiểu ngữ cảnh rất chính xác của BERT.
- Điều này đến từ kiến trúc **Encoder-only** với cơ chế chú ý hai chiều (bidirectional attention).

---

### 3.2. Bài 2 – Text Generation với GPT-2

- Văn bản sinh ra có nội dung mạch lạc và tự nhiên, ví dụ:
  > "The best thing about learning NLP is that it feels like a real tool..."

- GPT-2 chỉ quan sát các token đứng trước, nhưng vẫn tạo ra văn bản có cấu trúc hợp lý và giàu ngữ nghĩa.

---

### 3.3. Bài 3 – Vector biểu diễn câu

- Kết quả thu được là một tensor có kích thước **[1, 768]**.
- Đây là vector biểu diễn nén thông tin ngữ nghĩa của toàn bộ câu từ không gian ẩn của BERT.

---

## 4. Trả lời câu hỏi thu hoạch

### Câu 1: Kích thước của vector biểu diễn là bao nhiêu? Nó tương ứng với tham số nào của BERT?

**Trả lời:**  
- Kích thước vector biểu diễn là **768**.
- Con số này tương ứng với tham số **`hidden_size`** (hay còn gọi là \(d_{model}\)) của mô hình **BERT-base**.
- Đây là kích thước của các tầng ẩn và cũng là chiều của vector đặc trưng mà mỗi token nhận được sau khi đi qua các lớp Encoder.

---

### Câu 2: Tại sao cần sử dụng `attention_mask` khi thực hiện Mean Pooling?

**Trả lời:**  
Chúng ta cần sử dụng `attention_mask` vì hai lý do chính:

1. **Loại bỏ token đệm (Padding tokens):**  
   Khi xử lý theo batch, các câu ngắn hơn sẽ được đệm thêm token `[PAD]`. Những token này không mang thông tin ngữ nghĩa và cần được loại bỏ khi tính embedding.

2. **Đảm bảo phép trung bình chính xác:**  
   Mean Pooling tính trung bình các vector token. Nếu không dùng mask, vector của `[PAD]` sẽ bị cộng vào và chia cho tổng độ dài chuỗi (bao gồm cả padding), khiến vector đại diện của câu bị sai lệch và "pha loãng" về mặt giá trị.

Việc sử dụng `attention_mask` đảm bảo rằng phép trung bình chỉ được tính trên các token thực sự có ý nghĩa trong câu.

---

