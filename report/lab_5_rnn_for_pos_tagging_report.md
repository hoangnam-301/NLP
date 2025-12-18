# BÁO CÁO KẾT QUẢ LAB 5: XÂY DỰNG MÔ HÌNH RNN CHO BÀI TOÁN POS TAGGING

link code: https://github.com/hoangnam-301/NLP/blob/main/notebook/lab5_rnn_for_pos_tagging.ipynb

---

## 1. Giải thích các bước thực hiện

### Bước 1: Tiền xử lý dữ liệu

**Kết nối & Giải nén dữ liệu**  
Sử dụng Google Colab để mount Google Drive và giải nén tệp dữ liệu `UD_English-EWT.tar.gz`. Việc giải nén trực tiếp vào môi trường chạy giúp giảm thời gian truy xuất và thuận tiện trong quá trình huấn luyện.

**Đọc dữ liệu CoNLL-U**  
Xây dựng hàm `load_conllu` để đọc dữ liệu theo định dạng CoNLL-U. Trong bài lab này, chương trình chỉ sử dụng hai trường thông tin quan trọng:
- `FORM`: từ gốc trong câu
- `UPOS`: nhãn Part-of-Speech theo chuẩn Universal POS

Mỗi câu được biểu diễn dưới dạng một danh sách từ và danh sách nhãn tương ứng.

**Xây dựng Vocabulary**  
Thiết lập hai từ điển:
- `word_to_ix`: ánh xạ từ → chỉ số
- `tag_to_ix`: ánh xạ nhãn POS → chỉ số

Bổ sung các token đặc biệt:
- `<PAD>` (index = 0): dùng để padding các câu trong batch
- `<UNK>` (index = 1): dùng cho các từ chưa xuất hiện trong tập huấn luyện

---

### Bước 2: Thiết kế Dataset và DataLoader

**Lớp POSDataset**  
Xây dựng lớp `POSDataset` để chuyển đổi dữ liệu văn bản (từ và nhãn) thành các tensor chỉ số phù hợp với PyTorch.

**Kỹ thuật Padding**  
Sử dụng `collate_fn` kết hợp với hàm `pad_sequence` để đệm các câu trong cùng một batch về cùng độ dài. Đây là bước bắt buộc để mô hình RNN có thể xử lý dữ liệu theo lô (batch) một cách hiệu quả.

---

### Bước 3: Xây dựng mô hình RNN

Mô hình **SimpleRNNForTokenClassification** bao gồm các thành phần chính:

- **Embedding layer (`nn.Embedding`)**  
  Chuyển đổi chỉ số từ thành vector đặc trưng với kích thước embedding = 128.

- **Recurrent layer (`nn.RNN`)**  
  Xử lý chuỗi tuần tự của các từ trong câu, với hidden dimension = 256.

- **Fully Connected layer (`nn.Linear`)**  
  Ánh xạ đầu ra của RNN tại mỗi thời điểm sang không gian nhãn POS (17 nhãn UPOS).

---

### Bước 4: Huấn luyện mô hình

- **Hàm mất mát:** `CrossEntropyLoss(ignore_index=0)` để bỏ qua các vị trí padding khi tính loss.
- **Optimizer:** Adam với learning rate = $10^{-3}$.
- **Số epoch:** 5 epoch.

Trong mỗi epoch, mô hình được huấn luyện trên tập Train và đánh giá trên tập Dev.

---

## 2. Hướng dẫn chạy code (Reproduce)

### Chuẩn bị
- Đảm bảo file dữ liệu nén nằm tại đường dẫn:
```
/content/drive/MyDrive/UD_English-EWT.tar.gz
```

### Thực thi tuần tự
1. Chạy cell **Mount Drive** để cấp quyền truy cập Google Drive.
2. Chạy cell **Giải nén & Load dữ liệu** để chuẩn bị tập Train/Dev.
3. Chạy cell **Định nghĩa POSDataset và mô hình RNN**.
4. Chạy cell **Train loop** để huấn luyện mô hình trong 5 epoch.
5. Quan sát **Loss** và **Accuracy** được in ra sau mỗi epoch.
6. Cuối cùng, chạy hàm `predict()` để kiểm tra mô hình với các câu mới.

---

## 3. Phân tích kết quả thực nghiệm

### Kết quả huấn luyện

| Epoch | Loss (Train) | Accuracy trên tập Dev |
|------|---------------|----------------------|
| 1 | 1.0265 | 75.72% |
| 2 | 0.5951 | 80.17% |
| 3 | 0.4463 | 83.24% |
| 4 | 0.3484 | 84.88% |
| 5 | 0.2772 | 85.62% |

---

### Ví dụ dự đoán câu mới

**Câu 1:**  
"The head of the state is coming today"

**Dự đoán:**  
```
[('The', 'DET'), ('head', 'NOUN'), ('of', 'ADP'), ('the', 'DET'),
 ('state', 'NOUN'), ('is', 'AUX'), ('coming', 'VERB'), ('today', 'NOUN')]
```

**Nhận xét:**  
Mô hình phân biệt chính xác giữa danh từ (head, state) và trợ động từ (is), cho thấy khả năng nắm bắt ngữ cảnh ngắn hạn tốt.

---

**Câu 2:**  
"I love NLP"

**Dự đoán:**  
```
[('I', 'PRON'), ('love', 'VERB'), ('NLP', 'PROPN')]
```

**Nhận xét:**  
Mô hình nhận diện đúng "NLP" là danh từ riêng (PROPN) dựa trên ngữ cảnh đứng sau động từ "love", mặc dù đây là từ ít xuất hiện trong tập huấn luyện.

---

## Kết luận

- **Hiệu năng:** Độ chính xác 85.62% trên tập Dev là kết quả khá tốt đối với một mô hình RNN cơ bản.
- **Sự hội tụ:** Loss giảm đều từ 1.02 xuống 0.27 cho thấy mô hình học ổn định và chưa xuất hiện hiện tượng overfitting nghiêm trọng trong 5 epoch đầu.
- **Hạn chế & Cải tiến:** Simple RNN có thể gặp khó khăn với các câu dài do hiện tượng triệt tiêu gradient. Để đạt độ chính xác trên 90%, mô hình có thể được nâng cấp lên kiến trúc **LSTM** hoặc **GRU**, hoặc kết hợp thêm **BiRNN** và **dropout**.

