# BÁO CÁO KẾT QUẢ LAB 5: NHẬN DẠNG THỰC THỂ TÊN (NER)

link code: https://github.com/hoangnam-301/NLP/blob/main/notebook/lab5_rnn_for_ner.ipynb
---

## 1. Giải thích các bước thực hiện

Quy trình thực hiện bài Lab được chia thành các giai đoạn logic như sau:

### Bước 1: Tải và Tiền xử lý dữ liệu

- Sử dụng thư viện **`datasets` của Hugging Face** để tải bộ dữ liệu **CoNLL 2003 (conll2003_noMISC)**.
- Xây dựng từ điển **`word_to_ix`** để ánh xạ từ sang số nguyên:
  - Index `0`: `[PAD]` (padding)
  - Index `1`: `[UNK]` (từ không xuất hiện trong tập huấn luyện)
- Xây dựng **`tag_to_ix`** để quản lý các nhãn thực thể NER như: `B-PER`, `I-ORG`, `B-LOC`, `O`, …

---

### Bước 2: Xây dựng Dataset & DataLoader

- Tạo lớp **`NERDataset`** để chuyển đổi các câu văn bản thành tensor chỉ số (word indices và tag indices).
- Sử dụng hàm **`collate_fn`** kết hợp với **`pad_sequence`** để đồng bộ độ dài các câu trong cùng một batch.
- Các nhãn được đệm với giá trị **`-1`** nhằm phục vụ cơ chế **masking**, giúp bỏ qua các vị trí padding khi tính loss.

---

### Bước 3: Thiết kế mô hình Bi-LSTM

Mô hình Bi-LSTM cho bài toán NER gồm các thành phần chính:

- **Embedding Layer**  
  Chuyển đổi chỉ số từ thành vector embedding 128 chiều.

- **Bi-directional LSTM Layer**  
  - Sử dụng LSTM hai chiều với **128 hidden units cho mỗi chiều**.  
  - Kiến trúc hai chiều giúp mô hình học được ngữ cảnh từ cả phía trước và phía sau của mỗi từ trong câu.

- **Linear Layer**  
  Ánh xạ đầu ra của Bi-LSTM (256 chiều) về **7 nhãn NER** tương ứng.

---

### Bước 4: Huấn luyện và Đánh giá

- **Hàm lỗi:** `CrossEntropyLoss(ignore_index = -1)` để không tính loss tại các vị trí padding.
- **Optimizer:** Adam.
- **Số epoch huấn luyện:** 5 epoch.

Trong quá trình huấn luyện, mô hình được đánh giá trên tập Validation sau mỗi epoch để theo dõi mức độ hội tụ.

---

## 2. Hướng dẫn chạy code trên Google Colab

### Môi trường
- Mở một Notebook mới trên **Google Colab**.

### Cài đặt thư viện
Tại ô đầu tiên, chạy lệnh:
```bash
pip install datasets
```

### Thực thi
- Copy lần lượt các đoạn mã nguồn (khai báo thư viện, tiền xử lý dữ liệu, định nghĩa Dataset, Model và vòng lặp huấn luyện) vào các cell riêng biệt.
- Chạy toàn bộ notebook bằng cách chọn:
```
Runtime -> Run all
```

### Quan sát kết quả
- Loss và Accuracy theo từng epoch sẽ hiển thị trực tiếp ở output.
- Kết quả dự đoán cho câu ví dụ sẽ được in ra ở cell cuối.

---

## 3. Phân tích kết quả thực nghiệm

### 3.1. Độ chính xác trên tập Validation

Sau 5 epoch huấn luyện, mô hình đạt được kết quả như sau:

| Thông số | Kết quả thực tế |
|--------|----------------|
| Validation Accuracy | **95.41%** |
| Loss (Epoch 5) | **0.0224** |

**Nhận xét:**  
Độ chính xác 95.41% là một kết quả rất ấn tượng, cho thấy mô hình Bi-LSTM hoạt động hiệu quả trong bài toán nhận dạng thực thể tên. Giá trị Loss giảm mạnh và Accuracy tăng đều qua từng epoch chứng tỏ mô hình hội tụ tốt và học được các mẫu ngôn ngữ quan trọng.

---

### 3.2. Ví dụ dự đoán câu mới

**Câu kiểm tra:**  
"VNU University is located in Hanoi"

**Kết quả dự đoán chi tiết:**

| Từ | Nhãn dự đoán | Đánh giá |
|----|--------------|----------|
| VNU | B-ORG | Đúng (Bắt đầu tổ chức) |
| University | I-PER | Sai (Đúng phải là I-ORG) |
| is | O | Đúng |
| located | O | Đúng |
| in | O | Đúng |
| Hanoi | B-LOC | Đúng (Địa điểm) |

**Nhận xét:**  
Mô hình nhận diện rất tốt các thực thể phổ biến như **ORG** và **LOC**. Tuy nhiên, lỗi tại từ *"University"* cho thấy mô hình vẫn còn hạn chế trong việc gán đúng nhãn tiếp diễn thực thể (I-ORG), đặc biệt với các cụm danh từ dài.

---

## Kết luận

- **Ưu điểm:** Mô hình Bi-LSTM cho kết quả chính xác cao (95.41%), thể hiện rõ ưu thế của việc học ngữ cảnh hai chiều trong bài toán NER.
- **Hạn chế:** Một số lỗi nhỏ vẫn xuất hiện ở ranh giới thực thể hoặc các cụm thực thể nhiều từ.
- **Hướng cải tiến:**  
  - Kết hợp **Bi-LSTM + CRF** để mô hình hóa ràng buộc giữa các nhãn.  
  - Sử dụng embedding ngữ cảnh như **BERT** hoặc **RoBERTa** để cải thiện độ chính xác.

