# BÁO CÁO CHI TIẾT LAB 5: PHÂN LOẠI VĂN BẢN VỚI RNN / LSTM

---

## 1. Bảng so sánh kết quả định lượng (Quantitative Results)

Kết quả đo lường trên **tập kiểm tra (Test Set)** sau khi kết thúc quá trình huấn luyện:

| Pipeline | F1-score (Macro) | Accuracy (Test) | Test Loss | Nhận xét |
|--------|------------------|-----------------|-----------|----------|
| TF-IDF + Logistic Regression | 0.5386 | 0.5738 | N/A | Hoạt động tốt nhất nhờ bắt đặc trưng từ khóa (n-gram). |
| Word2Vec (Avg) + Dense | 0.0033 | 0.0167 | 4.1676 | Hiệu suất kém do mất hoàn toàn cấu trúc câu. |
| Embedding (Pre-trained) + LSTM | 0.0002 | 0.0074 | 4.1702 | Gần như không học được tri thức mới. |
| Embedding (Scratch) + LSTM | 0.1110 | 0.1356 | 3.6237 | Tốt nhất trong nhóm DL, có khả năng học ngữ cảnh. |

---

## 2. Phân tích định tính (Qualitative Analysis)

Các câu **"khó"** được sử dụng nhằm đánh giá khả năng xử lý **phủ định** và **phụ thuộc xa (long-range dependency)**.

### 2.1. Phân tích câu ví dụ điển hình

**Câu test:**  
> "find a flight from new york to london but not through paris"

**Thực tế:** Ý định tìm chuyến bay, với điều kiện **loại trừ Paris**.

- **TF-IDF dự đoán:** `general_negate`  
  **Lý do:** TF-IDF chỉ phát hiện từ khóa *"not"* và đưa ra dự đoán dựa trên tần suất, không hiểu rằng *"not"* đang bổ nghĩa cho cụm *"through paris"*.

- **LSTM (Scratch) dự đoán:** `datetime_convert`  
  *(Kết quả chưa đúng, nhưng Loss thấp hơn TF-IDF).*

**Giải thích:**  
Về lý thuyết, LSTM sử dụng các **cổng (gates)** để lưu trữ thông tin quan trọng như từ *"not"* trong **Cell State**. Tuy nhiên, với **tập dữ liệu nhỏ** và **từ vựng phong phú**, mô hình LSTM Scratch chưa đủ dữ liệu để học được mối liên hệ giữa cụm *"not through"* và nhãn `flight_search`.

---

### 2.2. Trường hợp phụ thuộc xa (Long-range Dependency)

**Câu ví dụ:**  
> "can you remind me to not call my mom"

Trong câu này, từ *"remind"* (đầu câu) và *"not"* (giữa câu) có mối quan hệ ngữ nghĩa chặt chẽ.

**Vì sao LSTM tốt hơn (về lý thuyết):**
- TF-IDF coi văn bản là **túi từ (Bag-of-Words)**, không xét thứ tự.
- LSTM duy trì **Hidden State** truyền qua từng token.
- Khi mô hình đọc đến *"call"*, nó vẫn giữ được thông tin từ *"remind"* và *"not"* trước đó.

**Thực tế trong bài Lab:**
- LSTM Scratch đạt **Accuracy = 13.56%**.
- **Loss giảm rõ rệt**, cho thấy mô hình đã bắt đầu học được các chuỗi phụ thuộc trong dữ liệu.

---

## 3. Nhận xét Ưu và Nhược điểm của các phương pháp

| Phương pháp | Ưu điểm | Nhược điểm |
|------------|---------|------------|
| TF-IDF + Logistic | - Đơn giản, huấn luyện rất nhanh. <br> - Độ chính xác cao với câu ngắn. | - Không hiểu ngữ pháp. <br> - Thất bại với câu đảo ngữ/phủ định. |
| Word2Vec (Avg) | - Giảm chiều dữ liệu hiệu quả. | - Lấy trung bình làm mất hoàn toàn trật tự từ. |
| LSTM (Chuỗi) | - Hiểu ngữ cảnh. <br> - Giải quyết tốt phụ thuộc xa. | - Cần nhiều dữ liệu và thời gian huấn luyện. <br> - Dễ overfitting khi dữ liệu nhỏ. |

---

## 4. Hướng dẫn chạy Code

### 4.1. Chuẩn bị dữ liệu
- Tải file `hwu.tar.gz`.
- Đưa vào Google Drive theo đường dẫn:
```
/content/drive/MyDrive/
```

### 4.2. Môi trường
- Sử dụng **Google Colab**.
- Cài đặt thư viện cần thiết:
```bash
pip install gensim
```

### 4.3. Thực thi
- Chạy các ô **Tiền xử lý**:
  - Mã hóa nhãn
  - Tokenizer
  - Padding chuỗi
- Chạy các ô **Huấn luyện (Nhiệm vụ 1 → 4)**.

> **Lưu ý:**  
> Các cảnh báo `UserWarning` liên quan đến `input_length` có thể bỏ qua do sự khác biệt giữa các phiên bản Keras.

- Kết quả cuối cùng sẽ được **tự động tổng hợp** trong bảng đánh giá ở **Nhiệm vụ 5**.

---

