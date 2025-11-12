# BÁO CÁO LAB 6


## I. Mục tiêu 

Mục tiêu của bài lab này là xây dựng và so sánh hiệu năng giữa nhiều mô hình phân loại ý định (Intent Classification) khác nhau, bao gồm:

1. **TF-IDF + Logistic Regression** (mô hình baseline truyền thống)  
2. **Word2Vec trung bình + Dense Layer**  
3. **LSTM với Embedding Word2Vec pretrained**  
4. **LSTM học Embedding từ đầu**

Tập dữ liệu sử dụng là **HWU64**, gồm hơn 10.000 câu thuộc 64 lớp ý định khác nhau trong các lĩnh vực như lịch, thời tiết, nhạc, email, điều khiển thiết bị IoT, v.v.

---

## II. Kết quả định lượng

| **Pipeline** | **F1-score (Macro)** | **Test Loss** |
|---------------|----------------------|---------------|
| TF-IDF + Logistic Regression | **0.839** | N/A |
| Word2Vec (Avg) + Dense | **0.686** | 0.983 |
| Embedding (Pre-trained) + LSTM | **0.002** | 3.943 |
| Embedding (Scratch) + LSTM | **0.104** | 2.976 |

### Nhận xét:
- **TF-IDF + Logistic Regression** hoạt động rất tốt, F1-score cao và ổn định nhất.  
- **Word2Vec + Dense** đạt kết quả khá, cho thấy việc dùng embedding giúp mô hình học được mối quan hệ ngữ nghĩa giữa các từ.  
- Hai mô hình **LSTM (Pretrained & Scratch)** cho kết quả thấp do:
  - Dữ liệu huấn luyện hạn chế, mô hình LSTM phức tạp nên bị **underfitting**.
  - Embedding pretrained khởi tạo chưa được tối ưu — nhiều từ trong tokenizer không có trong Word2Vec.
  - Dropout và recurrent_dropout cao khiến mô hình khó hội tụ.

---

## III. Phân tích định tính 

Ba câu kiểm thử điển hình:

| **Câu ví dụ** | **Ý định đúng** | **Kết quả mô hình** |
|----------------|----------------|----------------------|
| “can you remind me to not call my mom” | `reminder_create` | TF-IDF → `calendar_set` ; Word2Vec → `social_post` ; LSTM → `iot_hue_lightoff` |
| “is it going to be sunny or rainy tomorrow” | `weather_query` | TF-IDF → `weather_query` ; Word2Vec → `weather_query` ; LSTM → `alarm_set` |
| “find a flight from new york to london but not through paris” | `flight_search` | TF-IDF → `transport_ticket` ; Word2Vec → `transport_query` ; LSTM → `alarm_set` |

### Phân tích:

1. **TF-IDF + LR** xử lý tốt các mẫu ngắn hoặc có từ khóa mạnh (“weather”, “rainy”). Tuy nhiên, mô hình **không hiểu ngữ cảnh phủ định hoặc phụ thuộc xa** (“not through paris”).  
   → Kết quả “transport_ticket” là hợp lý về chủ đề, nhưng chưa đúng ý định cụ thể.

2. **Word2Vec + Dense** hiểu tốt hơn mối quan hệ giữa các từ có nghĩa gần nhau.  
   → “weather_query” được nhận diện đúng vì embedding đã học được mối quan hệ giữa các từ “sunny”, “rainy”, “tomorrow”.  
   Tuy nhiên, vẫn khó nắm bắt cấu trúc cú pháp dài.

3. **LSTM (Pre-trained / Scratch)** đáng ra phải xử lý tốt các **phụ thuộc ngữ cảnh xa** (ví dụ “not call my mom”), nhưng do embedding yếu và dữ liệu nhỏ → mô hình không học được chuỗi ngữ nghĩa → dự đoán sai.  
   Nếu được huấn luyện đúng cách, LSTM có thể nắm bắt quan hệ “not + verb” tốt hơn các mô hình truyền thống.

---

## IV. So sánh ưu – nhược điểm của các phương pháp

| **Phương pháp** | **Ưu điểm** | **Nhược điểm** |
|------------------|-------------|----------------|
| **TF-IDF + Logistic Regression** | Dễ huấn luyện, nhanh, ít overfitting, hoạt động ổn với dữ liệu nhỏ. | Không hiểu ngữ cảnh, không xử lý được từ đồng nghĩa hay phủ định. |
| **Word2Vec (Avg) + Dense** | Hiểu ngữ nghĩa tốt hơn, đơn giản, hội tụ nhanh. | Mất thứ tự từ, không nắm được cú pháp hay quan hệ xa. |
| **LSTM (Pre-trained)** | Có khả năng học phụ thuộc xa, tận dụng embedding pretrained. | Dễ underfitting nếu dữ liệu ít; cần tinh chỉnh embedding kỹ. |
| **LSTM (Scratch)** | Linh hoạt, có thể học embedding riêng phù hợp domain. | Huấn luyện lâu, yêu cầu dữ liệu lớn, dễ overfitting hoặc không hội tụ. |

---

## V. Kết luận và hướng cải thiện

- Các mô hình truyền thống (**TF-IDF, Word2Vec**) vẫn hoạt động rất tốt trên tập dữ liệu nhỏ và đa lớp.
- Các mô hình **LSTM cần thêm dữ liệu và fine-tuning embedding tốt hơn** (ví dụ: sử dụng GloVe hoặc FastText thay vì Word2Vec tự huấn luyện).
- Ngoài ra, có thể thử **mô hình Transformer (BERT hoặc DistilBERT)** để cải thiện độ hiểu ngữ cảnh và phủ định.

📈 Với cải tiến preprocessing, fine-tuning embedding, và giảm dropout, LSTM dự kiến có thể đạt F1-score > **0.70**, vượt Word2Vec Dense.

---

## Tài liệu tham khảo
- HWU64 Dataset for Intent Classification.  
- Mikolov et al. (2013). *Distributed Representations of Words and Phrases and their Compositionality.*  
- Hochreiter & Schmidhuber (1997). *Long Short-Term Memory.*  
- Scikit-learn, TensorFlow, Gensim documentation.

---

