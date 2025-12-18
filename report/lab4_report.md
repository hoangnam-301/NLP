# BÁO CÁO LAB 5: TEXT CLASSIFICATION

link code: https://github.com/hoangnam-301/NLP/blob/main/src/models/text_classifier.py
           https://github.com/hoangnam-301/NLP/blob/main/test/lab4_test.py
           https://github.com/hoangnam-301/NLP/blob/main/test/lab4_spark_sentiment_analysis.py
           https://github.com/hoangnam-301/NLP/blob/main/test/lab4_task4_improvement.py

## I. MỤC TIÊU (OBJECTIVE)
Xây dựng quy trình phân loại văn bản hoàn chỉnh:

Raw Text → Tokenization → Vectorization → Model Training → Evaluation.

Sử dụng thuật toán **Logistic Regression** để phân loại cảm xúc (tích cực / tiêu cực).  
So sánh hiệu năng giữa các phương pháp tiền xử lý và các thuật toán học máy khác nhau trên quy mô lớn với **Apache Spark**.

## II. CÁC BƯỚC THỰC HIỆN (IMPLEMENTATION)

### 1. Triển khai trên môi trường máy đơn (scikit-learn)

- **Task 1 & 2**: Xây dựng lớp `TextClassifier` trong `src/models/text_classifier.py`.  
  Lớp này tích hợp vectorizer để biến đổi văn bản và sử dụng `LogisticRegression` của `sklearn` làm mô hình lõi.

- **Task 3**:  
  - Sử dụng `RegexTokenizer` (Lab 1) và `TfidfVectorizer` (Lab 3) để tạo Pipeline.  
  - Chia dữ liệu theo tỉ lệ **80/20** cho huấn luyện và kiểm tra.

### 2. Triển khai trên môi trường phân tán (Apache Spark)

Sử dụng **Spark ML Pipeline** để xử lý file `sentiments.csv` với các giai đoạn:

Tokenizer → StopWordsRemover → HashingTF → IDF → LogisticRegression

## III. KẾT QUẢ THỰC NGHIỆM (EXPERIMENTAL RESULTS)

### 1. Kết quả kiểm thử cơ bản (Lab 5 Test)

- **Dữ liệu mẫu**: Các câu bình luận phim ngắn.
- **Đánh giá**: Mô hình dự đoán chính xác các câu mẫu mới.

Ví dụ:  
“I love the fantastic acting” → **Positive** (Chính xác)

### 2. Kết quả trên dữ liệu thực tế (Spark Sentiment Analysis)

Với tập dữ liệu `sentiments.csv` (dữ liệu tài chính / chứng khoán), mô hình cơ sở đạt được:

- **Accuracy**: 0.7295 (72.95%)

## IV. CẢI THIỆN MÔ HÌNH (TASK 4: IMPROVEMENT)

### 1. Bảng so sánh thuật toán

| Thuật toán | Accuracy | Ghi chú |
|-----------|----------|--------|
| Logistic Regression | 0.7737 | Tốt nhất, tăng ~5% so với mô hình cơ sở |
| Naive Bayes | 0.7349 | Hiệu quả với TF-IDF |
| Gradient-Boosted Trees | 0.7151 | Thấp nhất, dễ overfitting |

### 2. Các kỹ thuật cải thiện

- **Noise Filtering**: Loại bỏ URL, ký hiệu chứng khoán và ký tự đặc biệt.
- **Feature Expansion**: Tăng `numFeatures` trong `HashingTF` lên **20,000**.
- **Stratified Splitting**: Giữ cân bằng nhãn giữa train/test.

## V. PHÂN TÍCH VÀ KẾT LUẬN (CONCLUSION)

- Tiền xử lý quyết định hiệu năng mô hình.
- Logistic Regression phù hợp nhất với dữ liệu TF-IDF.
- Spark cho khả năng mở rộng và Pipeline mạnh.

## PHỤ LỤC

- `src/models/text_classifier.py`
- `test/lab5_test.py`
- `test/lab5_spark_sentiment_analysis.py`
- `test/lab5_task4_improvement.py`
