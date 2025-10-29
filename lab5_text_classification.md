#  Báo Cáo Lab 5

## Mục tiêu

Triển khai một pipeline phân loại văn bản hoàn chỉnh, từ tiền xử lý đến huấn luyện và đánh giá. Mục tiêu chính là so sánh hiệu suất giữa mô hình **Logistic Regression (LR)** và **Multinomial Naive Bayes (MNB)** trên tập dữ liệu lớn bằng phương pháp **K-Fold Cross-Validation** để tìm ra mô hình tối ưu.

---

## Chi tiết Triển khai 

Quá trình triển khai được thực hiện trong hai file Python chính, sử dụng các lớp tự xây dựng (*CountVectorizer*) và mô hình từ thư viện **scikit-learn**.

### 1. Thiết lập Baseline (`test/lab5_test.py`)

* **Dữ liệu:** Sử dụng tập dữ liệu rất nhỏ (6 mẫu).
* **Tiền xử lý:** Sử dụng `RegexTokenizer` và `CountVectorizer` để chuyển đổi văn bản thành vector tần suất từ (Count features).
* **Mô hình:** Khởi tạo lớp `TextClassifier` đóng gói mô hình Logistic Regression.
* **Đánh giá:** Thực hiện Train/Test Split đơn giản (80/20).

### 2. Cải tiến và So sánh Hiệu suất (`test/lab5_improvement_test.py`)

* **Tải Dữ liệu Lớn:** Tải 5791 mẫu từ `data/sentiments.csv`.
* **Tiền xử lý Cải tiến:** Áp dụng lọc `min_df = 2` (lọc từ hiếm) và sử dụng Count Vectorization với Unigram features.
* **Đánh giá Ổn định:** Sử dụng **K = 5** Fold *Stratified Cross-Validation* để đảm bảo tính ổn định và khách quan của metrics.
* **So sánh Thuật toán:** Huấn luyện và so sánh hiệu suất của Logistic Regression và Multinomial Naive Bayes.

---

## Chạy Code 

### Yêu cầu Cài đặt

```bash
pip install scikit-learn numpy pandas
```

### Các bước Chạy

Thực thi tuần tự hai file để thu thập dữ liệu so sánh:

**Chạy Baseline:**

```bash
python test/lab5_test.py
```

**Chạy Cải tiến và So sánh:**

```bash
python test/lab5_improvement_test.py
```

---

## Phân tích Kết quả 

### 1. Báo cáo Metrics của Mô hình Baseline Ban đầu

| Metric   |   Giá trị  |
| :------- | :--------: |
| Accuracy | **0.0000** |
| F1-score | **0.0000** |

**Phân tích:** Kết quả bằng **0.0000** trên tập 6 mẫu khẳng định mô hình Baseline thất bại hoàn toàn. Nguyên nhân là do **quá khớp (overfitting)** nghiêm trọng vì dữ liệu huấn luyện quá ít, dẫn đến mô hình không thể tổng quát hóa, thậm chí còn **dự đoán ngược nhãn** (True: [1, 0], Pred: [0, 1]).

### 2. Báo cáo Metrics của Mô hình Cải tiến

| Mô hình                       | Đặc trưng     | Phương pháp | Accuracy (AVG) | F1-score (AVG) |
| :---------------------------- | :------------ | :---------- | :------------: | :------------: |
| Multinomial Naive Bayes (MNB) | Count/Unigram | 5-Fold CV   |      0.769     |      0.823     |
| Logistic Regression (LR)      | Count/Unigram | 5-Fold CV   |    **0.788**   |    **0.838**   |

### 3. So sánh và Phân tích Hiệu quả Kỹ thuật Cải tiến

| Yếu tố Cải tiến                     |      Hiệu quả     | Phân tích                                                                                                                                                                              |
| :---------------------------------- | :---------------: | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Tăng Kích thước Dữ liệu & CV**    |  Cực kỳ Hiệu quả  | Đưa hiệu suất từ *0.0000* lên **0.788**. Việc sử dụng 5791 mẫu và Cross-Validation là kỹ thuật cải tiến quan trọng nhất, khắc phục overfitting và cung cấp ước tính hiệu suất ổn định. |
| **So sánh Thuật toán (LR vs. MNB)** |  LR vượt trội MNB | Logistic Regression đạt **F1 = 0.838**, cao hơn MNB (F1 = 0.823). LR là mô hình tuyến tính mạnh, có khả năng học các mối quan hệ phức tạp hơn và vượt qua giả định độc lập từ của MNB. |
| **Lọc `min_df = 2`**                | Hiệu quả tích cực | Giúp giảm nhiễu (*noise*) bằng cách loại bỏ các từ hiếm chỉ xuất hiện 1 lần, làm cho mô hình tập trung vào các từ khóa có tính phân loại cao, đặc biệt quan trọng với Count features.  |

**Kết luận:** Sự cải thiện hiệu suất chủ yếu đến từ việc tăng quy mô dữ liệu và đánh giá ổn định. **Mô hình Logistic Regression** là lựa chọn tối ưu, đạt **F1-score = 0.838**.

---

## Thách thức và Giải pháp 

| Thách thức                   | Mô tả                                                                                                | Giải pháp                                                                                                                                             |
| :--------------------------- | :--------------------------------------------------------------------------------------------------- | :---------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Mô hình Thất bại Ban đầu** | Mô hình Baseline (6 mẫu) đạt 0% Accuracy, không thể đánh giá.                                        | Chuyển sang sử dụng tập dữ liệu lớn (*data/sentiments.csv*) và phương pháp Cross-Validation để có kết quả đáng tin cậy.                               |
| **Xử lý Vocabulary (Rò rỉ)** | Đảm bảo rằng việc xây dựng từ vựng (và lọc `min_df`) chỉ xảy ra trên tập huấn luyện của mỗi fold CV. | Sử dụng hàm `build_token_filter` để tạo tokenizer đã lọc, sau đó truyền vào CountVectorizer (đảm bảo `fit_transform()` chỉ học từ vựng từ tập Train). |
| **Đánh giá Công bằng**       | Kết quả đánh giá trên tập lớn có thể bị ảnh hưởng bởi lựa chọn ngẫu nhiên của Train/Test Split.      | Sử dụng `StratifiedKFold` (5-Fold CV) để đảm bảo phân phối nhãn đồng đều và lấy giá trị trung bình (AVG) của các metrics.                             |

---

## Tài liệu Tham khảo (References)

* **scikit-learn Documentation:** Hướng dẫn sử dụng các lớp `LogisticRegression`, `MultinomialNB`, `StratifiedKFold`, và các hàm đánh giá hiệu suất.
* **Giáo trình Lab 2 & 3:** Triển khai các thành phần tiền xử lý cốt lõi (`RegexTokenizer`, `CountVectorizer`).
* **Tập dữ liệu:** `data/sentiments.csv` – Nguồn dữ liệu thực nghiệm cho bài toán phân tích tình cảm.
