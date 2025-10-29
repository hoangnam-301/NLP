## Lab 5: Text Classification

### 📊 Phân tích Kết quả (Result Analysis)

#### 1. Báo cáo Metrics của Mô hình Baseline Ban Đầu

| Metric   |   Giá trị  |
| :------- | :--------: |
| Accuracy | **0.0000** |
| F1-score | **0.0000** |

**Phân tích:** Kết quả bằng **0.0000** trên tập 6 mẫu khẳng định mô hình Baseline thất bại hoàn toàn. Nguyên nhân chính là do **quá khớp (overfitting)** nghiêm trọng vì dữ liệu huấn luyện quá ít, dẫn đến mô hình không thể tổng quát hóa, thậm chí còn **dự đoán ngược nhãn** (True: [1, 0], Pred: [0, 1]).

---

#### 2. Báo cáo Metrics của Mô hình Cải tiến

| Mô hình                       |   Đặc trưng   | Phương pháp | Accuracy (AVG) | F1-score (AVG) |
| :---------------------------- | :-----------: | :---------: | :------------: | :------------: |
| Multinomial Naive Bayes (MNB) | Count/Unigram |  5-Fold CV  |      0.769     |      0.823     |
| Logistic Regression (LR)      | Count/Unigram |  5-Fold CV  |    **0.788**   |    **0.838**   |

---

#### 3. So sánh và Phân tích Hiệu quả Kỹ thuật Cải tiến

| Yếu tố Cải tiến                     |      Hiệu quả     | Phân tích                                                                                                                                                                                 |
| :---------------------------------- | :---------------: | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Tăng Kích thước Dữ liệu & CV**    |  Cực kỳ Hiệu quả  | Đưa hiệu suất từ *0.0000* lên **0.788**. Việc sử dụng 5791 mẫu và Cross-Validation là yếu tố cải tiến quan trọng nhất, giúp khắc phục overfitting và cung cấp ước tính hiệu suất ổn định. |
| **So sánh Thuật toán (LR vs. MNB)** |  LR vượt trội MNB | Logistic Regression đạt **F1 = 0.838**, cao hơn MNB (F1 = 0.823). LR là mô hình tuyến tính mạnh, có khả năng học các mối quan hệ phức tạp hơn và vượt qua giả định độc lập từ của MNB.    |
| **Lọc min_df = 2**                  | Hiệu quả tích cực | Giúp giảm nhiễu (*noise*) bằng cách loại bỏ các từ hiếm chỉ xuất hiện 1 lần, tập trung vào các từ khóa có tính phân loại cao.                                                             |

**Kết luận:** Sự cải thiện hiệu suất chủ yếu đến từ việc tăng quy mô dữ liệu và đánh giá ổn định. **Mô hình Logistic Regression** là lựa chọn tối ưu, đạt **F1-score = 0.838**.

---

### 🔗 Thách thức và Giải pháp (Challenges and Solutions)

| Thách thức                   | Mô tả                                                                                                | Giải pháp                                                                                                                                   |
| :--------------------------- | :--------------------------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------ |
| **Mô hình Thất bại Ban Đầu** | Mô hình Baseline (6 mẫu) đạt 0% Accuracy, không thể đánh giá.                                        | Sử dụng tập dữ liệu lớn (*data/sentiments.csv*) và Cross-Validation để có kết quả đáng tin cậy.                                             |
| **Xử lý Vocabulary (Rò rỉ)** | Cần đảm bảo việc xây dựng từ vựng và lọc *min_df* chỉ thực hiện trên tập huấn luyện của mỗi fold CV. | Sử dụng hàm `build_token_filter` để tạo tokenizer đã lọc, sau đó truyền vào CountVectorizer (fit_transform() chỉ học từ vựng từ tập Train). |
| **Đánh giá Công bằng**       | Kết quả trên tập lớn có thể bị ảnh hưởng bởi Train/Test Split ngẫu nhiên.                            | Sử dụng `StratifiedKFold` (5-Fold CV) đảm bảo phân phối nhãn đồng đều và lấy giá trị trung bình (AVG) của các metrics.                      |

---

### 📚 Tài liệu Tham khảo (References)

* **scikit-learn Documentation:** Hướng dẫn sử dụng các lớp `LogisticRegression`, `MultinomialNB`, `StratifiedKFold` và các hàm đánh giá hiệu suất.
* **Giáo trình Lab 2 & 3:** Triển khai các thành phần tiền xử lý cốt lõi (`RegexTokenizer`, `CountVectorizer`).
* **Tập dữ liệu:** *data/sentiments.csv* (Nguồn dữ liệu thực nghiệm cho bài toán phân tích tình cảm).
