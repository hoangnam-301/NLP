# 📄 Báo Cáo Lab 17: Xây Dựng Spark NLP Pipeline

---

## I. Mục Tiêu Và Các Giai Đoạn Thực Hiện

Mục tiêu của Lab là xây dựng một **Spark ML Pipeline** để tiền xử lý và chuyển đổi dữ liệu văn bản lớn (tập dữ liệu **C4**) thành vector đặc trưng phục vụ cho các tác vụ Machine Learning, đồng thời thử nghiệm các phương pháp vectorization khác nhau (**TF-IDF** và **Word2Vec**).

### 1. Các Giai Đoạn Chính Của Pipeline

1. **Data Loading**  
   Đọc dữ liệu JSON (`c4-train.json.gz`) thành Spark `DataFrame`.

2. **Preprocessing**  
   - Tokenization (sử dụng `RegexTokenizer`).  
   - Loại bỏ từ dừng (sử dụng `StopWordsRemover`).

3. **Vectorization (Thí nghiệm)**  
   Chuyển đổi tokens thành vector đặc trưng số (TF-IDF hoặc Word2Vec).

4. **Modeling (Thí nghiệm)**  
   Huấn luyện mô hình phân loại (LogisticRegression) dựa trên các vector đặc trưng đã sinh ra.

---

## II. Triển Khai Và Cấu Hình Kỹ Thuật

### 1. Các Bước Triển Khai (Implementation Steps)

| Giai đoạn               | Hành động thực hiện                                                                                                                                                                                                 |
| ----------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1. Cấu hình môi trường  | Cài đặt **Java JDK 17** và **sbt (Scala Build Tool)**. Cấu hình `build.sbt` để sử dụng **Spark 3.5.1** và **Scala 2.12.18**.                                                                                       |
| 2. Data & ID/Label      | Đọc dữ liệu, giới hạn ~1000 records, và thêm cột `id` (để theo dõi tài liệu) cùng cột `label` (nhãn giả, sử dụng cho Ex3).                                                                                          |
| 3. Pipeline Stages      | Định nghĩa các stage: `RegexTokenizer`, `StopWordsRemover`.                                                                                                                                                          |
| 4. TF-IDF Pipeline      | Sử dụng `HashingTF` (numFeatures từ 20,000 xuống 1,000), `IDF`, và `Normalizer` (chuẩn hóa L2). Thêm `LogisticRegression` cho Ex3.                                                                                  |
| 5. Word2Vec Pipeline    | Thay thế chuỗi TF-IDF + Normalizer bằng stage `Word2Vec` (vectorSize = 100) cho Ex4.                                                                                                                               |
| 6. Performance & Logging| Thêm logic đo thời gian (sử dụng `nanoTime`) cho các giai đoạn **Read**, **Fit**, và **Transform**.                                                                                                                |
| 7. Similarity Check     | Thêm hàm `findSimilarDocuments` sử dụng **Cosine Similarity** trên các vector đã chuẩn hóa để tìm tài liệu tương tự.                                                                                                |

### 2. Cách Chạy Code Và Ghi Log

- **Lưu code:**  
  Đảm bảo file `Lab17_NLPPipeline.scala` được lưu trong thư mục `src/main/scala/...`.

- **Chạy lệnh:**  
  Từ thư mục gốc của dự án (`spark_labs`), chạy:

```bash
sbt run
```

- **Kết quả log / output (theo cấu hình bài làm):**
  - **Hiệu suất & Similarity (Log):**  
    - `log/lab17_metrics.log`  
    - (hoặc `log/lab17_run_log_ex4.txt` cho Ex4).
  - **Vector đầu ra (kết quả transform / predict):**  
    - `results/lab17_pipeline_output.txt`  
    - (hoặc các biến thể: `_ex3`, `_ex4`).

---

## III. Phân Tích Kết Quả Thu Được

### 1. Phân Tích Tokenization (Ex1: Tokenizer vs. RegexTokenizer)

| Phương pháp          | Kích thước từ vựng (ước lượng) | Nhận xét                                                                                                                                      |
| -------------------- | ------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------- |
| RegexTokenizer (gốc) | ≈ 31,355 terms                 | Loại bỏ các ký tự đặc biệt, tạo ra các tokens "sạch" (thường là từ). Phù hợp cho NLP vì giảm nhiễu từ punctuation và ký tự lạ.               |
| Tokenizer (Ex1)      | ≈ 46,838 terms                 | Kích thước từ vựng tăng mạnh. Giữ lại nhiều chuỗi không phải từ (ví dụ dấu câu dính với từ, số, ký tự đặc biệt), làm từ vựng phình to, nhiễu.|

**Kết luận:**  
`RegexTokenizer` hiệu quả hơn trong việc làm sạch dữ liệu văn bản và kiểm soát kích thước từ vựng.

---

### 2. Phân Tích Vectorization (Ex2: HashingTF Size)

Khi giảm `numFeatures` từ **20,000** xuống **1,000** (trong khi số từ duy nhất khoảng **31,355**):

- **Tác động chính:**
  - Xảy ra hiện tượng **hash collision** (va chạm băm): nhiều từ khác nhau bị ánh xạ vào cùng một chỉ mục vector.
- **Hệ quả:**
  - Vector đặc trưng bị nén mạnh xuống 1,000 chiều ⇒ **mất mát thông tin**.
  - Giảm khả năng phân biệt giữa các tài liệu, đặc biệt khi dùng Cosine Similarity để đo mức độ tương đồng:
    - Giá trị similarity trở nên "nhiễu" hơn, kém phản ánh đúng nội dung thực.

---

### 3. Phân Tích Word2Vec vs. TF-IDF (Ex4)

| Đặc điểm          | TF-IDF (Output: 20,000 chiều)                                                                 | Word2Vec (Output: 100 chiều)                                                                                 |
| ----------------- | ---------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------ |
| Ý nghĩa vector    | Dựa trên **tần suất và mức độ quan trọng** của từ trong tài liệu/toàn bộ tập dữ liệu.         | Dựa trên **ngữ nghĩa**, mô hình học **ngữ cảnh** xung quanh từ (word embeddings).                            |
| Cấu trúc vector   | Rất thưa (**sparse**), kích thước lớn.                                                         | Dày đặc (**dense**), kích thước nhỏ (100 chiều), tiết kiệm bộ nhớ hơn.                                       |
| Thời gian huấn luyện | Thường nhanh hơn, chủ yếu là tính toán IDF và transform.                                   | Chậm hơn (≈ 6.25 giây trong bài lab) vì phải huấn luyện mô hình mạng nơ-ron nhỏ để học embedding.           |
| Cosine Similarity | Tốt cho các tài liệu dùng **chung từ khóa** / chung chủ đề bề mặt.                            | Tốt cho các tài liệu **ngữ nghĩa tương tự**, kể cả khi dùng **từ khác nhau** nhưng cùng ngữ cảnh.          |

**Nhận xét:**  
- TF-IDF mạnh về tần suất, phù hợp cho các mô hình tuyến tính, văn bản ngắn, và bài toán phân loại cổ điển.  
- Word2Vec nắm bắt ngữ nghĩa tốt hơn, hữu ích cho tasks như similarity, clustering, hoặc làm input cho mô hình sâu.

---

### 4. Phân Tích Logistic Regression (Ex3)

- **Mô hình:**  
  `LogisticRegression` được thêm vào cuối Pipeline.

- **Dữ liệu đầu vào:**  
  Sử dụng TF-IDF đã được chuẩn hóa L2 làm **features**, cùng với **nhãn giả** (`label` = 0.0 hoặc 1.0).

- **Quan sát log thời gian:**
  - Thời gian fitting tăng đáng kể (khoảng **3.0 giây**).  
  - Nguyên nhân: thuật toán tối ưu hóa (L-BFGS) cần nhiều vòng lặp trên toàn bộ dữ liệu để hội tụ.

- **Đầu ra:**  
  `DataFrame` kết quả chứa:
  - `label`, `prediction`, (và có thể `probability`, `rawPrediction` tùy cấu hình),
  - Xác nhận toàn bộ **Machine Learning Pipeline** (từ load → preprocess → vectorize → train → predict) hoạt động đúng.

---

