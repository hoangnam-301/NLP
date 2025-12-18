# BÁO CÁO LAB 3.1: GIẢM CHIỀU VÀ TRỰC QUAN HÓA WORD VECTOR
 link code: https://github.com/hoangnam-301/NLP/blob/main/notebook/lab3.ipynb
---

## 1. Giải thích các bước thực hiện

Quy trình thực hiện bài Lab được chia thành **4 giai đoạn chính**, từ chuẩn bị dữ liệu đến phân tích kết quả trực quan hóa.

### 1.1. Chuẩn bị dữ liệu

Trong bài lab này, ta sử dụng bộ **GloVe (Global Vectors for Word Representation)** – một tập word embedding được huấn luyện sẵn trên hàng tỷ từ từ Wikipedia và các nguồn văn bản lớn.

- File sử dụng: `glove.6B.50d.txt`
- Số lượng từ: khoảng **400.000 từ**
- Mỗi từ được biểu diễn bằng một **vector 50 chiều**

Việc sử dụng embedding huấn luyện sẵn giúp ta tập trung vào việc **phân tích và trực quan hóa không gian ngữ nghĩa**, thay vì phải huấn luyện mô hình từ đầu.

---

### 1.2. Đọc và xử lý vector

Chương trình đọc từng dòng trong file GloVe, trong đó:
- Phần tử đầu tiên là **tên từ**
- Các phần tử tiếp theo là **giá trị số của vector embedding**

Các bước xử lý chính:
- Tách từ và vector tương ứng
- Chuyển vector sang mảng **NumPy** với kiểu dữ liệu `float32` để tối ưu hóa bộ nhớ và tốc độ tính toán
- Lưu trữ dữ liệu dưới dạng danh sách hoặc từ điển để thuận tiện cho việc truy xuất

---

### 1.3. Giảm chiều dữ liệu (Dimensionality Reduction)

Do vector GloVe ban đầu nằm trong không gian **50 chiều**, ta không thể trực quan hóa trực tiếp. Vì vậy, thuật toán **PCA (Principal Component Analysis)** được sử dụng để giảm chiều dữ liệu xuống **2 chiều**.

Nguyên lý của PCA:
- Tìm ra các **thành phần chính (principal components)**
- Đây là các phương hướng mà dữ liệu có phương sai lớn nhất
- Giữ lại các đặc trưng quan trọng nhất khi chiếu dữ liệu xuống mặt phẳng $Oxy$

Việc giảm từ 50 chiều xuống 2 chiều giúp ta **quan sát cấu trúc ngữ nghĩa** của từ vựng một cách trực quan.

---

### 1.4. Trực quan hóa (Visualization)

- Vẽ **scatter plot** với khoảng **10.000 từ phổ biến** làm nền (màu xám nhạt)
- Các từ khóa mục tiêu được:
  - Giảm chiều bằng PCA
  - Vẽ nổi bật bằng **màu đỏ**
  - Gắn nhãn văn bản trực tiếp trên biểu đồ

Cách biểu diễn này giúp làm rõ mối quan hệ không gian giữa các từ trong embedding.

---

## 2. Hướng dẫn chạy code trên Google Colab

Để tái hiện lại kết quả của bài Lab, thực hiện các bước sau:

### Bước 1: Tải dữ liệu GloVe

Chạy đoạn lệnh sau trong cell đầu tiên của notebook:

```python
!wget http://nlp.stanford.edu/data/glove.6B.zip
!unzip glove.6B.zip
!mkdir -p datasets
!mv glove.6B.50d.txt datasets/
```

---

### Bước 2: Thực thi code xử lý

- Copy toàn bộ mã nguồn Python (import thư viện, đọc dữ liệu, PCA, hàm `plot_words`) vào cell tiếp theo.
- Đảm bảo đã cài đặt các thư viện cần thiết như `numpy`, `scikit-learn`, `matplotlib`.

---

### Bước 3: Xem kết quả trực quan hóa

Gọi hàm trực quan hóa, ví dụ:

```python
plot_words(['man', 'woman', 'king', 'queen', 'paris', 'france', 'tokyo', 'japan'])
```

Biểu đồ PCA 2D sẽ được hiển thị ngay bên dưới cell thực thi.

---

## 3. Phân tích kết quả thực nghiệm (Quan trọng)

Dựa trên kết quả trực quan hóa, ta có thể rút ra nhiều nhận xét quan trọng về cách mô hình học và lưu trữ ngữ nghĩa.

### 3.1. Quan hệ tương quan ngữ nghĩa

Các từ cùng nhóm ngữ nghĩa (ví dụ: màu sắc, giới tính, quốc gia) có xu hướng **co cụm (cluster)** lại gần nhau trong không gian 2D. Điều này chứng minh rằng:
- Word embedding không chỉ lưu trữ mặt chữ
- Mà thực sự học được **ý nghĩa của từ dựa trên ngữ cảnh xuất hiện**

---

### 3.2. Cấu trúc logic và tính bắc cầu (Analogy)

Các cặp từ có mối quan hệ tương ứng như:
- `tokyo – japan`
- `paris – france`

Tạo thành các đoạn thẳng gần như **song song** trên biểu đồ PCA. Điều này cho thấy:
- Mô hình đã học được **quan hệ logic và tri thức thế giới**
- Đây chính là cơ sở cho các phép toán nổi tiếng như:

\[ king - man + woman \approx queen \]

---

### 3.3. Khả năng phân loại ngữ cảnh (Outliers)

Những từ không liên quan về mặt ngữ nghĩa (ví dụ: `facebook`) bị đẩy ra xa khỏi các nhóm từ như màu sắc (`blue`, `red`, `green`).

Vị trí tách biệt này cho thấy:
- Word embedding có khả năng **phân loại ngữ cảnh** tốt
- Các từ khác chủ đề sẽ có biểu diễn không gian khác biệt rõ rệt

---

### 3.4. Biểu diễn sự biến đổi từ vựng

Các dạng so sánh của tính từ như:
- `good – better – best`

Có xu hướng di chuyển theo một **lộ trình có quy luật** trong không gian embedding. Điều này thể hiện rằng mô hình:
- Hiểu được mối quan hệ ngữ pháp
- Biểu diễn được các cấp độ biến đổi của từ

---

### 3.5. Nhận xét chung

- Việc giảm từ **50 chiều xuống 2 chiều** chắc chắn gây mất mát thông tin
- Tuy nhiên, **các đặc trưng ngữ nghĩa cốt lõi vẫn được giữ lại**
- Các từ phổ biến có vị trí ổn định và chính xác hơn do được huấn luyện trên nhiều ngữ cảnh

---

## 4. Kết luận

Bài lab đã minh họa rõ ràng sức mạnh của **Word Embedding** trong việc mã hóa ngữ nghĩa từ vựng dưới dạng vector số. Thông qua PCA và trực quan hóa, ta có thể quan sát được:

- Cấu trúc ngữ nghĩa tiềm ẩn trong không gian vector
- Các mối quan hệ logic và ngữ pháp giữa từ
- Khả năng tổng quát hóa tri thức của mô hình

Đây là nền tảng quan trọng cho các mô hình NLP hiện đại như **Word2Vec, GloVe, FastText và Transformer-based embeddings**.
