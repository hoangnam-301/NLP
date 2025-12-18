# BÁO CÁO LAB 3.2: WORD EMBEDDINGS WITH WORD2VEC

link code: https://github.com/hoangnam-301/NLP/blob/main/src/representations/word_embedder.py
           https://github.com/hoangnam-301/NLP/blob/main/test/lab3_test.py

## I. GIỚI THIỆU CHUNG (OBJECTIVE)
Mục tiêu của bài lab là chuyển đổi phương thức biểu diễn văn bản từ dạng thưa thớt, đa chiều (TF-IDF) sang dạng vector dày đặc (dense vectors) ít chiều hơn nhưng mang đậm tính ngữ nghĩa. Qua đó, học cách sử dụng các mô hình đã được huấn luyện sẵn (Pre-trained) và huấn luyện mô hình từ đầu trên hệ thống tính toán phân tán Apache Spark.

## II. CÁC BƯỚC THỰC HIỆN (IMPLEMENTATION STEPS)

### Phần 1: Word Embedding với Gensim (Máy đơn)

- **Cấu hình môi trường**: Cài đặt thư viện `gensim` và các phụ thuộc trong `requirements.txt`.
- **Triển khai lớp WordEmbedder**: Viết file `word_embedder.py` để nạp mô hình `glove-wiki-gigaword-50`.
- **Các phương thức**:
  - `get_vector(word)`
  - `get_similarity(w1, w2)`
  - `get_most_similar(word)`
- **Document Embedding**: Trung bình vector các từ.

### Phần 2: Word2Vec với Apache Spark

- Khởi tạo Spark Session
- Tiền xử lý dữ liệu C4 (`.json.gz`)
- Huấn luyện Word2Vec (`vectorSize=100`, `minCount=5`)

## III. CÁCH CHẠY CODE

```bash
python test/lab3_test.py
python test/lab3_spark_word2vec_demo.py
```

## IV. GIẢI THÍCH KẾT QUẢ

- `king` – `queen`: similarity cao
- Vector văn bản được nén thành 1 vector duy nhất
- Kết quả Spark mang tính thực dụng (desktop, laptop)

## V. THAM CHIẾU

- Gensim Documentation
- Spark MLlib Word2Vec
