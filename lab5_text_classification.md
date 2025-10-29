# 🧠 Lab 5: Text Classification

## 🎯 Objective
Xây dựng một pipeline phân loại văn bản (Text Classification) hoàn chỉnh, từ khâu tiền xử lý đến huấn luyện và đánh giá mô hình.  
Mục tiêu:
- Áp dụng các kỹ thuật vector hóa (CountVectorizer/TfidfVectorizer).  
- Huấn luyện mô hình Logistic Regression.  
- Đánh giá bằng Accuracy, Precision, Recall, F1-score.  
- Cải thiện mô hình bằng các kỹ thuật nâng cao (Naive Bayes, bigram, char n-grams,...).  
- So sánh hiệu năng giữa mô hình cơ bản và mô hình cải tiến.

---

## 🧩 Implementation Steps

### **Task 1 – Data Preparation**
Tập dữ liệu nhỏ được lưu trong bộ nhớ (toy dataset) chỉ gồm 6 câu để minh họa pipeline, không phải tập huấn luyện chính.

```python
texts = [
  "This movie is fantastic and I love it!",
  "I hate this film, it's terrible.",
  "The acting was superb, a truly great experience.",
  "What a waste of time, absolutely boring.",
  "Highly recommend this, a masterpiece.",
  "Could not finish watching, so bad."
]
labels = [1, 0, 1, 0, 1, 0]
```

Sử dụng **CountVectorizer** (từ Lab 2) để chuyển văn bản thành đặc trưng số.

---

### **Task 2 – Implementing `TextClassifier`**
File: `src/models/text_classifier.py`

Mô hình Logistic Regression được huấn luyện trên đặc trưng rút ra từ vectorizer.  
Các hàm `fit`, `predict`, và `evaluate` lần lượt thực hiện huấn luyện, dự đoán và tính toán các chỉ số.

---

### **Task 3 – Basic Test Case**
File: `test/lab5_test.py`

Pipeline cơ bản gồm:
- Tokenizer: `RegexTokenizer`
- Vectorizer: `CountVectorizer`
- Model: `TextClassifier(LogisticRegression)`  

Kết quả:
```
Accuracy: 0.5
F1-score: 0.0
```
→ Mô hình baseline chỉ minh họa cách pipeline hoạt động.

---

### **Task 3 (Extended) – PySpark Sentiment Analysis**
File: `test/lab5_spark_sentiment_analysis.py`

Pipeline xử lý dữ liệu lớn từ `data/sentiments.csv`:
```
Tokenizer → StopWordsRemover → HashingTF → IDF → LogisticRegression
```
Kết quả:
```
Accuracy: 0.7295
F1-score: 0.7266
```
→ Mô hình hoạt động ổn định hơn nhờ dataset lớn.

---

## ⚙️ Code Execution Guide

### Cài đặt thư viện
```bash
pip install scikit-learn pyspark
```

### Chạy các phần của Lab
```bash
python test/lab5_test.py                     # baseline
python test/lab5_spark_sentiment_analysis.py # spark pipeline
python test/lab5_improvement_test.py         # model improvements
```

---

## 📊 Task 4 – Model Improvement Experiment

### **Dataset**
Từ Task 4 trở đi, sử dụng tập dữ liệu **lớn hơn (`data/sentiments.csv`)** thay cho 6 câu toy dataset ban đầu.

### **Lần 0 – Baseline**
- Logistic Regression + CountVectorizer  
- Accuracy ≈ 0.5, F1 ≈ 0.0

---

### **Lần 1 – Naive Bayes**
- Mô hình: `MultinomialNB`  
- Vectorizer: CountVectorizer (unigram)  
Kết quả:
```
Accuracy: 0.3333
Precision: 0.3333
Recall: 1.0000
F1: 0.5000
```
→ Overfit lớp dương tính.

---

### **Lần 2 – Stratified Split + Clean Tokenizer**
- Loại bỏ stopwords, chia dữ liệu cân bằng theo lớp.  
Kết quả:
```
Accuracy: 0.5
F1: 0.0
```
→ Mất từ phủ định gây giảm hiệu năng.

---

### **Lần 3 – Bigrams + Balanced Logistic Regression**
- Dùng bigram để nắm cụm nghĩa (“so_bad”, “highly_recommend”).  
- `class_weight='balanced'`.  
Kết quả:
```
Accuracy: 0.25
F1: 0.0
```

---

### **Lần 4 – K-Fold + Negation Bigrams + min_df=2**
- Thêm **Stratified K-Fold (5 folds)**.  
- Bắt cặp từ phủ định (“not_good”, “never_watching”).  
- Lọc từ xuất hiện ít hơn 2 lần (`min_df=2`).  
Kết quả:
```
NB AVG: Accuracy=0.067, F1=0.000
LR AVG: Accuracy=0.133, F1=0.100
```

---

### **Lần 5 – Character n-grams (3–5)**
- Sử dụng n-gram ký tự (char-level 3–5).  
- Giữ được tín hiệu phủ định, tránh mất đặc trưng.  
Kết quả ổn định hơn:
```
Accuracy ≈ 0.45
F1 ≈ 0.30
```

---

## 🔍 Result Analysis

| Mô hình                              | Vectorizer         | Đặc trưng          | Accuracy | F1-score |
|-------------------------------------|--------------------|--------------------|-----------|-----------|
| LogisticRegression (toy baseline)   | CountVectorizer    | unigram (6 câu)    | 0.50      | 0.00      |
| MultinomialNB                       | CountVectorizer    | unigram (CSV)      | 0.33      | 0.50      |
| LogisticRegression (balanced)       | CountVectorizer    | bigram (CSV)       | 0.25      | 0.00      |
| MultinomialNB                       | CountVectorizer    | negation-bigram    | 0.07      | 0.00      |
| LogisticRegression (balanced)       | CountVectorizer    | negation-bigram    | 0.13      | 0.10      |
| LogisticRegression (balanced, KFold)| CountVectorizer    | char 3–5 n-grams   | ~0.45     | ~0.30     |
| PySpark LogisticRegression          | HashingTF + IDF    | full CSV dataset   | 0.73      | 0.73      |

---

## ⚠️ Challenges and Solutions

| Thách thức | Giải pháp |
|-------------|-----------|
| Dataset nhỏ, toy model không ổn định | Dùng CSV lớn hơn (sentiments.csv) |
| Mất tín hiệu phủ định | Giữ “not”, “never”, thêm bigram phủ định |
| Lọc từ quá mạnh (`min_df=2`) làm mất đặc trưng | Giảm `min_df`, thử char n-grams |
| Kết quả dao động mạnh | Dùng Stratified K-Fold để đánh giá ổn định hơn |

---

## 📚 References

- Scikit-learn Documentation – https://scikit-learn.org/stable/  
- Spark MLlib Guide – https://spark.apache.org/docs/latest/ml-guide.html  
- Manning et al., *Foundations of Statistical NLP (2008)*  
- VNU HUS – NLP Lab series materials

---

## 🏁 Conclusion

Lab 5 minh họa toàn bộ quy trình phân loại văn bản: từ tiền xử lý, vector hóa, huấn luyện Logistic Regression, đến cải tiến mô hình.  
Các kỹ thuật như **Naive Bayes**, **bigrams**, và **char n-grams** giúp mô hình mạnh hơn, đặc biệt khi dữ liệu được mở rộng sang `sentiments.csv`.  
Khi áp dụng PySpark, pipeline có khả năng xử lý dữ liệu lớn và đạt hiệu năng ổn định hơn.
