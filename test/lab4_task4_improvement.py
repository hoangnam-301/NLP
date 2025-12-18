import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, regexp_replace, lower
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF
from pyspark.ml.classification import LogisticRegression, NaiveBayes, GBTClassifier
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

def main():
    spark = SparkSession.builder.appName("Lab4_Task4_Improvement").master("local[*]").getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    data_path = "data/sentiments.csv"
    if not os.path.exists(data_path):
        print("Không tìm thấy file data/sentiments.csv")
        return

    # 1. Tải và TIỀN XỬ LÝ NÂNG CAO (Noise Filtering)
    df = spark.read.csv(data_path, header=True, inferSchema=True)
    
    # Làm sạch text: Bỏ link, bỏ mã chứng khoán ($SPY), bỏ số và ký tự đặc biệt
    clean_df = df.withColumn("text", regexp_replace(col("text"), r"http\S+", "")) \
                 .withColumn("text", regexp_replace(col("text"), r"\$\w+", "")) \
                 .withColumn("text", regexp_replace(col("text"), r"[^a-zA-Z\s]", "")) \
                 .withColumn("text", lower(col("text"))) \
                 .withColumn("label", when(col("sentiment") <= 0, 0.0).otherwise(1.0)) \
                 .dropna(subset=["text"])

    (train_data, test_data) = clean_df.randomSplit([0.8, 0.2], seed=42)

    # 2. Định nghĩa các Stages cho Pipeline
    tokenizer = Tokenizer(inputCol="text", outputCol="words")
    sw_remover = StopWordsRemover(inputCol="words", outputCol="filtered")
    
    # CẢI THIỆN: Tăng numFeatures lên 20,000 để giảm xung đột hash
    hashing_tf = HashingTF(inputCol="filtered", outputCol="raw_features", numFeatures=20000)
    idf = IDF(inputCol="raw_features", outputCol="features")

    # 3. Danh sách các mô hình để so sánh
    classifiers = {
        "Logistic Regression": LogisticRegression(maxIter=10, regParam=0.01, labelCol="label"),
        "Naive Bayes": NaiveBayes(labelCol="label"),
        "Gradient-Boosted Trees": GBTClassifier(maxIter=10, labelCol="label")
    }

    evaluator = MulticlassClassificationEvaluator(labelCol="label", metricName="accuracy")

    print(f"{'Algorithm':<25} | {'Accuracy':<10}")
    print("-" * 40)

    for name, clf in classifiers.items():
        # Tạo pipeline riêng cho từng model
        pipeline = Pipeline(stages=[tokenizer, sw_remover, hashing_tf, idf, clf])
        
        # Huấn luyện
        model = pipeline.fit(train_data)
        
        # Dự đoán
        predictions = model.transform(test_data)
        
        # Đánh giá
        accuracy = evaluator.evaluate(predictions)
        print(f"{name:<25} | {accuracy:.4f}")

    spark.stop()

if __name__ == "__main__":
    main()