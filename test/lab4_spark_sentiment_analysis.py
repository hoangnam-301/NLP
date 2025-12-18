import os
import sys
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

def main():
    # 1. Khởi tạo Spark Session
    spark = SparkSession.builder \
        .appName("Lab5_Spark_SentimentAnalysis") \
        .master("local[*]") \
        .getOrCreate()

    # Thiết lập mức log để tránh làm rối màn hình
    spark.sparkContext.setLogLevel("ERROR")

    # 2. Đường dẫn file dữ liệu của bạn
    data_path = "data/sentiments.csv"

    print(f"--- Loading data from: {data_path} ---")
    
    if not os.path.exists(data_path):
        print(f"Error: File not found at {data_path}. Please check the path.")
        spark.stop()
        return

    try:
        # 3. Đọc dữ liệu CSV
        # Giả sử file có cột 'text' và 'sentiment' (giá trị -1 và 1)
        df = spark.read.csv(data_path, header=True, inferSchema=True)

        # 4. Tiền xử lý nhãn (Label Normalization)
        # Chuyển sentiment: -1 -> 0 (Negative), 1 -> 1 (Positive)
        df = df.withColumn("label", when(col("sentiment") <= 0, 0.0).otherwise(1.0))
        
        # Loại bỏ các dòng trống để tránh lỗi Pipeline
        df = df.dropna(subset=["text", "sentiment"])

        # 5. Chia dữ liệu Train/Test (80/20)
        train_data, test_data = df.randomSplit([0.8, 0.2], seed=42)

        # 6. Xây dựng Pipeline các công đoạn
        # a. Tách từ
        tokenizer = Tokenizer(inputCol="text", outputCol="words")
        
        # b. Loại bỏ từ dừng (Stopwords)
        sw_remover = StopWordsRemover(inputCol="words", outputCol="filtered_words")
        
        # c. Vector hóa bằng HashingTF (Tạo vector tần suất từ)
        hashing_tf = HashingTF(inputCol="filtered_words", outputCol="raw_features", numFeatures=10000)
        
        # d. Tính toán IDF (Làm nổi bật các từ quan trọng)
        idf = IDF(inputCol="raw_features", outputCol="features")
        
        # e. Mô hình Logistic Regression
        lr = LogisticRegression(maxIter=10, regParam=0.001, featuresCol="features", labelCol="label")

        # Ghép tất cả thành một Pipeline thống nhất
        pipeline = Pipeline(stages=[tokenizer, sw_remover, hashing_tf, idf, lr])

        # 7. Huấn luyện mô hình
        print("Training Spark ML Pipeline on CSV data...")
        model = pipeline.fit(train_data)

        # 8. Dự đoán trên tập Test
        predictions = model.transform(test_data)

        # 9. Đánh giá kết quả
        evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
        accuracy = evaluator.evaluate(predictions)
        
        print("\n" + "="*30)
        print(f"Spark Model Accuracy: {accuracy:.4f}")
        print("="*30)

        # Hiển thị một số kết quả dự đoán mẫu
        print("\nSample Predictions:")
        predictions.select("text", "sentiment", "prediction").show(5)

    except Exception as e:
        print(f"An error occurred: {e}")
    
    finally:
        spark.stop()
        print("\nSpark Session stopped.")

if __name__ == "__main__":
    main()