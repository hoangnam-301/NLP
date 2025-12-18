import sys
import os
from pyspark.sql import SparkSession
from pyspark.ml.feature import Tokenizer, Word2Vec
from pyspark.sql.functions import col, lower, regexp_replace

def main():
    # 1. Khoi tao Spark Session
    spark = SparkSession.builder \
        .appName("Lab4_Spark_Word2Vec") \
        .master("local[*]") \
        .getOrCreate()

    # Thiet lap log level de do bi roi mat
    spark.sparkContext.setLogLevel("ERROR")

    # 2. Duong dan file cua ban (Da them .gz)
    data_path = "data/c4-train.00000-of-01024-30K.json.gz"

    print(f"Checking file at: {data_path}")
    if not os.path.exists(data_path):
        print(f"Error: File not found at {data_path}. Please check your folder structure.")
        spark.stop()
        return

    try:
        # 3. Doc du lieu JSON
        print("Loading dataset...")
        raw_df = spark.read.json(data_path)
        
        # 4. Tien xu ly
        # Chon cot 'text', chuyen thanh chu thuong va loai bo ky tu dac biet
        clean_df = raw_df.select(
            lower(regexp_replace(col("text"), r"[^a-zA-Z\s]", "")).alias("text")
        )

        # 5. Tokenization (Cat cau thanh mang cac tu)
        tokenizer = Tokenizer(inputCol="text", outputCol="words")
        words_df = tokenizer.transform(clean_df)

        # 6. Cau hinh va huan luyen Word2Vec
        # vectorSize=100: moi tu la mot vector 100 chieu
        # minCount=5: tu phai xuat hien it nhat 5 lan moi duoc hoc
        word2vec = Word2Vec(vectorSize=100, minCount=5, inputCol="words", outputCol="result")
        
        print("Training Word2Vec model on C4 dataset (This may take a minute)...")
        model = word2vec.fit(words_df)

        # 7. Tim cac tu tuong dong voi 'computer'
        print("\nTop 5 synonyms for 'computer' based on C4 dataset:")
        synonyms = model.findSynonyms("computer", 5)
        synonyms.show()

    except Exception as e:
        print(f"An error occurred during processing: {e}")
    
    finally:
        # Dong Spark
        spark.stop()

if __name__ == "__main__":
    main()