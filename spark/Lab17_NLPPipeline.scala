package com.harito.spark

import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.{RegexTokenizer, StopWordsRemover, Word2Vec} 
import org.apache.spark.ml.linalg.Vector 
import org.apache.spark.sql.functions._
import org.apache.spark.sql.{functions, Row}

import java.time.LocalDateTime
import java.time.temporal.ChronoUnit
import java.io.{File, PrintWriter}

object Lab17_NLPPipeline {
  
  // Cấu hình (Update Request 1)
  val inputPath = "../data/c4-train.00000-of-01024-30K.json.gz"
  // BỎ QUA outputPath và logPath để đơn giản hóa
  val limitDocuments = 1000 

  // --- Hàm Tìm Độ Tương Đồng (Dùng Word2Vec) ---
  // Hàm này vẫn giữ nguyên, nhưng sẽ in kết quả trực tiếp ra console
  def findSimilarDocuments(spark: SparkSession, transformedDF: org.apache.spark.sql.DataFrame): Unit = {
    import spark.implicits._
    
    if (transformedDF.isEmpty) return

    val queryDocument: Row = transformedDF.first()
    val queryId = queryDocument.getAs[Long]("id")
    val queryVector = queryDocument.getAs[Vector]("word_vectors").toDense 
    val queryText = queryDocument.getAs[String]("text")

    println(s"\n--- Top 5 Similar Documents (Word2Vec Cosine Similarity) ---")
    println(s"Query Document (ID: $queryId):\nText: ${queryText.take(100)}...")

    val calculateSimilarity = udf((vectorB: Vector) => {
      queryVector.dot(vectorB.toDense)
    })

    val similarityDF = transformedDF
      .filter($"id" =!= queryId) 
      .withColumn("similarity", calculateSimilarity(col("word_vectors"))) 
      .select("id", "text", "similarity")
      .orderBy(col("similarity").desc)

    println("Top 5 Most Similar:")
    similarityDF.take(5).zipWithIndex.foreach { case (row, index) =>
      println(s"${index + 1}. ID: ${row.getLong(0)}, Similarity: ${"%.4f".format(row.getDouble(2))} \n   Text: ${row.getString(1).take(100)}...")
    }
  }

  def main(args: Array[String]): Unit = {
    val startTime = LocalDateTime.now()
    // Loại bỏ logWriter và logPath để tránh lỗi ghi file
    // Sử dụng System.currentTimeMillis() để đo thời gian đơn giản hơn trong trường hợp này
    
    val spark = SparkSession.builder
      .appName("NLP Pipeline Example - Exercise 4 (Console Output)")
      .master("local[*]")
      .getOrCreate()

    import spark.implicits._
    println("Spark Session created successfully.")
    
    // 1. --- Read Dataset ---
    val readStartTime = System.currentTimeMillis()
    var initialDF = spark.read.json(inputPath)
      .withColumn("id", monotonically_increasing_id()) 
      .limit(limitDocuments) 

    var data = initialDF.select("id", "text").cache()
    val readDuration = System.currentTimeMillis() - readStartTime
    println(s"Time to read data: ${readDuration}ms") 

    // 2. --- Define Pipeline Stages ---
    val tokenizer = new RegexTokenizer()
      .setInputCol("text")
      .setOutputCol("tokens")
      .setPattern("\\s+|[.,;!?()\"']") 

    val stopWordsRemover = new StopWordsRemover()
      .setInputCol(tokenizer.getOutputCol)
      .setOutputCol("filtered_tokens")

    // ĐỊNH NGHĨA Word2Vec (EXERCISE 4)
    val word2Vec = new Word2Vec()
      .setInputCol(stopWordsRemover.getOutputCol)
      .setOutputCol("word_vectors")
      .setVectorSize(100)           
      .setMinCount(5)               

    val pipeline = new Pipeline()
      .setStages(Array(tokenizer, stopWordsRemover, word2Vec)) 

    // --- Time the main operations ---
    val fitStartTime = System.nanoTime()
    val pipelineModel = pipeline.fit(data) 
    val fitDuration = (System.nanoTime() - fitStartTime) / 1e9d
    println(f"--> Pipeline fitting (Word2Vec) took $fitDuration%.2f seconds.")

    val transformStartTime = System.nanoTime()
    val transformedData = pipelineModel.transform(data).cache() 
    val transformCount = transformedData.count() 
    val transformDuration = (System.nanoTime() - transformStartTime) / 1e9d
    println(f"--> Data transformation took $transformDuration%.2f seconds.")

    // Log Actual Vocab Size
    val actualVocabSize = transformedData
      .select(explode($"filtered_tokens").as("word"))
      .filter(length($"word") > 1) 
      .distinct()
      .count()
    println(s"Actual vocabulary size: $actualVocabSize unique terms.")
    
    // 3. FIND SIMILAR DOCUMENTS (IN RA CONSOLE)
    findSimilarDocuments(spark, transformedData) 

    // 4. SAVE RESULTS (IN RA CONSOLE - THAY CHO saveAsTextFile GÂY LỖI)
    println("\n--- Sample Word2Vec Vectors (First 20 Results) ---")
    transformedData.select("id", "text", "word_vectors")
      .take(20)
      .foreach { row =>
        val vector = row.getAs[Vector](2)
        println(s"ID: ${row.getLong(0)}")
        println(s"Text: ${row.getString(1).take(100)}...")
        println(s"Word2Vec Vector (Dim=${vector.size}): ${vector.toString.take(100)}...")
        println("-" * 50)
      }
    
    // Final Cleanup
    spark.stop()
    println("Spark Session stopped.")
  }
}