package com.m2.ml.lr

import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.{DoubleType, IntegerType, StructField, StructType}

object LogisticRegression {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder().appName("Linear")
      .config("spark.testing.memory", "2147480000")
      .config("spark.driver.memory", "2147480000")
      .master("local").getOrCreate()
    val dataDF = spark.read.option("header", true).schema(StructType(List(StructField("score1", DoubleType), StructField("score2", DoubleType), StructField("result", IntegerType))))
      .csv("src/test/resources/scores.csv")
    //    dataDF.show(false)

    //    dataDF.printSchema()
    dataDF.describe().show(false)

    // columns that need to added to feature column
    val cols = Array("score1", "score2")

    // VectorAssembler to add feature column
    // input columns - cols
    // feature column - features
    val assembler = new VectorAssembler()
      .setInputCols(cols)
      .setOutputCol("features")
    val featureDf = assembler.transform(dataDF)
    featureDf.printSchema()
    featureDf.show(5, false)
    val featureLabelDF = featureDf.withColumn("label", col("result").cast(DoubleType))

    val Array(trainingData, testData) = featureLabelDF.randomSplit(Array(0.7, 0.3),seed = 5043)
    println("trcount=",trainingData.count())
    println("tscount=",testData.count())

    val logisticRegression = new LogisticRegression()//.setMaxIter(100).setRegParam(0.02).setElasticNetParam(0.8)
    val logisticRegressionModel = logisticRegression.fit(trainingData)
    // run model with test data set to get predictions
    // this will add new columns rawPrediction, probability and prediction
    val predictionDF = logisticRegressionModel.transform(testData)
    predictionDF.show(false)

    // evaluate model with area under ROC
    val evaluator = new BinaryClassificationEvaluator()
      .setLabelCol("label")
      .setRawPredictionCol("prediction")
      .setMetricName("areaUnderROC")

    // measure the accuracy
    val accuracy = evaluator.evaluate(predictionDF)
    println(accuracy)

  }
}
