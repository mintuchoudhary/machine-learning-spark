package com.m2.ml.lr

import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types.{DoubleType, IntegerType, StructField, StructType}

/**
 * Default format : libsvm  (Binomial logistic regression) which can be directly used within API
 * looks like : <label> <index1>:<value1> <index2>:<value2> ... <indexN>:<valueN>
 * ref : https://www.analyticsvidhya.com/blog/2022/08/complete-guide-to-run-machine-learning-on-spark-using-spark-mllib/
 */
object LinearRegression {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder().appName("Linear")
      .config("spark.testing.memory", "2147480000")
      .config("spark.driver.memory", "2147480000")
      .master("local").getOrCreate()
    val dataDF = spark.read.option("header", true).schema(StructType(Seq(StructField("Time_to_Study", IntegerType), StructField("Grades", DoubleType)))).csv("src/test/resources/")

    dataDF.show(false)
    println("getfirstcol:", dataDF.columns.take(1).toString)
    dataDF.printSchema()
    //input col = Time_to_Study
    val vectorAssmblr = new VectorAssembler().setInputCols(dataDF.columns.take(1)).setOutputCol("features")
    val dataFeatureDF = vectorAssmblr.transform(dataDF)
    //    dataFeatureDF.show(false)

    val finalizedData = dataFeatureDF.select("features", "Grades").withColumnRenamed("grades", "label")
    finalizedData.show()

    val Array(trainingData, testData) = finalizedData.randomSplit(Array(0.7, 0.3))
    //    println("trcount ",trainingData.count())
    //    println("tscount ",testData.count())

    //See statistic of training data
    trainingData.describe().show()

    val lr = new LinearRegression()
    //      .setMaxIter(10)
    //      .setRegParam(0.3)
    //      .setElasticNetParam(0.8)

    val model = lr.fit(trainingData)
    val predictionSummary = model.evaluate(testData)
    predictionSummary.predictions.show(false)
  }
}
