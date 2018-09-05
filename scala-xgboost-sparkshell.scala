
#spark-shell --jars /home/krbipulesh/xgboost/jvm-packages/xgboost4j/target/xgboost4j-0.72-jar-with-dependencies.jar,/home/krbipulesh/xgboost/jvm-packages/xgboost4j-spark/target/xgboost4j-spark-0.72-jar-with-dependencies.jar --master yarn




//package ml.dmlc.xgboost4j.scala.example.spark

import ml.dmlc.xgboost4j.scala.Booster
import ml.dmlc.xgboost4j.scala.spark.XGBoost
import org.apache.spark.sql.SparkSession
import org.apache.spark.SparkConf
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.ml.linalg.DenseVector
import org.apache.spark.sql.Row
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics

// create SparkSession
val sparkConf = new SparkConf().setAppName("XGBoost-spark-example").set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
sparkConf.registerKryoClasses(Array(classOf[Booster]))

// val sqlContext = new SQLContext(new SparkContext(sparkConf))
val sparkSession = SparkSession.builder().config(sparkConf).getOrCreate()

/*val files = List("train-split-aa.txt", 
                     "training_files_svm_light_2g_aa_ab.txt", 
                     "training_files_svm_light_3g_aa_ab_ac.txt",
                     "training_files_svm_light_4g_aa_ab_ac_ad.txt",
                     "training_files_svm_light_5g_aa_ab_ac_ad_ae.txt",
                     "training_files_svm_light_7g_all.txt")*/


val inputTrainPath = "gs://kb-advanced-bracketology/talkingdata-adtracking-fraud-detection/processed_files/training_files_libsvm/training_files_svm_light_5g_aa_ab_ac_ad_ae.txt"

// Load training data in LIBSVM format.
val data = sparkSession.sqlContext.read.format("libsvm").load(inputTrainPath)


// Split data into training (80%) and test (20%).
val splits = data.randomSplit(Array(0.8, 0.2), seed = 11L)
val trainDF = splits(0).cache()
val testDF = splits(1)

// create training and testing dataframes
val numRound = 2

// start training
val paramMap = List(
  "eta" -> 0.1f,
  "max_depth" -> 2,
  "objective" -> "binary:logistic").toMap


val xgboostModel = XGBoost.trainWithDataFrame(
  trainDF, paramMap, numRound, nWorkers = 3.toInt, useExternalMemory = true)


// xgboost-spark appends the column containing prediction results


//=============
val toArr: Any => Array[Double] = _.asInstanceOf[DenseVector].toArray
val toArrUdf = udf(toArr)

val prob_pred = xgboostModel.transform(testDF).select("probabilities","label")
val prob_arr_pred = prob_pred.withColumn("probabilities_arr",toArrUdf('probabilities)).drop("probabilities")

//https://stackoverflow.com/questions/43731181/how-can-i-extract-value-from-array-in-a-column-of-spark-dataframe
val predictionAndLabels = prob_arr_pred.withColumn("prob", $"probabilities_arr".getItem(1).cast("double")).drop("probabilities_arr").rdd

val rdd_predictionAndLabels = predictionAndLabels.map{case Row(label:Double, prediction: Double) => (prediction, label)}

// Get evaluation metrics.
val metrics = new BinaryClassificationMetrics(rdd_predictionAndLabels)
val auROC = metrics.areaUnderROC()

println("Area under ROC = " + auROC)

//=======================






