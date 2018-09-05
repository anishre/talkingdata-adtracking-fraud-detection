
import ml.dmlc.xgboost4j.scala.Booster
import ml.dmlc.xgboost4j.scala.spark.XGBoost
import org.apache.spark.sql.SparkSession
import org.apache.spark.SparkConf
import org.apache.spark.sql.types._
import org.apache.spark.ml.feature.VectorAssembler

// create SparkSession
val sparkConf = new SparkConf().setAppName("XGBoost-spark-example").set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")

sparkConf.registerKryoClasses(Array(classOf[Booster]))

// val sqlContext = new SQLContext(new SparkContext(sparkConf))

val sparkSession = SparkSession.builder().config(sparkConf).getOrCreate()


// create training and testing dataframes
// val numRound = args(0).toInt
// val inputTrainPath = args(2)
// val inputTestPath = args(3)

val numRound = 2
val inputTrainPath = "gs://kb-advanced-bracketology/talkingdata-adtracking-fraud-detection/competition_files/train_sample.csv"
val inputTestPath = "gs://kb-advanced-bracketology/talkingdata-adtracking-fraud-detection/competition_files/test.csv"

val trainschema = StructType(Array(
    StructField("ip", IntegerType, true),
    StructField("app", IntegerType, true),
    StructField("device", IntegerType, true),
    StructField("os", IntegerType, true),
    StructField("channel", IntegerType, true),
    StructField("click_time", TimestampType, true),
    StructField("attributed_time", TimestampType, true),
    StructField("is_attributed", IntegerType, true)))

val testschema = StructType(Array(
    StructField("click_id", IntegerType, true),
    StructField("ip", IntegerType, true),
    StructField("app", IntegerType, true),
    StructField("device", IntegerType, true),
    StructField("os", IntegerType, true),
    StructField("channel", IntegerType, true),
    StructField("click_time", TimestampType, true)))


// build dataset
val trainDF = sparkSession.sqlContext.read.format("csv").option("header", true).schema(trainschema).load(inputTrainPath)
val testDF = sparkSession.sqlContext.read.format("csv").option("header", true).schema(testschema).load(inputTestPath)

// start training
val paramMap = List(
  "eta" -> 0.1f,
  "max_depth" -> 2,
  "objective" -> "binary:logistic").toMap


val assembler =  new VectorAssembler().setInputCols(Array("ip", "app", "device", "os", "channel")).setOutputCol("features")
val vectedtrainDF = assembler.transform(trainDF).withColumnRenamed("is_attributed", "label").drop("ip", "app", "device", "os", "channel", "click_time", "attributed_time")
val vectedtestDF = assembler.transform(testDF).drop("click_id","ip", "app", "device", "os", "channel", "click_time")

val xgboostModel = XGBoost.trainWithDataFrame(
      vectedtrainDF, paramMap, numRound, nWorkers = 2, useExternalMemory = true)


// xgboost-spark appends the column containing prediction results
xgboostModel.transform(vectedtestDF).show()

// predict use 1 tree
val predicts1 = xgboostModel.predict(vectedtestDF, false, 1)

// by default all trees are used to do predict
val predicts2 = xgboostModel.predict(testMat)

val eval = new CustomEval
    println("error of predicts1: " + eval.eval(predicts1, testMat))
println("error of predicts2: " + eval.eval(predicts2, testMat))