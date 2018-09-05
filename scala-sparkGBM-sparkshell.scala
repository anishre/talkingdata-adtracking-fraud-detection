/*spark-shell --jars /home/krbipulesh/SparkGBM/target/SparkGBM-0.0.1.jar*/
/*spark-shell --jars /home/krbipulesh/SparkGBM/target/SparkGBM-0.0.1.jar --master yarn -i scala-sparkGBM.scala*/

import org.apache.spark.sql._
import org.apache.spark.sql.functions._
import org.apache.spark.ml.classification._
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator

val spark = SparkSession.builder.appName("GBMClassifierExample").getOrCreate()

spark.sparkContext.setLogLevel("INFO")

/*val files = List("train-split-aa.txt", 
                     "training_files_svm_light_2g_aa_ab.txt", 
                     "training_files_svm_light_3g_aa_ab_ac.txt",
                     "training_files_svm_light_4g_aa_ab_ac_ad.txt",
                     "training_files_svm_light_5g_aa_ab_ac_ad_ae.txt",
                     "training_files_svm_light_7g_all.txt")*/
val files = List("training_files_svm_light_5g_aa_ab_ac_ad_ae.txt")

for ( file <- files)
{
    println(s"=========USING FILE: ${file}===========")
    /*
    val train = spark.read.format("libsvm").load("data/a9a").select(((col("label") + 1) / 2).cast("int").as("label"), col("features")).repartition(64)
    val test = spark.read.format("libsvm").load("data/a9a.t").select(((col("label") + 1) / 2).cast("int").as("label"), col("features"))
    val data = spark.read.format("libsvm").load("sparkGBM_poc/svm_train").select(((col("label") + 1) / 2).cast("int").as("label"), col("features")).repartition(64)
    val data = spark.read.format("libsvm").load("/tmp/train-sample-2.txt/part*").select(((col("label") + 1) / 2).cast("int").as("label"), col("features")).repartition(64)
    */

    val inputPathTrain = s"gs://kb-advanced-bracketology/talkingdata-adtracking-fraud-detection/processed_files/training_files_libsvm/${file}"
    println(s"=========INPUT PATH: ${inputPathTrain}===========")

    //val data = spark.read.format("libsvm").load("gs://kb-advanced-bracketology/talkingdata-adtracking-fraud-detection/training_files_svm_light/train-split-ac-combined/*").select(((col("label") + 1) / 2).cast("int").as("label"), col("features")).repartition(64)
    val data = spark.read.format("libsvm").load(inputPathTrain).select(((col("label") + 1) / 2).cast("int").as("label"), col("features")).repartition(64)
    
    val splits = data.randomSplit(Array(0.8, 0.2), seed = 15L)
    val train = splits(0).cache()
    val test = splits(1).cache()
    
    val modelSavePath = s"/tmp/sparkGBM/spark-modelsave-${System.nanoTime}"
    
    val modelCheckpointPath = s"/tmp/sparkGBM/spark-modelcheckpoint-${System.nanoTime}"
    
    val gbmc = new GBMClassifier
    gbmc.setBoostType("gbtree").
      setBaseScore(0.0).
      setStepSize(0.2).
      setMaxIter(10).
      setMaxDepth(5).
      setMaxLeaves(1000).
      setMaxBins(128).
      setMinGain(0.0).
      setSubSample(0.9).
      setColSampleByTree(0.9).
      setColSampleByLevel(0.9).
      setRegAlpha(0.1).
      setRegLambda(1.0).
      setObjectiveFunc("logistic").
      setEvaluateFunc(Array("logloss", "auc", "error")).
      setCheckpointInterval(3).
      setEarlyStopIters(5).
      setModelCheckpointInterval(4).
      setModelCheckpointPath(modelCheckpointPath)
    
    /** train with validation */
    val model = gbmc.fit(train, test)
    
    println(s"model has ${model.numTrees} trees")
    
    /*
    /** model save and load */
    model.save(modelSavePath)
    val model2 = GBMClassificationModel.load(modelSavePath)
    
    /** load the model snapshots saved during training */
    val modelSnapshot4 = GBMClassificationModel.load(s"$modelCheckpointPath/model-4")
    println(s"modelSnapshot4 has ${modelSnapshot4.numTrees} trees")
    val modelSnapshot8 = GBMClassificationModel.load(s"$modelCheckpointPath/model-8")
    println(s"modelSnapshot8 has ${modelSnapshot8.numTrees} trees")
    
    */
    
    /** using only 5 tree for the following feature importance computation, prediction and leaf transformation */
    model.setFirstTrees(5)
    
    /** feature importance */
    println(s"featureImportances of first 5 trees ${model.featureImportances}")
    
    /** prediction with first 5 trees */
    val predictions = model.transform(test)
    predictions.show(10)
    
    /** path/leaf transformation with first 5 trees, one-hot encoded */
    val leaves = model.setEnableOneHot(true).leaf(test)
    leaves.show(10)
    
    
    /** model should have 10 trees, the first 10 trees have weight=0.2, the last 10 ones have weight=0.1 */
    println(s"model has ${model.numTrees} trees, with weights=${model.weights.mkString("(", ",", ")")}")
    
    val evaluator = new BinaryClassificationEvaluator()
    evaluator.setLabelCol("label").
      setRawPredictionCol("rawPrediction").
      setMetricName("areaUnderROC")
    
    /** auc of model with all trees */
    val auc = evaluator.evaluate(model.transform(test))
    println(s"AUC on test data $auc")
}


spark.stop()
