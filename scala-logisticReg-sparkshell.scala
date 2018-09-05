import org.apache.spark.mllib.classification.{LogisticRegressionModel, LogisticRegressionWithLBFGS}
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.util.MLUtils

/*val files = List("train-split-aa.txt", 
                     "training_files_svm_light_2g_aa_ab.txt", 
                     "training_files_svm_light_3g_aa_ab_ac.txt",
                     "training_files_svm_light_4g_aa_ab_ac_ad.txt",
                     "training_files_svm_light_5g_aa_ab_ac_ad_ae.txt",
                     "training_files_svm_light_7g_all.txt")*/

// Load training data in LIBSVM format.
val data = MLUtils.loadLibSVMFile(sc, "gs://kb-advanced-bracketology/talkingdata-adtracking-fraud-detection/processed_files/training_files_libsvm/train-split-aa.txt")

// Split data into training (80%) and test (20%).
val splits = data.randomSplit(Array(0.8, 0.2), seed = 11L)
val training = splits(0).cache()
val test = splits(1)

// Run training algorithm to build the model
val model = new LogisticRegressionWithLBFGS().
  setNumClasses(2).
  run(training)

// Compute raw scores on the test set.
val predictionAndLabels = test.map { case LabeledPoint(label, features) =>
  val prediction = model.predict(features)
  (prediction, label)
}


// Get evaluation metrics.
val metrics = new BinaryClassificationMetrics(predictionAndLabels)
val auROC = metrics.areaUnderROC()

println("Area under ROC = " + auROC)


/*// Get evaluation metrics.
val metrics = new MulticlassMetrics(predictionAndLabels)
val accuracy = metrics.accuracy
println(s"Accuracy = $accuracy")*/

// Save and load model
model.save(sc, "gs://kb-advanced-bracketology/savedModels/scalaLogisticRegressionWithLBFGSModel")
//val sameModel = LogisticRegressionModel.load(sc,"gs://kb-advanced-bracketology/savedModels/scalaLogisticRegressionWithLBFGSModel")