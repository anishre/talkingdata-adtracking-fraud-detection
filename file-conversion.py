import pandas as pd
from sklearn.datasets import dump_svmlight_file

df=pd.read_csv("/home/krbipulesh/talkingdata/competition_files/train_sample.csv",header=0)
X=df[['ip', 'app', 'device', 'os', 'channel']]

#list of values
Y=df['is_attributed'].tolist() 
X.shape
Y.shape

dump_svmlight_file(X,Y,"/home/krbipulesh/talkingdata/competition_files/svm_light_files/svm_train",zero_based=False)




==========PYSPARK version 1==========

from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext
from sklearn.datasets import dump_svmlight_file
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, FloatType, TimestampType

#conf = SparkConf().setAppName("train-file-convert-v1")
#sc = SparkContext(conf=conf)
#sqlContext = SQLContext(sc)

#path = "gs://kb-advanced-bracketology/talkingdata-adtracking-fraud-detection/competition_files/train.csv"
#path = "gs://kb-advanced-bracketology/talkingdata-adtracking-fraud-detection/competition_files/train_sample.csv"
path = "gs://kb-advanced-bracketology/talkingdata-adtracking-fraud-detection/training_files_csv/train-split-aa"

trainschema = StructType([
    StructField("ip", IntegerType(), True),
    StructField("app", IntegerType(), True),
    StructField("device", IntegerType(), True),
    StructField("os", IntegerType(), True),
    StructField("channel", IntegerType(), True),
    StructField("click_time", TimestampType(), True),
    StructField("attributed_time", TimestampType(), True),
    StructField("is_attributed", IntegerType(), True)])

df = sqlContext.read.format("com.databricks.spark.csv").option("header", True).schema(trainschema).load(path)
df.cache()

pdf_X = df[['ip', 'app', 'device', 'os', 'channel']].toPandas()

#list of values
Y=df.select("is_attributed").rdd.flatMap(lambda x: x).collect()

#dump_svmlight_file(X_pandas,Y,"gs://kb-advanced-bracketology/talkingdata-adtracking-fraud-detection/svm_light_files/train_sample.txt")
dump_svmlight_file(X_pandas,Y,"/tmp/train-sample-1.txt")


==========PYSPARK version 2==========

from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext
from pyspark.mllib.util import MLUtils
from pyspark.mllib.regression import LabeledPoint

from sklearn.datasets import dump_svmlight_file
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, FloatType, TimestampType

conf = SparkConf().setAppName("train-file-convert-v2")
sc = SparkContext(conf=conf)
sqlContext = SQLContext(sc)

#path = "gs://kb-advanced-bracketology/talkingdata-adtracking-fraud-detection/competition_files/train.csv"
#path = "gs://kb-advanced-bracketology/talkingdata-adtracking-fraud-detection/competition_files/train_sample.csv"
path = "gs://kb-advanced-bracketology/talkingdata-adtracking-fraud-detection/competition_files/train_sample.csv"

trainschema = StructType([
    StructField("ip", IntegerType(), True),
    StructField("app", IntegerType(), True),
    StructField("device", IntegerType(), True),
    StructField("os", IntegerType(), True),
    StructField("channel", IntegerType(), True),
    StructField("click_time", TimestampType(), True),
    StructField("attributed_time", TimestampType(), True),
    StructField("is_attributed", IntegerType(), True)])

df = sqlContext.read.format("com.databricks.spark.csv").option("header", True).schema(trainschema).load(path)

rdd1 = df.rdd
print (rdd1.take(3))
#(ip=87540, app=12, device=1, os=13, channel=497, click_time=datetime.datetime(2017, 11, 7, 9, 30, 38), attributed_time=None, is_attributed=0)


rdd2 = rdd1.map(lambda line: LabeledPoint(line[7],[line[0],line[1],line[2],line[3],line[4]]))
print (rdd2.take(3))

MLUtils.saveAsLibSVMFile(rdd2, "gs://kb-advanced-bracketology/talkingdata-adtracking-fraud-detection/processed_files/training_files_libsvm/train_sample")



===============REF: https://stackoverflow.com/questions/43920111/convert-dataframe-to-libsvm-format =============

from pyspark.mllib.util import MLUtils
from pyspark.mllib.regression import LabeledPoint

# A DATAFRAME
>>> df.show()
+---+---+---+
| _1| _2| _3|
+---+---+---+
|  1|  3|  6|  
|  4|  5| 20|
|  7|  8|  8|
+---+---+---+

# FROM DATAFRAME TO RDD
>>> c = df.rdd # this command will convert your dataframe in a RDD
>>> print (c.take(3))
[Row(_1=1, _2=3, _3=6), Row(_1=4, _2=5, _3=20), Row(_1=7, _2=8, _3=8)]

# FROM RDD OF TUPLE TO A RDD OF LABELEDPOINT
>>> d = c.map(lambda line: LabeledPoint(line[0],[line[1:]])) # arbitrary mapping, it's just an example
>>> print (d.take(3))
[LabeledPoint(1.0, [3.0,6.0]), LabeledPoint(4.0, [5.0,20.0]), LabeledPoint(7.0, [8.0,8.0])]

# SAVE AS LIBSVM
>>> MLUtils.saveAsLibSVMFile(d, "/your/Path/nameFolder/")