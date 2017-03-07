# Ejecutarse con
# ./bin/spark-submit --master spark://dhcp014.aic.uniovi.es:7077 /Users/albertocastano/development/quantification/examples/planton_spark.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from pyspark.sql import SparkSession

spark = SparkSession\
    .builder\
    .appName("LR Plancton")\
    .getOrCreate()
spark.sparkContext.setLogLevel('ERROR')
path = '/Users/albertocastano/Desktop/plancton.csv'
pd_df = pd.read_csv(path, delimiter=' ')
pd_df.drop(["Sample"], axis=1, inplace=True)
le = LabelEncoder()
pd_df['class'] = le.fit_transform(pd_df['class'])

raw_data_df = spark.createDataFrame(pd_df)


from pyspark.mllib.regression import LabeledPoint
def parse_points(df):
    def parse_point(row):
        feats = row[:-1]
        if row[-1] == 3:
            cls = 1.0
        else:
            cls = 0.0
        return LabeledPoint(cls, feats)
    df2 = (df.rdd
           .map(lambda row: parse_point(row)))
    return df2


parsed_data_df = parse_points(raw_data_df)
first_point_features = parsed_data_df.first().features
first_point_label = parsed_data_df.first().label
print first_point_features, first_point_label

d = len(first_point_features)
print d

weights = [.8, .1, .1]
seed = 42
parsed_train_data_df, parsed_val_data_df, parsed_test_data_df = parsed_data_df.randomSplit(weights, seed)
parsed_train_data_df.cache()
parsed_val_data_df.cache()
parsed_test_data_df.cache()
n_train = parsed_train_data_df.count()
n_val = parsed_val_data_df.count()
n_test = parsed_test_data_df.count()

print n_train, n_val, n_test, n_train + n_val + n_test
print parsed_data_df.count()



from pyspark.mllib.classification import LogisticRegressionWithLBFGS

num_iters = 500  # iterations
reg = 1e-1  # regParam
use_intercept = True  # intercept

model = LogisticRegressionWithLBFGS.train(parsed_train_data_df, iterations=num_iters, regParam=reg, intercept=use_intercept)
labelsAndPreds = parsed_test_data_df.map(lambda p: (p.label, model.predict(p.features)))
trainErr = labelsAndPreds.filter(lambda (v, p): v == p).count() / float(parsed_test_data_df.count())
print("Testing Accuracy = " + str(trainErr))



spark.stop()