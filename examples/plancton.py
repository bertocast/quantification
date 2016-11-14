# coding=utf-8
import pandas as pd
from sklearn.datasets.base import Bunch
from sklearn.preprocessing import LabelEncoder

from quantification.classify_and_count import ClassifyAndCount
from quantification.classify_and_count import MulticlassAdjustedCount


def load_plankton_file(path, sample_col="Sample", target_col="class"):

    data_file = pd.read_csv(path, delimiter=' ')
    le = LabelEncoder()
    data_file[target_col] = le.fit_transform(data_file[target_col])
    data = data_file.groupby(sample_col)
    target = [sample[1].values for sample in data[target_col]]
    features = [sample[1].drop([sample_col, target_col], axis=1, inplace=False).values for sample in data]
    return Bunch(data=features, target=target,
                 target_names=le.classes_), le


if __name__ == '__main__':
    plankton,le = load_plankton_file('/Users/albertocastano/Dropbox/PlataformaCuantificación/plancton.csv')
    cc = MulticlassAdjustedCount()
    X = plankton.data[0]
    y = plankton.target[0]
    cc.fit(X, y, local=True)
    print "Fitted"
    predictions = cc.predict(X, local=True)
    for pr in predictions:
        print ["{0:0.2f}".format(i) for i in pr]