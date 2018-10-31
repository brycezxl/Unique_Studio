import csv
import os
import struct
import numpy as np
import matplotlib.pyplot as pltdef
import pandas as pd
from sklearn.feature_extraction import DictVectorizer


def load(kind='train'):

    # load whole data
    csv_data = csv.reader(open(r'./titanic/%s.csv' % kind, 'r'))
    data_get = []
    for stu in csv_data:
        data_get.extend(stu)
    data_in = np.reshape(np.array(data_get, dtype=np.str), (int(len(data_get) / 12), 12))


    # delete some features, divide x and y
    data_in = data_in[1:, :]
    y_in = data_in[:, 1]
    y_in = y_in.astype(np.int8)
    x_in = data_in[:, (2, 4, 5, 6, 7, 9, 11)]

    df = pd.DataFrame(x_in)
    df = df.fillna(method='fill')
    print(df)
    #
    return x_in, y_in

def test():
    # x, y = load()
    # # print(np.shape(x), y)

    data_frame = pd.read_csv('./titanic/train.csv')

    data_frame = data_frame.fillna(method='ffill')    # 待改进

    del data_frame['PassengerId']
    del data_frame['Cabin']
    del data_frame['Name']
    del data_frame['Ticket']

    dvec = DictVectorizer(sparse=False)

    data_get = dvec.fit_transform(data_frame.to_dict(orient='record'))
    x_get = data_get[:, :-2]
    y_get = data_get[:, -1]


a = np.array([1, 2, 3, 4])
# a = a.reshape((2, 2))
b = np.array([1, 2, 3, 4])
b = b.reshape((2, 2))
print(np.sum(a, axis=0))
