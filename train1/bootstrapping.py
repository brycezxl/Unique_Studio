import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction import DictVectorizer

# 超参
EPOCH = 3000
LAMBDA = 0
AlPHA = 0.01
ITER = 10


def load_train():
    """Load titanic train data and process into vectorized martix.

    Returns:
        x_get: matrix, data
        y_get: matrix, label

    """
    data_frame = pd.read_csv('./titanic/train.csv')

    # 填补空缺值
    data_frame = data_frame.fillna(method='ffill')  # 待改进

    # 删除不必要特征
    del data_frame['PassengerId']
    del data_frame['Cabin']
    del data_frame['Name']
    del data_frame['Ticket']

    # 特征向量化
    dvec = DictVectorizer(sparse=False)
    data_get = dvec.fit_transform(data_frame.to_dict(orient='record'))

    x_get = data_get[:, :-2]
    y_get = data_get[:, -1]
    return x_get, y_get


def load_test():
    """Load titanic test data and process into vectorized martix.

    Returns:
        x_get: matrix, data
        y_get: matrix, label

    """
    x_frame = pd.read_csv('./titanic/test.csv')
    y_frame = pd.read_csv('./titanic/gender_submission.csv')

    # 填补空缺值
    x_frame = x_frame.fillna(method='ffill')  # 待改进

    # 删除不必要特征
    del x_frame['PassengerId']
    del x_frame['Cabin']
    del x_frame['Name']
    del x_frame['Ticket']

    # 特征向量化
    dvec = DictVectorizer(sparse=False)
    data_get = dvec.fit_transform(x_frame.to_dict(orient='record'))
    x_get = data_get[:, :-1]
    y_get = y_frame.values[:, 1]

    return x_get, y_get


class LogisticRegression(object):
    """Do logistic regression to predict 0 or 1.

    Contains normalize, forward prop, cost function, back prop, and plot j

    Attribute:
        images: input data to predict
        labels: answer
        rows: rows of images
        cols: columns of images
        j: cost
        j_list: store j for visualize
        theta: weight of each input
        predict: the predict of the algorithm
    """

    def __init__(self, images, labels):
        self.images = images
        self.labels = labels
        self.rows, self.cols = np.shape(images)
        self.j = 999
        self.j_list = []
        self.theta = np.random.random(((self.cols + 1), 1))
        self.predict = np.zeros((self.rows, 1))

    def normalize(self):
        """turn images into numbers near 0
        """

        # reshape data
        self.images = np.hstack((np.ones((self.rows, 1)), self.images))  # 加一列1
        self.labels = self.labels.reshape((self.rows, 1))

        # norm
        average = np.mean(self.images, axis=0)
        std = np.std(self.images, axis=0)
        for m in range(self.cols):
            if std[m] == 0:
                continue
            else:
                self.images[:, m] = (self.images[:, m] - average[m]) / std[m]
        return 0

    def cost_function(self):
        """calculate cost j for visualization"""
        self.j = -1 / self.rows * (np.sum((self.labels * np.log(self.predict)), axis=0)
                                   + np.sum(((1 - self.labels) * np.log(1 - self.predict)), axis=0))
        self.j_list.append(self.j)
        return 0

    def back(self):
        """do backprop to adjust theta"""
        grad = AlPHA / self.rows * np.dot(self.images.T, (self.predict - self.labels))
        self.theta = self.theta - grad
        return 0

    def plot_j(self):
        """Visualize the decline of j"""
        plt.scatter(range(EPOCH), self.j_list, linewidths=0.01, c="r")
        plt.show()
        return 0

    def forward(self):
        """forward prop with sigmoid"""
        self.predict = 1 / (1 + np.exp(-np.dot(self.images, self.theta)))
        return 0


def acc_cross():
    """计算acc"""

    count = 0
    predict_copy2 = np.zeros((logistic_regression_test.rows, 1))

    for j in range(logistic_regression_test.rows):

        # 处理predict
        if logistic_regression_test.predict[j] < 0.5:
            predict_copy2[j] = 0
        else:
            predict_copy2[j] = 1
        if predict_copy2[j] == label_boot[j]:
            count = count + 1

    accu = count / logistic_regression.rows * 100
    return accu


# 自助法
x, y = load_train()
acc_add = 0

for i in range(ITER):

    data_boot = np.zeros_like(x)
    label_boot = np.zeros_like(y)

    # 随机取出
    for k in range(np.size(x, axis=0)):
        random_int = np.random.randint(low=0, high=(np.size(x, axis=0) - 1))
        label_boot[k] = y[random_int]
        data_boot[k] = x[random_int, :]

    # train
    logistic_regression = LogisticRegression(x, y)
    logistic_regression.normalize()

    for epoch in range(EPOCH):
        logistic_regression.forward()
        logistic_regression.cost_function()
        logistic_regression.back()

    logistic_regression_test = LogisticRegression(data_boot, label_boot)
    logistic_regression_test.normalize()
    logistic_regression_test.forward()
    accu_out = acc_cross()
    acc_add = acc_add + accu_out

print("ACC: %.2f%%" % (acc_add / ITER))