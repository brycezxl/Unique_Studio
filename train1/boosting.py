import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from sklearn.feature_extraction import DictVectorizer

# Adaboost

# 超参
EPOCH = 3000
AlPHA = 0.01


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
        for i in range(self.cols):
            if std[i] == 0:
                continue
            else:
                self.images[:, i] = (self.images[:, i] - average[i]) / std[i]
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


def acc():
    """计算acc """

    count = 0
    predict_copy1 = np.zeros((logistic_regression_test.rows, 1))

    for j in range(logistic_regression_test.rows):

        # 处理predict
        if logistic_regression_test.predict[j] < 0.5:
            predict_copy1[j] = 0
        else:
            predict_copy1[j] = 1
        if predict_copy1[j] == y_test[j]:
            count = count + 1

    accu = count / logistic_regression_test.rows * 100
    print("acc:            %.4f%%" % accu)
    return 0


# load
x, y = load_train()
row, col = np.shape(x)

# 初始化权重
D = np.array([1 / row] * row)

for i in range(EPOCH):
    




logistic_regression = LogisticRegression(x, y)
logistic_regression.normalize()

pbar = tqdm(total=EPOCH)

for epoch in range(EPOCH):
    logistic_regression.forward()
    logistic_regression.cost_function()
    logistic_regression.back()
    if epoch % 50 == 0:
        pbar.update(50)

pbar.close()
logistic_regression.plot_j()
print("Minimized cost: %.5f" % logistic_regression.j)


# test
x_test, y_test = load_test()
logistic_regression_test = LogisticRegression(x_test, y_test)
logistic_regression_test.normalize()
logistic_regression_test.forward()


# 评估
acc()
