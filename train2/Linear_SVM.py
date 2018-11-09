import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
# from tqdm import tqdm


# 超参
EPOCH = 6
LAMBDA = 0
AlPHA = 0.001
SCALE = 0.5


def leave_out():
    """Divide data into 70% for train and 30% for test.

    Returns:
        x_get: matrix, data
        y_get: matrix, label
        x_get_train: matrix, data
        y_get_train: matrix, label

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

    leave_num = int((np.size(data_get, axis=0) - (np.size(data_get, axis=0) % 10)) / 10 * 7)
    x_get = data_get[:leave_num, :-2]
    y_get = data_get[:leave_num, -1]
    x_test_get = data_get[(leave_num + 1):, :-2]
    y_test_get = data_get[(leave_num + 1):, -1]
    return x_get, y_get, x_test_get, y_test_get


class LinearSVM(object):
    """Do logistic regression.

    Methods:　fit     -- Train the model with given x and y,
　　　　　　　　score   -- Show the mean accuracy on the given test data and labels

    Attribute:
        __params: parameters
        __count: decide when to initialize params

    """
    def __init__(self):
        self.__params = 0
        self.__count = 0

    def __get_params(self):
        """Get parameters for this estimator."""
        params = self.__params
        return params

    def __set_params(self, params_in):
        """Set parameters for this estimator."""
        self.__params = params_in
        return 0

    def fit(self, x_in, y_in):
        """Fit the model according to the given training data."""
        # pbar = tqdm(total=EPOCH)

        x_in, y_in = self.__normalize(x_in, y_in)

        # initialize the list to store cost j
        j_list = []

        # fit
        for epoch in range(EPOCH):
            for i in range(np.size(x_in, axis=0)):
                self.__back(x_in[i, :], y_in[i, ])
                # if epoch % 50 == 0:
                #     # pbar.update(50)
                #     print("EPOCH: %4d" % epoch, " | Cost: ", j_list[-1])
            j_list = self.__cost_function(x_in, y_in, j_list)
        # # pbar.close()

        # plot
        self.__plot_j(j_list)
        for j in range(np.size(y_in, axis=0)):
            if y_in[j] == -1:
                y_in[j] = 0
        print("Minimized cost: %.5f" % j_list[-1])
        return self

    def score(self, x_score, y_score):
        """Returns the mean accuracy on the given test data and labels."""

        x_score, y_score = self.__normalize(x_score, y_score)
        y_pred = np.dot(x_score, self.__get_params())

        # record the right classify
        count = 0

        # turn into 0/1, then judge
        for j in range(np.size(y_score, axis=0)):
            if y_pred[j] >= SCALE:
                y_pred[j] = 1
            else:
                y_pred[j] = 0
            if y_pred[j] == y_score[j]:
                count = count + 1
        accuracy = count / np.size(y_score, axis=0) * 100
        print("acc:            %.4f%%" % accuracy)
        return 0

    def predict(self, x_pre, y_pre):
        x_score, y_score = self.__normalize(x_pre, y_pre)
        y_pred = np.dot(x_score, self.__get_params())

        for j in range(np.size(y_score, axis=0)):
            if y_pred[j] >= SCALE:
                y_pred[j] = 1
            else:
                y_pred[j] = 0
        return y_pred

    def __normalize(self, x_norm, y_norm):
        """process raw data"""

        # reshape
        self.__count = self.__count + 1
        x_norm = np.hstack((np.ones((np.size(x_norm, axis=0), 1)), x_norm))  # 加一列1
        y_norm = y_norm.reshape((np.size(x_norm, axis=0), 1))

        # initialize params
        if self.__count == 1:
            self.__set_params(np.random.random((np.size(x_norm, axis=1), 1)))

        # normalization
        average = np.mean(x_norm, axis=0)
        std = np.std(x_norm, axis=0)
        for i in range(np.size(x_norm, axis=1)):
            if std[i] == 0:
                continue
            else:
                x_norm[:, i] = (x_norm[:, i] - average[i]) / std[i]
        return x_norm, y_norm

    def __cost_function(self, x_inner, y_inner, j_list_in):
        """calculate cost j to visualize"""
        j = 0
        for i in range(np.size(y_inner, axis=0)):
            if y_inner[i] == 0:
                y_inner[i] = -1
        hinge = np.sum(1 - y_inner * np.dot(x_inner, self.__get_params()))
        if hinge > 0:
            j += hinge
        j_list_in.append(j)
        return j_list_in

    def __slack(self, y_get_in, y_inner):
        """calculate the slack.

        For example: y= -1 or 1
                     slack = max{0, y(wx)}

        Parameters:
            y_inner: label data
            y_get_in: predict data
        """
        slack = np.zeros_like(y_get_in)
        for i in range(np.size(y_inner, axis=0)):
            if y_inner[i] == 0:
                if y_get_in[i, ] > -1:
                    slack[i] = y_get_in[i]
                else:
                    slack[i] = 0
            else:
                if y_get_in[i, ] < 1:
                    slack[i] = y_get_in[i]
                else:
                    slack[i] = 0
        return slack

    def __back(self, x_in, y_in):
        """do backprop to adjust params.

        Parameters:
            x_in: a row of x
            y_in: a row of y
        """
        if y_in == 0:
            y_in = -1
        condition = np.sum(np.dot(x_in.T, self.__get_params()))
        if condition < 1:
            grad = LAMBDA * self.__get_params() - (x_in * y_in).reshape(10, 1)
            params = self.__get_params()
            params = params - AlPHA * grad
            self.__set_params(params)
        return 0

    def __plot_j(self, j_list_in):
        """Visualize the change of j"""
        plt.plot(range(EPOCH), j_list_in, c="r")
        plt.show()
        return 0


if __name__ == "__main__":
    x, y, x_test, y_test = leave_out()
    clf = LinearSVM().fit(x, y)
    clf.score(x_test, y_test)
