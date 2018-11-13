import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction import DictVectorizer


# 超参数
EPOCH = 1000
LAMBDA = 10
ALPHA = 0.01


class FC(object):
    def __init__(self, layer_num, layer_size, activation='ReLU'):
        self.__layer_num = layer_num
        self.__params_list = []
        self.__hidden_layer_size = layer_size
        self.__label_num = 0
        self.__activation = activation

    def fit(self, x_in, y_in):
        x_in, y_in = self.__initialize(x_in, y_in)
        j_list = []
        for epoch in range(EPOCH):
            y_pred = self.__forward(x_in)
            j_list.extend(self.__cost(y_pred, y_in))
            self.__backward(y_pred, y_in)
            if epoch % 10 == 0:
                print("EPOCH: %6d / %6d" % epoch, EPOCH, " | Cost: %6f", j_list[-1])
        self.__plot_j(j_list)
        return self

    def __plot_j(self, j_list):
        """Visualize the change of j.

        Parameters:
            j_list: record j of every epoch
        """
        plt.plot(range(EPOCH), j_list, c="r")
        plt.show()
        return 0

    def score(self, x_in, y_in):
        x_in, y_in = self.__initialize(x_in, y_in)
        y_pred = self.__forward(x_in)
        pass

    def predict(self):
        pass

    def __cost(self, y_pred, y_in):
        j1 = np.sum(np.sum((-y_in * np.log(y_pred) - (1 - y_in) * np.log(1 - y_pred)), axis=1), axis=0)
        j2 = 0
        for i in range(len(self.__params_list)):
            j2 += np.sum(np.sum((self.__params_list[i] * self.__params_list[i]), axis=1), axis=0)

        j = 1 / np.size(y_in, axis=0) * j1 + LAMBDA / 2 / np.size(y_in, axis=0) * j2
        return j

    def __forward(self, x_in):
        """get y_pred"""
        for i in range(self.__layer_num - 1):
            x_in = np.hstack((np.ones((np.size(x_in, axis=0), 1)), x_in))  # 加一列1
            x_in = np.dot(x_in, self.__params_list[i])
            if i != self.__layer_num - 1:                                  # 最后一层不激活
                if self.__activation == 'ReLU':
                    x_in = ReLU().forward(x_in)
        return x_in

    def __backward(self, y_pred, y_in):
        pass

    def __initialize(self, x_in, y_in):
        """Normalize x_in, initialize params, and reshape y_in.

        Parameters:
             x_in: raw x      (M, N)
             y_in: raw label  (M, )

        Returns:
            x_in: after normalization  (M, N)
            y_in: turn into matrix     (M, label_num)
        """

        # reshape y
        self.__label_num = self.__label_count(y_in)
        y_matrix = np.zeros((np.size(y_in, axis=0), self.__label_num))
        for j in range(np.size(y_in, axis=0)):
            y_matrix[j, y_in[j]] = 1

        # normalization
        average = np.mean(x_in, axis=0)
        std = np.std(x_in, axis=0)
        for i in range(np.size(x_in, axis=1)):
            if std[i] == 0:
                continue
            else:
                x_in[:, i] = (x_in[:, i] - average[i]) / std[i]

        # initialize params
        for k in range(self.__layer_num - 1):
            if k == 0:                           # 第一层
                params = np.random.random((np.size(x_in, axis=1) + 1, self.__hidden_layer_size[0]))
                epsilon_init = np.sqrt(6) / np.sqrt(np.size(x_in, axis=1) + 1 + self.__hidden_layer_size[0])
                params = 2 * epsilon_init * params - epsilon_init
                self.__params_list.append(params)
            elif k == self.__layer_num - 1:      # 最后
                params = np.random.random((self.__hidden_layer_size[-1] + 1, self.__layer_num))
                epsilon_init = np.sqrt(6) / np.sqrt(self.__hidden_layer_size[-1] + 1 + self.__layer_num)
                params = 2 * epsilon_init * params - epsilon_init
                self.__params_list.append(params)
            else:                                # 中间
                params = np.random.random((self.__hidden_layer_size[k-1] + 1, self.__hidden_layer_size[k]))
                epsilon_init = np.sqrt(6) / np.sqrt(self.__hidden_layer_size[k-1] + 1 + self.__hidden_layer_size[k])
                params = 2 * epsilon_init * params - epsilon_init
                self.__params_list.append(params)

        return x_in, y_matrix

    def __label_count(self, y_in):
        """将每个label与出现次数转为字典 {label值:出现次数}

        Parameters:
            y_in: label

        Returns:
            len(clf_data): 字典长度
        """

        clf_data = {}
        for i in range(np.size(y_in, axis=0)):
            if y_in[i] not in clf_data:
                clf_data[y_in[i]] = 1
            else:
                clf_data[y_in[i]] += 1
        return len(clf_data)


class ReLU(object):
    def forward(self, inputs):
        for i in range(np.size(inputs, axis=0)):
            if inputs[i] < 0:
                inputs[i] = 0
        return inputs

    def backward(self, inputs):
        for i in range(np.size(inputs, axis=0)):
            if inputs[i] > 0:
                inputs[i] = 1
            else:
                inputs[i] = 0
        return inputs


class Sigmoid(object):
    pass
