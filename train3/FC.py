import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction import DictVectorizer


# 超参数
EPOCH = 600
LAMBDA = 3
ALPHA = 0.1
Momentum = 0.9


def leave_out(data):
    """Divide data into 70% for train and 30% for test.

    Returns:
        x_get: matrix, data
        y_get: matrix, label
        x_get_train: matrix, data
        y_get_train: matrix, label

    """
    if data == 'titanic':
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

    elif data == 'wine':
        data_frame = pd.read_csv('./wine.csv')
        data_get = np.array(data_frame)

        data_boot = np.zeros_like(data_get)
        for k in range(np.size(data_get, axis=0)):
            random_int = np.random.randint(low=0, high=(np.size(data_get, axis=0) - 1))
            data_boot[k] = data_get[random_int, :]

        x_get = data_boot[:, 1:]
        y_get = data_boot[:, 0]
        x_test_get = data_get[:, 1:]
        y_test_get = data_get[:, 0]
        return x_get, y_get, x_test_get, y_test_get

    else:
        raise KeyError('Input titanic or wine!')


class FC(object):
    def __init__(self, layer_num, layer_size, activation):
        self.__layer_num = layer_num
        self.__params_list = []
        self.__hidden_layer_size = layer_size
        self.__label_num = 0
        self.__activation = activation
        self.__a_list = []
        self.__z_list = []
        self.__start_from_1 = 0
        self.__count = 0
        self.__momentum = [0] * (self.__layer_num - 1)

    def fit(self, x_in, y_in):
        x_in, y_in = self.__initialize(x_in, y_in)
        j_list = []
        for epoch in range(EPOCH):
            y_pred = self.__forward(x_in)
            j_list.append(self.__cost(y_pred, y_in))
            self.__backward(y_pred, y_in)
            if epoch % 10 == 0:
                print("EPOCH: %4d / %4d" % (epoch, EPOCH), ' | Cost: %7.4f' % float(j_list[-1]))
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
        y_pred = self.predict(x_in, y_in)
        count = 0
        for i in range(np.size(y_pred, axis=0)):
            if y_pred[i] == y_in[i]:
                count += 1
        acc = count / np.size(y_pred, axis=0) * 100
        print("ACC: %.4f%%" % acc)

    def predict(self, x_in, y_in):
        x_in, y_in = self.__initialize(x_in, y_in)
        y_pred = self.__forward(x_in)
        y_output = np.argmax(y_pred, axis=1)
        # if self.__start_from_1 == 1:
        #     y_output += 1
        return y_output

    def __cost(self, y_pred, y_in):

        j1 = np.sum(np.sum((-y_in * np.log(y_pred) - (1 - y_in) * np.log(1 - y_pred)), axis=1), axis=0)  # 把pred变成 0, 1
        # j1 = np.sum(np.sum((y_in - y_pred), axis=1), axis=0) ** 2 / 2
        j2 = 0
        for i in range(len(self.__params_list)):
            theta = self.__params_list[i]
            theta[:, 0] = 0
            j2 += np.sum(np.sum((theta * theta), axis=1), axis=0)

        j = 1 / np.size(y_in, axis=0) * j1 + LAMBDA / 2 / np.size(y_in, axis=0) * j2
        return j

    def __forward(self, x_in):
        """get y_pred

               layer1   layer2  ...     layer-1
        a:       a0        a1   ...         an       激活后
        z:               z0     ...   z(n-1)         未激活
        """

        # 清空列表再储存
        self.__a_list = []
        self.__z_list = []

        for i in range(self.__layer_num - 1):

            if i == 0:
                self.__a_list.append(np.hstack((np.ones((np.size(x_in, axis=0), 1)), x_in)))  # 初始x值

            # 参数传播
            x_in = np.hstack((np.ones((np.size(x_in, axis=0), 1)), x_in))                     # 加一列1
            x_in = np.dot(x_in, self.__params_list[i])
            self.__z_list.append(np.hstack((np.ones((np.size(x_in, axis=0), 1)), x_in)))      # 未激活前

            if i != self.__layer_num - 2:                                                     # 最后一层不激活
                x_in = self.__activation.forward(x_in)
            else:
                x_in = Softmax().forward(x_in)
            self.__a_list.append(np.hstack((np.ones((np.size(x_in, axis=0), 1)), x_in)))      # 激活后
        return x_in

    def __backward(self, y_pred, y_in):
        delta_list = [0] * (self.__layer_num - 1)
        delta_sum_list = [0] * (self.__layer_num - 1)

        y_new = np.zeros_like(y_in)

        for row in range(np.size(y_in, axis=0)):
            y_new[row, np.argmax(y_pred[row], axis=0)] = 1                              # 把pred变成 0, 1
            for layer in range(self.__layer_num - 2, -1, -1):                           # num-2--0 num-1个

                if layer == self.__layer_num - 2:                                       # 最后一层
                    delta_list[layer] = Softmax().backward(y_in[row], y_pred[row]).reshape(np.size(y_in, axis=1), 1)
                    a_get = self.__a_list[layer][row, :]
                    a_get = a_get.reshape(np.size(a_get, axis=0), 1)
                    delta_sum_list[layer] += np.dot(a_get, delta_list[layer].T)
                else:
                    a_back = self.__activation.backward(self.__z_list[layer][row, :])   # 激活函数反向
                    a_back = a_back.reshape(np.size(a_back, axis=0), 1)

                    delta = np.dot(self.__params_list[layer + 1], delta_list[layer + 1]) * a_back
                    delta_list[layer] = delta[1:]
                    a_get = self.__a_list[layer][row, :]
                    a_get = a_get.reshape(np.size(a_get, axis=0), 1)
                    delta_sum_list[layer] += np.dot(a_get, delta_list[layer].T)

        # 更新
        for i in range(self.__layer_num - 1):
            theta = self.__params_list[i]
            # theta[:, 0] = 0
            grad = Momentum * self.__momentum[i] + ALPHA * (delta_sum_list[i] + LAMBDA * theta) / np.size(y_in, axis=0)
            self.__params_list[i] -= grad
            self.__momentum[i] = grad

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
        self.__start_from_1 = 1
        for label in y_in:
            if label == 0:                 # 从0开始
                self.__start_from_1 = 0
        if self.__start_from_1 == 1:
            y_in -= 1                      # 从1开始的话-1,最后再加1

        self.__label_num = self.__label_count(y_in)
        y_matrix = np.zeros((np.size(y_in, axis=0), self.__label_num))
        for j in range(np.size(y_in, axis=0)):
            y_matrix[j, int(y_in[j])] = 1

        # normalization
        average = np.mean(x_in, axis=0)
        std = np.std(x_in, axis=0)
        for i in range(np.size(x_in, axis=1)):
            if std[i] == 0:
                continue
            else:
                x_in[:, i] = (x_in[:, i] - average[i]) / std[i]

        # initialize params
        self.__count += 1
        if self.__count == 1:
            for k in range(self.__layer_num - 1):
                if k == 0:                           # 第一层
                    params = np.random.random((np.size(x_in, axis=1) + 1, self.__hidden_layer_size[0]))
                    epsilon_init = np.sqrt(6) / np.sqrt(np.size(x_in, axis=1) + 1 + self.__hidden_layer_size[0])
                    params = 2 * epsilon_init * params - epsilon_init
                    self.__params_list.append(params)
                elif k == self.__layer_num - 2:      # 最后
                    params = np.random.random((self.__hidden_layer_size[-1] + 1, self.__label_num))
                    epsilon_init = np.sqrt(6) / np.sqrt(self.__hidden_layer_size[-1] + 1 + self.__label_num)
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
            if int(y_in[i]) not in clf_data:
                clf_data[int(y_in[i])] = 1
            else:
                clf_data[int(y_in[i])] += 1
        return len(clf_data)


class Tanh(object):
    def forward(self, inputs):
        inputs = (np.exp(inputs) - np.exp(-inputs)) / (np.exp(inputs) + np.exp(-inputs))
        return inputs

    def backward(self, inputs):
        inputs = 1 - self.forward(inputs) ** 2
        return inputs


class Sigmoid(object):
    def forward(self, inputs):
        inputs = 1 / (1 + np.exp(-inputs))
        return inputs

    def backward(self, inputs):
        inputs = self.forward(inputs)
        inputs = inputs * (1 - inputs)
        return inputs


class ReLU(object):
    def forward(self, inputs):
        for j in range(np.size(inputs, axis=1)):
            for i in range(np.size(inputs, axis=0)):
                if inputs[i, j] < 0:
                    inputs[i, j] = 0
        return inputs

    def backward(self, inputs):
        for i in range(np.size(inputs, axis=0)):
            if inputs[i] > 0:
                inputs[i] = 1
            else:
                inputs[i] = 0
        return inputs


class LeakyReLU(object):
    def forward(self, inputs):
        for j in range(np.size(inputs, axis=1)):
            for i in range(np.size(inputs, axis=0)):
                if inputs[i, j] < 0:
                    inputs[i, j] = 0.01 * inputs[i, j]
        return inputs

    def backward(self, inputs):
        for i in range(np.size(inputs, axis=0)):
            if inputs[i] >= 0:
                inputs[i] = 1
            else:
                inputs[i] = 0.01
        return inputs


class ELU(object):
    def forward(self, inputs):
        for j in range(np.size(inputs, axis=1)):
            for i in range(np.size(inputs, axis=0)):
                if inputs[i, j] < 0:
                    inputs[i, j] = 0.01 * (np.exp(inputs[i, j] - 1))
        return inputs

    def backward(self, inputs):
        for i in range(np.size(inputs, axis=0)):
            if inputs[i] >= 0:
                inputs[i] = 1
            else:
                inputs[i] = 0.01 * np.exp(inputs[i])
        return inputs


class Softmax(object):
    def forward(self, inputs):
        inputs_new = np.exp(inputs)
        inputs_sum = np.sum(inputs_new, axis=1)
        for i in range(np.size(inputs_new, axis=0)):
            inputs_new[i, :] = inputs_new[i, :] / inputs_sum[i]
        return inputs_new

    def backward(self, y, y_pred):
        return y_pred - y


if __name__ == "__main__":
    x, y, x_test, y_test = leave_out('titanic')
    clf = FC(layer_num=3, layer_size=[20], activation=ReLU()).fit(x, y)
    clf.score(x_test, y_test)
