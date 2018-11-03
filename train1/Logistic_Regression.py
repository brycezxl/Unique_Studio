import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# from tqdm import tqdm
from sklearn.feature_extraction import DictVectorizer


# 超参
EPOCH = 2000
LAMBDA = 0
AlPHA = 0.01
SCALE = 0.5


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
    """Do logistic regression.

    Methods:　predict -- Show predict labels with given x,
　　　　　　　　fix     -- Train the model with given x and y,
　　　　　　　　score   -- Show the mean accuracy on the given test data and labels

    Attribute:
        __params: parameters
        __count: decide when to initialize params
    """

    def __init__(self):
        self.__params = 0
        self.__count = 0

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

    def __cost_function(self, y_get, y_in, j_list_in):
        """calculate cost j for visualization"""

        j = -1 / np.size(y_get, axis=0) * (np.sum((y_in * np.log(y_get)), axis=0)
                                           + np.sum(((1 - y_in) * np.log(1 - y_get)), axis=0))
        j_list_in.append(j)
        return j_list_in

    def __back(self, x_in, y_get, y_in):
        """do backprop to adjust params"""

        grad = AlPHA / np.size(x_in, axis=0) * np.dot(x_in.T, (y_get - y_in))
        params = self.__get_params()
        params = params - grad
        self.__set_params(params)
        return 0

    def __plot_j(self, j_list_in):
        """Visualize the change of j"""

        plt.plot(range(EPOCH), j_list_in, c="r")
        plt.show()
        return 0

    def __forward(self, x_forward):
        """forward prop with sigmoid"""

        y_predict = 1 / (1 + np.exp(-np.dot(x_forward, self.__get_params())))
        return y_predict

    def fit(self, x_in, y_in):
        """Fit the model according to the given training data."""
        # pbar = tqdm(total=EPOCH)

        x_in, y_in = self.__normalize(x_in, y_in)

        # initialize the list to store cost j
        j_list = []

        # fit
        for epoch in range(EPOCH):
            y_get = self.__forward(x_in)
            j_list = self.__cost_function(y_get, y_in, j_list)
            self.__back(x_in, y_get, y_in)
            if epoch % 50 == 0:
                # pbar.update(50)
                print("EPOCH: ", epoch, " | Cost: ", j_list[-1])
        # pbar.close()

        # plot
        self.__plot_j(j_list)
        print("Minimized cost: %.5f" % j_list[-1])
        return self

    def predict(self, x_pre):
        """Predict class labels for samples in X."""

        # y_pre is useless here
        x_pre, y_pre = self.__normalize(x_pre, np.ones((np.size(x_pre, axis=0), 1)))

        y_pre = self.__forward(x_pre)

        # 化成0/1
        for j in range(np.size(y_pre, axis=0)):
            if y_pre[j] >= SCALE:
                y_pre[j] = 1
            else:
                y_pre[j] = 0

        return y_pre

    def __get_params(self):
        """Get parameters for this estimator."""

        params = self.__params
        return params

    def __set_params(self, params_in):
        """Set parameters for this estimator."""

        self.__params = params_in
        return 0

    def score(self, x_score, y_score):
        """Returns the mean accuracy on the given test data and labels."""

        x_score, y_score = self.__normalize(x_score, y_score)
        y_pred = self.__forward(x_score)

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


x, y = load_train()
x_test, y_test = load_test()
clf = LogisticRegression().fit(x, y)
clf.score(x_test, y_test)

# def precision_recall_f1_roc_auc(beta=1):
#     """计算查准率与查全率并图示，打印f-beta度量、宏f1，微f1, ROC, AUC
#
#     Args:
#         beta: if beta < 1, emphasis on precision
#               if beta > 1, emphasis on recall
#     """
#
#     sort_predict = np.sort(logistic_regression_test.predict, axis=0)
#     tp = np.zeros((logistic_regression_test.rows, 1))
#     fp = np.zeros((logistic_regression_test.rows, 1))
#     fn = np.zeros((logistic_regression_test.rows, 1))
#     tn = np.zeros((logistic_regression_test.rows, 1))
#     predict_copy2 = np.zeros((logistic_regression_test.rows, 1))
#     for k in range(logistic_regression_test.rows):
#         for l in range(logistic_regression_test.rows):
#
#             # 处理predict
#             if logistic_regression_test.predict[l] < sort_predict[logistic_regression_test.rows - 1 - k]:
#                 predict_copy2[l] = 0
#             else:
#                 predict_copy2[l] = 1
#
#             # 计算混淆矩阵
#             if predict_copy2[l] == y_test[l] and y_test[l] == 1:
#                 tp[k] = tp[k] + 1
#             elif predict_copy2[l] == y_test[l] and y_test[l] == 0:
#                 tn[k] = tn[k] + 1
#             elif predict_copy2[l] != y_test[l] and y_test[l] == 1:
#                 fn[k] = fn[k] + 1
#             else:
#                 fp[k] = fp[k] + 1
#
#     p = tp / (tp + fp)  # 查准率
#     r = tp / (tp + fn)  # 查全率
#     plt.figure(1)
#     plt.subplot(121)
#     plt.xlabel('P')
#     plt.ylabel('R')
#     plt.title('P-R')
#     plt.plot(p, r)
#
#     # F_beta度量(根据beta比较学习器，重视较小值)
#     f_beta = ((1 + beta ** 2) * r * p) / (beta ** 2 * p + r)
#     f_beta = np.mean(f_beta, axis=0)
#     print("F_beta:         %.2f" % f_beta)
#
#     # 宏F1
#     macro_p = np.sum(p, axis=0) / logistic_regression_test.rows
#     macro_r = np.sum(r, axis=0) / logistic_regression_test.rows
#     macro_f1 = 2 * macro_r * macro_p / (macro_r + macro_p)
#     print("macro-F1:       %.2f" % macro_f1)
#
#     # 微F1
#     micro_tp = np.mean(tp, axis=0)
#     micro_fn = np.mean(fn, axis=0)
#     micro_fp = np.mean(fp, axis=0)
#     micro_p = micro_tp / (micro_tp + micro_fp)
#     micro_r = micro_tp / (micro_tp + micro_fn)
#     micro_f1 = float(2 * micro_p * micro_r / (micro_p + micro_r))
#     print("micro-F1:       %.2f" % micro_f1)
#
#     # ROC
#     tpr = tp / (tp + fn)
#     fpr = fp / (tn + fp)
#     plt.subplot(122)
#     plt.xlabel('FPR')
#     plt.ylabel('TPR')
#     plt.title('ROC')
#     plt.plot(fpr, tpr)
#     plt.show()
#
#     # AUC(ROC的面积）
#     auc = 0
#     for f in range(logistic_regression_test.rows - 1):
#         auc = auc + 1 / 2 * (tpr[f + 1] - tpr[f]) * (fpr[f] + fpr[f + 1])
#     print("AUC:            %.2f" % (1 - auc))
#
#     return 0

