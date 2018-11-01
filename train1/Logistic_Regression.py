import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from sklearn.feature_extraction import DictVectorizer


# 超参
EPOCH = 3000
LAMBDA = 0
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


def precision_recall_f1_roc_auc(beta=1):
    """计算查准率与查全率并图示，打印f-beta度量、宏f1，微f1, ROC, AUC

    Args:
        beta: if beta < 1, emphasis on precision
              if beta > 1, emphasis on recall
    """

    sort_predict = np.sort(logistic_regression_test.predict, axis=0)
    tp = np.zeros((logistic_regression_test.rows, 1))
    fp = np.zeros((logistic_regression_test.rows, 1))
    fn = np.zeros((logistic_regression_test.rows, 1))
    tn = np.zeros((logistic_regression_test.rows, 1))
    predict_copy2 = np.zeros((logistic_regression_test.rows, 1))
    for k in range(logistic_regression_test.rows):
        for l in range(logistic_regression_test.rows):

            # 处理predict
            if logistic_regression_test.predict[l] < sort_predict[logistic_regression_test.rows - 1 - k]:
                predict_copy2[l] = 0
            else:
                predict_copy2[l] = 1

            # 计算混淆矩阵
            if predict_copy2[l] == y_test[l] and y_test[l] == 1:
                tp[k] = tp[k] + 1
            elif predict_copy2[l] == y_test[l] and y_test[l] == 0:
                tn[k] = tn[k] + 1
            elif predict_copy2[l] != y_test[l] and y_test[l] == 1:
                fn[k] = fn[k] + 1
            else:
                fp[k] = fp[k] + 1

    p = tp / (tp + fp)  # 查准率
    r = tp / (tp + fn)  # 查全率
    plt.figure(1)
    plt.subplot(121)
    plt.xlabel('P')
    plt.ylabel('R')
    plt.title('P-R')
    plt.scatter(p, r)

    # F_beta度量(根据beta比较学习器，重视较小值)
    f_beta = ((1 + beta ** 2) * r * p) / (beta ** 2 * p + r)
    f_beta = np.mean(f_beta, axis=0)
    print("F_beta:         %.2f" % f_beta)

    # 宏F1
    macro_p = np.sum(p, axis=0) / logistic_regression_test.rows
    macro_r = np.sum(r, axis=0) / logistic_regression_test.rows
    macro_f1 = 2 * macro_r * macro_p / (macro_r + macro_p)
    print("macro-F1:       %.2f" % macro_f1)

    # 微F1
    micro_tp = np.mean(tp, axis=0)
    micro_fn = np.mean(fn, axis=0)
    micro_fp = np.mean(fp, axis=0)
    micro_p = micro_tp / (micro_tp + micro_fp)
    micro_r = micro_tp / (micro_tp + micro_fn)
    micro_f1 = float(2 * micro_p * micro_r / (micro_p + micro_r))
    print("micro-F1:       %.2f" % micro_f1)

    # ROC
    tpr = tp / (tp + fn)
    fpr = fp / (tn + fp)
    plt.subplot(122)
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title('ROC')
    plt.scatter(fpr, tpr)
    plt.show()

    # AUC(ROC的面积）
    auc = 0
    for f in range(logistic_regression_test.rows - 1):
        auc = auc + 1 / 2 * (tpr[f + 1] - tpr[f]) * (fpr[f] + fpr[f + 1])
    print("AUC:            %.2f" % (1 - auc))

    return 0


# train
x, y = load_train()
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
precision_recall_f1_roc_auc()
