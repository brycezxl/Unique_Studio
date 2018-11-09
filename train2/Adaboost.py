import numpy as np
import pandas as pd
# from tqdm import tqdm
from sklearn.feature_extraction import DictVectorizer

# 引用三个算法
from Linear_SVM import LinearSVM
from CART import CartDecisionTree
from Logistic_Regression import LogisticRegression

# 超参
Adaboost_EPOCH = 15


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


def adaboost(x_ada, y_ada, x_test_in, y_test_in):
    # 初始化权重
    weight = np.ones((np.size(x_ada, axis=0), 1))
    weight /= np.size(x_ada, axis=0)
    weight_list = []
    classifier_list = []

    # 训练算法
    clf1 = LogisticRegression().fit(x_ada, y_ada)
    predict1 = clf1.predict(x_ada, y_ada)
    clf2 = LinearSVM().fit(x_ada, y_ada)
    predict2 = clf2.predict(x_ada, y_ada)
    clf3 = CartDecisionTree().fit(x_ada, y_ada)
    predict3 = clf3.predict(x_ada, y_ada)

    # 组合分类器
    for i in range(Adaboost_EPOCH):
        e1 = 0
        e2 = 0
        e3 = 0

        # 计算误差
        for j in range(np.size(x_ada, axis=0)):
            if predict1[j] != y_ada[j]:
                e1 += weight[j]
            if predict2[j] != y_ada[j]:
                e2 += weight[j]
            if predict3[j] != y_ada[j]:
                e3 += weight[j]

        # 选择小误差的模型
        if e1[0] <= e2[0] and e1[0] <= e3[0]:
            clf = clf1
            a = 1 / 2 * np.log((1 - e1[0]) / e1[0])
            predict = predict1
        elif e2[0] <= e1[0] and e2[0] <= e3[0]:
            clf = clf2
            a = 1 / 2 * np.log((1 - e2[0]) / e2[0])
            predict = predict2
        else:
            clf = clf3
            a = 1 / 2 * np.log((1 - e3[0]) / e3[0])
            predict = predict3

        # 更新权重
        z = np.sum(np.exp(-a * (y_ada - 0.5) * (predict - 0.5) * 4), axis=0)      # x, y化成０或１
        weight = weight * np.exp(-a * (y_ada - 0.5) * (predict - 0.5) * 4) / z
        weight_list.append(a)
        classifier_list.append(clf)

    # 评估acc
    predict_sum = 0
    predict_get = np.zeros_like(y_test_in)
    acc_count = 0
    for l in range(Adaboost_EPOCH):
        predict_sum += weight_list[l] * (classifier_list[l].predict(x_test_in, y_test_in) - 0.5) * 2
    for k in range(np.size(y_test_in, axis=0)):
        if predict_sum[k] / Adaboost_EPOCH >= 0:
            predict_get[k] = 1
        else:
            predict_get[k] = 0
        if predict_get[k] == y_test_in[k]:
            acc_count += 1
    acc = acc_count / np.size(y_test_in, axis=0) * 100
    print('Adaboost ACC: %.2f%%' % acc)
    return 0


if __name__ == "__main__":
    x, y, x_test, y_test = leave_out()
    adaboost(x, y, x_test, y_test)
