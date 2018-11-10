import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer

# 引用三个算法
from Linear_SVM import LinearSVM
from CART import CartDecisionTree
from Logistic_Regression import LogisticRegression


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


def bagging(x_in, y_in, x_test_in, y_test_in):
    """改变权重,训练多个分类器,投票"""

    # 先预测
    clf1 = LogisticRegression().fit(x_in, y_in)
    predict1 = clf1.predict(x_test_in, y_test_in)
    clf2 = LinearSVM().fit(x_in, y_in)
    predict2 = clf2.predict(x_test_in, y_test_in)
    clf3 = CartDecisionTree().fit(x_in, y_in)
    predict3 = clf3.predict(x_test_in, y_test_in)

    # 收集投票
    predict = np.zeros_like(predict1)
    count = 0
    for i in range(np.size(y_test_in, axis=0)):
        if predict1[i] == predict2[i]:
            predict[i] = predict2[i]
        elif predict1[i] == predict3[i]:
            predict[i] = predict1[i]
        else:
            predict[i] = predict2[i]
        if predict[i] == y_test_in[i]:
            count += 1
    acc = count / np.size(y_test_in, axis=0) * 100
    print("Bagging ACC: %.2f%%" % acc)
    return 0


if __name__ == "__main__":
    x, y, x_test, y_test = leave_out()
    bagging(x, y, x_test, y_test)



