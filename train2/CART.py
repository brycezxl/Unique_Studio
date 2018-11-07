import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer


def leave_out():
    """Divide data into 70% for train and 30% for test.

    Returns:
        x_get: matrix, data
        y_get: matrix, label
        x_get_train: matrix, data
        y_get_train: matrix, label

    """
    data_frame = pd.read_csv('titanic/train.csv')

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


class CartDecisionTree(object):
    """Cart Decision Tree.

    Methods:
        fit   -- build the tree with given data
        score -- return the acc predicted by the tree

    Attributes:
        value: chosen split value
        truebranch, falsebranch: binary branches which the data are divided into each time
        kind: kinds of data in the chosen column
        col: the chosen column
        summary: details of each point
        current_gain: gini of data before split

    """
    def __init__(self, value=None, truebranch=None, falsebranch=None, kind=None, col=-1, summary=None, result=None):
        self.value = value
        self.truebranch = truebranch
        self.falsebranch = falsebranch
        self.kind = kind
        self.col = col
        self.summary = summary
        self.current_gain = 0
        self.result = result
        self.ban_list = []

    def fit(self, x_in, y_in):
        """build the tree by x_in and y_in"""
        y_in = np.reshape(y_in, (np.size(y_in, axis=0), 1))
        data_total = np.hstack((x_in, y_in))
        return self.__build_tree(data_total)

    def __build_tree(self, data_build):

        self.current_gain = self.__gini(data_build)
        dic = {'impurity': '%6.3f' % self.current_gain, 'samples': '%d' % np.size(data_build, axis=0)}

        # 类别完全相同
        clf1 = self.__type_count(data_build, -1)
        for c in clf1:
            if clf1[c] == np.size(data_build, axis=0):
                return CartDecisionTree(kind=c, summary=dic, result=data_build)

        # 遍历完所有的特征
        if self.__feature_end(data_build):
            return CartDecisionTree(kind=self.__major_feature(data_build), summary=dic, result=data_build)

        # get best gain and split data
        best_col, best_val, best_gain = self.__choose_best_gain(data_build)
        data1, data2 = self.__split_data(data_build, best_val, best_col)

        # 一个特征取一次
        self.ban_list.append(best_col)

        # stop or not
        if best_gain > 0:
            true_branch = self.__build_tree(data1)
            false_branch = self.__build_tree(data2)
            return CartDecisionTree(col=best_col, value=best_val, truebranch=true_branch, falsebranch=false_branch,
                                    summary=dic)
        else:
            return CartDecisionTree(kind=self.__type_count(data_build, -1), summary=dic, result=data_build)

    def __feature_end(self, data_in):
        type_counter = 1
        for col in range(np.size(data_in, axis=1) - 1):
            if col in self.ban_list:
                continue
            if len(self.__type_count(data_in, col)) != 1:
                type_counter = 0
        return type_counter

    def __choose_best_gain(self, data_to_choose):
        """choose the best gain to decide how to split data"""
        best_gain_in = 0
        best_val_in = None
        best_col_in = None
        for i in range(np.size(data_to_choose, axis=1) - 1):
            if i in self.ban_list:
                continue
            kinds = self.__type_count(data_to_choose, i)
            for kind in kinds:
                data_get1, data_get2 = self.__split_data(data_to_choose, kind, i)
                if isinstance(data_get1, int) or isinstance(data_get2, int):
                    continue
                gini_get = self.__gini_gain(data_get1, data_get2)
                if gini_get >= best_gain_in:
                    best_gain_in = gini_get
                    best_val_in = kind
                    best_col_in = i
        return best_col_in, best_val_in, best_gain_in

    def __gini_gain(self, data_in1, data_in2):
        """calculate gini gain"""
        p = np.size(data_in1, axis=0) / (np.size(data_in1, axis=0) + np.size(data_in2, axis=0))
        gain = self.current_gain - p * self.__gini(data_in1) - (1 - p) * self.__gini(data_in2)
        return gain

    def score(self, x_in, y_in):
        pass

    def __gini(self, x_gini):
        """计算gini增量."""
        clf_x = self.__type_count(x_gini, -1)
        d = np.size(x_gini, axis=0)
        gini_sum = 0
        for k in clf_x:
            gini_sum += (clf_x[k] / d) ** 2
        return 1 - gini_sum

    def __split_data(self, data_split, value, column):
        """split data by value,column

        Parameters:
            data_split: data
            column: split feature
            value: where to separate feature

        Returns:
            2 array
        """
        split1 = 0
        split2 = 0

        # 数字
        if isinstance(value, int) or isinstance(value, float):
            for i in range(np.size(data_split, axis=0)):
                if data_split[i, column] >= value:
                    if isinstance(split1, int):                                        # 初始化
                        split1 = data_split[i, :]
                    else:
                        split1 = np.vstack((split1, data_split[i, :]))     # 竖着叠加
                else:
                    if isinstance(split2, int):
                        split2 = data_split[i, :]
                    else:
                        split2 = np.vstack((split2, data_split[i, :]))

        # 字符
        else:
            for i in range(np.size(data_split, axis=0)):
                if data_split[i, column] == value:
                    if split1 == 0:
                        split1 = data_split[i, :]
                    else:
                        split1 = np.vstack((split1, data_split[i, :]))
                else:
                    if split2 == 0:
                        split2 = data_split[i, :]
                    else:
                        split2 = np.vstack((split1, data_split[i, :]))
        if np.shape(split1) == (10, ):
            split1 = split1.reshape((1, 10))
        if np.shape(split2) == (10, ):
            split1 = split2.reshape((1, 10))
        return split1, split2

    def __type_count(self, data_split, col):
        """将每个特征与特征出现次数转为字典"""
        if np.shape(data_split) == (10, ):
            data_split = data_split.reshape((1, 10))
        clf_data = {}
        for i in range(np.size(data_split, axis=0)):
            if data_split[i, col] not in clf_data:
                clf_data[data_split[i, col]] = 1
            else:
                clf_data[data_split[i, col]] += 1
        return clf_data

    def __prune(self):
        pass

    def __major_feature(self, data_in):
        dic_get = self.__type_count(data_in, -1)
        major = 0
        for kind in dic_get:
            if dic_get[kind] > major:
                major = dic_get[kind]
        return major


if __name__ == "__main__":
    x, y, x_test, y_test = leave_out()
    clf = CartDecisionTree().fit(x, y)
    clf.score(x_test, y_test)
