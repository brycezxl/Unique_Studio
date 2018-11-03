import numpy as np
import csv
from sklearn import svm
import pandas as pd


def process_data():
    def load():
        """
        Func:Load x_train and y_train as array
        Return:x_train and y_train
        """
        csv_file_x_train = csv.reader(open(r'X_train.csv', 'r'))
        x_train_get = []
        i = 0
        for stu in csv_file_x_train:
            i = i + 1
            if i == 1:
                continue
            else:
                x_train_get.extend(stu)
        x_train_get = np.reshape(np.array(x_train_get), (int(len(x_train_get) / 32), 32))

        # load y
        csv_file_y_train = csv.reader(open(r'y_train.csv', 'r'))
        y_train_get = []
        j = 0
        for stu in csv_file_y_train:
            j = j + 1
            if j == 1:
                continue
            else:
                y_train_get.extend(stu)
        y_train_get = np.reshape(np.array(y_train_get), (int(len(y_train_get)), 1))
        return x_train_get, y_train_get

    def test_load():
        """
        Func:Load x_test as array
        Out:x_test
        """
        csv_file_x_test = csv.reader(open(r'X_test.csv', 'r'))
        x_test_get = []
        i = 0
        for stu in csv_file_x_test:
            i = i + 1
            if i == 1:
                continue
            else:
                x_test_get.extend(stu)
        x_test_get = np.reshape(np.array(x_test_get), (int(len(x_test_get) / 32), 32))
        return x_test_get

    def calculate_mean(x_train_inner, location_inner):
        """
        Func: calculate mean for fill_blank
        :return: mean
        """
        row_inner, col_inner = np.shape(x_train_inner)
        sum_inner = 0
        count = 0
        for i in range(0, row_inner):
            if x_train_inner[i, location_inner] == '':
                continue
            else:
                count = count + 1
                value = float(x_train_inner[i, location_inner])
            sum_inner = sum_inner + value
        mean = sum_inner / count
        return mean

    def fill_blank(x_train_in):
        """
        Func: Fill in lost blanks
        """
        row, col = np.shape(x_train_in)
        # calculate mean
        mean_years_since_last_promotion = calculate_mean(x_train_in, -2)
        mean_years_in_current_role = calculate_mean(x_train_in, -3)
        mean_years_with_curr_manager = calculate_mean(x_train_in, -1)
        mean_years_at_company = calculate_mean(x_train_in, -4)
        mean_stock_option_level = calculate_mean(x_train_in, -8)
        # fill blank with mean
        for l in range(0, row):
            if x_train_in[l, -2] == '':
                x_train_in[l, -2] = str(
                    mean_years_since_last_promotion / mean_years_in_current_role * float(x_train_in[l, -3]))
                x_train_in[l, -2] = str(
                    float(x_train_in[l, -2]) + mean_years_since_last_promotion / mean_years_with_curr_manager * float(
                        x_train_in[l, -1]))
                x_train_in[l, -2] = str(0.5 * float(x_train_in[l, -2]))
            if x_train_in[l, -4] == '':
                x_train_in[l, -4] = str(mean_years_at_company / mean_stock_option_level * float(x_train_in[l, -8]))
        return x_train_in

    def feature_code_x(x_train_in):
        """
        Func: turn feature as string to num(1,2,3...)
        """
        row, col = np.shape(x_train_in)
        for i in range(0, col):  # 循环每一列
            try:
                float(x_train_in[1, i])
            except:  # 不是数字
                variety = []
                for j in range(0, row):  # 循环每一行
                    if x_train_in[j, i] in variety:
                        pass
                    else:
                        variety.append(x_train_in[j, i])
                    variety.sort()
                    x_train_in[j, i] = str(variety.index(x_train_in[j, i]) + 1)
        return x_train_in

    def feature_code_y(y_train_in):
        """
        Func: turn YES or NO into 1 or 0
        """
        row, col = np.shape(x_train_outer)
        for k in range(0, row):  # 循环每一列
            variety = ['Yes', 'No']
            if y_train_in[k] == variety[0]:
                y_train_in[k] = '1'
            else:
                y_train_in[k] = '0'
        return y_train_in

    def normalized(x_train_in):
        """
        Func: turn x into smaller and intenser numbers
        """
        row, col = np.shape(x_train_in)
        x_train_in = x_train_in.astype(dtype=np.float)
        std_x_test = np.std(x_train_in, axis=0)
        mean_x_test = np.mean(x_train_in, axis=0)
        for i in range(0, col):  # 每一列
            for j in range(0, row):
                if std_x_test[i] == 0:
                    continue
                x_train_in[j, i] = (x_train_in[j, i] - mean_x_test[i]) / std_x_test[i]
        return x_train_in

    # train x, y
    x_train_outer, y_train_outer = load()
    x_train_outer = fill_blank(x_train_outer)
    x_train_outer = feature_code_x(x_train_outer)
    y_train_outer = feature_code_y(y_train_outer)
    x_train_outer = normalized(x_train_outer)
    y_train_outer = y_train_outer.reshape(np.size(y_train_outer, 0), 1)
    y_train_outer = y_train_outer.astype(dtype=np.float)
    # test
    x_test_outer = test_load()
    x_test_outer = fill_blank(x_test_outer)
    x_test_outer = feature_code_x(x_test_outer)
    x_test_outer = normalized(x_test_outer)
    return x_train_outer, y_train_outer, x_test_outer


def predict():
    x_train, y_train, x_test = process_data()
    clf = svm.SVC(C=1,
                  coef0=0,
                  gamma='auto',
                  kernel='rbf')
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    (col, ) = np.shape(y_pred)
    y_pred = y_pred.astype(dtype=np.str)
    for i in range(0, col):                     # 数字转YES or NO
        if y_pred[i] == '1.0':
            y_pred[i] = 'YES'
        elif y_pred[i] == '0.0':
            y_pred[i] = 'NO'
    print(y_pred)
    y_pred = y_pred.astype(dtype=np.str)
    pd_data = pd.DataFrame(y_pred, columns=['Attribution'])
    pd_data.to_csv('pd_data.csv')


predict()