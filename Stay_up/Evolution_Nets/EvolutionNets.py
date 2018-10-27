import numpy as np
import csv


def load_data():
    csv_file = csv.reader(open(r'/home/bryce/Documents/UniqueStudio/EvolutionNets/evolution.csv', 'r'))
    rev = []
    i_in = 0
    for part in csv_file:
        i_in = i_in + 1
        if i_in == 1:
            continue
        rev.extend(part)
    array_rev = np.array(rev, dtype=np.float64)
    size = np.size(array_rev, axis=0)
    array_rev = array_rev.reshape(int(size / 4), 4)
    x1_get = array_rev[:, 1]
    x2_get = array_rev[:, 2]
    y_get = array_rev[:, 3]
    return x1_get, x2_get, y_get


def normalization(data_in):
    data_in = (data_in - np.mean(data_in, axis=0)) / np.std(data_in, axis=0)
    return data_in


def tanh(data_in):
    act_val = (np.exp(data_in) - np.exp(-data_in)) / (np.exp(data_in) + np.exp(-data_in))
    return act_val


def sigmoid(data_in):
    act_val = 1 / (1 + np.exp(-data_in))
    return act_val


def layer_1(x_in):
    x_in = np.hstack((np.zeros((1, )), x_in))
    theta0 = np.random.random((4, 3))
    x_out = tanh(np.dot(theta0, x_in))
    return x_out, theta0


def layer_2(x_in):
    x_in = np.hstack((np.zeros((1, )), x_in))
    theta0 = np.random.random((1, 5))
    x_out = sigmoid(np.dot(theta0, x_in))
    return x_out, theta0


def forward_prop(x_in, theta1_in, theta2_in):
    x_in = np.hstack((np.ones((1,)), x_in))
    x_out = tanh(np.dot(theta1_in, x_in))
    x_out2 = np.hstack((np.ones((1,)), x_out))
    y_get_in = sigmoid(np.dot(theta2_in, x_out2))
    return y_get_in, x_in, x_out2


def cost_func(y_get_in, y_in):
    j = -(sum(y_in * np.log(y_get_in)) + sum((1 - y_in) * np.log(1 - y_get_in)))
    return j


def back_prop(x_in, x_mid_in, y_get_in, y_in):



x1, x2, y = load_data()
x1 = normalization(x1)
x2 = normalization(x2)
x = np.vstack((x1, x2))
x = x.reshape((400, 2))
theta1 = np.random.random((4, 3))
theta2 = np.random.random((1, 5))
# y_get = get_y(x, theta1, theta2)
for i in range(400):
    y_get, x, x_mid = forward_prop(x[i, :], theta1, theta2)
    j = cost_func(y_get, y[i])
    grad = back_prop(x[i, :], x_mid, y_get, y[i])