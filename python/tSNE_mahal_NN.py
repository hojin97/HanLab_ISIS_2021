from keras.models import Sequential
from keras import layers
from keras.optimizers import SGD
from keras import metrics
import numpy as np
import tensorflow as tf
import os
import matplotlib.pyplot as plt
import math
import scipy as sp
from scipy.stats import multivariate_normal
from sklearn.mixture import GaussianMixture
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go

# The NN range includes the Mahalanobis distance.

UPPER_GYM_WORKOUT = ['Dumbbell_Curl', 'Dumbbell_Kickback', 'Hammer_Curl', 'Reverse_Curl']
mean_variance_dict = {
    'Dumbbell_Curl' : [[[-4.51528012, 2.8905306]], [[[ 2.34988035, -2.12924464], [-2.12924464, 6.79838459]]]],
    'Hammer_Curl' : [[[-3.00212511, 4.13081913]], [[[  7.77063184, -15.92849129], [-15.92849129, 35.63191858]]]],
    'Dumbbell_Kickback' : [[[9.91068313, -24.36830735]], [[[0.76274582, 0.04291525], [0.04291525, 0.97909614]]]],
    'Reverse_Curl' : [[[-0.25586194, 5.94336189]], [[[ 1.0657791, -0.20925074], [-0.20925074, 7.72194966]]]]
}

def open_coordinate_data():
    file_path = os.getcwd() + '/embedding_vecs'
    file_name = os.listdir(file_path)
    fp = os.path.join(file_path, file_name[0])
    file = open(fp, "r")

    coordinate_list = []
    target_list = []

    for line in file.readlines():
        data = line.split()
        xy_pos = np.array([float(data[0]), float(data[1])]).flatten()
        coordinate_list.append(xy_pos.copy())
        target_list.append(data[2])

    return coordinate_list, target_list

def open_matrix():
    base_path = os.getcwd() + '/emg_processing/emg_result'
    dir_1 = ["AAFT(0)"]
    dir_2 = ["Bi", "Hammer", "Rvcurl", "Tri"]

    matrix_list = []
    target_list = []

    for d1 in dir_1:
        for d2 in dir_2:
            matrix_path = os.path.join(base_path, d1, d2)
            matrix_filelist = os.listdir(matrix_path)
            for m_name in matrix_filelist:
                matrix_file = os.path.join(matrix_path, m_name)
                matrix = open(matrix_file, "r")
                matrix_data = matrix.readlines()
                for i, mr in enumerate(matrix_data):
                    matrix_data[i] = mr.split()
                    for j, value in enumerate(matrix_data[i]):
                        matrix_data[i][j] = float(value)
                matrix_data = np.array(matrix_data).flatten()
                matrix_list.append(matrix_data.copy())

    return matrix_list, target_list

def MahalanovisCalcDistance(input_data):
    # C, M, K, Y

    mahal_list = []

    for two_dim_data in input_data:
        tmp_mahal_list=[]
        for workout in UPPER_GYM_WORKOUT:
            mean = mean_variance_dict[workout][0]
            cov = mean_variance_dict[workout][1]

            x_mu = two_dim_data - mean
            mahal = np.dot(x_mu, np.linalg.inv(cov))
            mahal = np.dot(mahal, np.transpose(x_mu))
            tmp_mahal_list.append(mahal[0][0][0])
        mahal_list.append(tmp_mahal_list)

    return mahal_list

def Make_Sequential_model(show_summary=False):
    # relu, tanh, linear, exponential, sigmoid
    model = Sequential([
        layers.Dense(2048, activation='tanh', input_shape=(144,)),
        layers.Dense(2048, activation='tanh'),
        layers.Dense(2048, activation='tanh'),
        layers.Dense(2048, activation='tanh'),
        layers.Dense(1024, activation='tanh'),
        layers.Dense(1024, activation='tanh'),
        layers.Dense(1024, activation='tanh'),
        layers.Dense(512, activation='tanh'),
        layers.Dense(512, activation='tanh'),
        layers.Dense(4)
    ])
    if show_summary:
        model.summary()
    sgd = SGD(lr=0.001, decay=1e-07, momentum=0.5, nesterov=True)
    model.compile(loss='mean_squared_error', optimizer=sgd, metrics=['accuracy'])
    return model

def Concat_Data(mahal_list, matrix_list):
    DATA_SET = []

    for x, y in zip(matrix_list, mahal_list):
        DATA_SET.append([x, y])

    return DATA_SET

def Separate_Data(data_set):
    x_data = []
    y_data = []

    for data in data_set:
        x_data.append(data[0])
        y_data.append(data[1])
    return x_data, y_data

def Split_Dataset(DATA_SET, random_state = False):
    D1 = DATA_SET[0:60]
    D2 = DATA_SET[60:120]
    D3 = DATA_SET[120:180]
    D4 = DATA_SET[180:240]

    if random_state:
        np.random.shuffle(D1)
        np.random.shuffle(D2)
        np.random.shuffle(D3)
        np.random.shuffle(D4)

    l1 = int(len(D1) * 0.8)
    l2 = l1 + int((len(D1) - l1) * 0.5)
    train = []
    valid = []
    test = []

    train.extend(D1[0:l1])  # 0.8
    train.extend(D2[0:l1])
    train.extend(D3[0:l1])
    train.extend(D4[0:l1])

    valid.extend(D1[l1:l2])  # 0.1
    valid.extend(D2[l1:l2])
    valid.extend(D3[l1:l2])
    valid.extend(D4[l1:l2])

    test.extend(D1[l2:])  # 0.1
    test.extend(D2[l2:])
    test.extend(D3[l2:])
    test.extend(D4[l2:])

    x_train, y_train = Separate_Data(train)
    x_test, y_test = Separate_Data(test)
    x_valid, y_valid = Separate_Data(valid)

    return x_train, y_train, x_test, y_test, x_valid, y_valid

def Processing_Dataset(mahal_list, matrix_list):
    DATA_SET = Concat_Data(mahal_list, matrix_list)
    x_train, y_train, x_test, y_test, x_valid, y_valid = Split_Dataset(DATA_SET, random_state=True)

    return np.array(x_train), np.array(y_train), np.array(x_test), np.array(y_test), np.array(x_valid), np.array(y_valid)

def Mahal_Normarlization(mahal_list):
    normalized_mahal_list = []
    for data in mahal_list:
        x_min = min(data)
        x_max = max(data)

        temp = []
        for var in data:
            value = (var-x_min)/(x_max-x_min)
            temp.append(value)

        normalized_mahal_list.append(temp)
    return normalized_mahal_list

if __name__ == '__main__':
    # Data pre-processing
    coordinate_list, coo_target_list = open_coordinate_data()
    matrix_list, mat_target_list = open_matrix()

    mahal_list = MahalanovisCalcDistance(coordinate_list)
    mahal_list = Mahal_Normarlization(mahal_list)

    model = Make_Sequential_model()

    x_train, y_train, x_test, y_test, x_valid, y_valid = Processing_Dataset(mahal_list, matrix_list)

    model.fit(x_train, y_train, epochs=100, batch_size=16, validation_data=(x_valid, y_valid), validation_batch_size=16)

    predict_train = model.predict(x_train)
    pass
    # predict_test = model.predict(x_test)



