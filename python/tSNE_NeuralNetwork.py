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

def combine_data(coordinate_list, matrix_list, coo_target_list):
    DATA_SET = []
    for mat, coo, coo_t in zip(matrix_list, coordinate_list, coo_target_list):
        DATA_SET.append([mat, coo, coo_t])
    return DATA_SET

def separate_DATA(dataset, type=None):
    x_data = []
    y_data = []

    if type==None:
        for row in dataset:
            x_data.append(row[0])
            y_data.append(row[1])
        x_data = np.array(x_data)
        y_data = np.array(y_data)
        return x_data, y_data
    elif type == "color":
        y_color_marker = []
        for row in dataset:
            x_data.append(row[0])
            y_data.append(row[1])
            if row[2] == 'bc' : y_color_marker.append(['c', '^'])
            elif row[2] == 'hc': y_color_marker.append(['m', 'o'])
            elif row[2] == 'rv': y_color_marker.append(['k', 's'])
            elif row[2] == 'tc': y_color_marker.append(['y', 'd'])

        x_data = np.array(x_data)
        y_data = np.array(y_data)
        y_color_marker = np.array(y_color_marker)
        return x_data, y_data, y_color_marker

# x,y_train x,y_test x,y_valid  [8:1:1]
def separate_DATA_SET(DATA_SET):
    D1 = DATA_SET[0:60]
    D2 = DATA_SET[60:120]
    D3 = DATA_SET[120:180]
    D4 = DATA_SET[180:240]

    np.random.shuffle(D1)
    np.random.shuffle(D2)
    np.random.shuffle(D3)
    np.random.shuffle(D4)

    l1 = int(len(D1)*0.8)
    l2 = l1 + int((len(D1)-l1)*0.5)
    train = []
    valid = []
    test = []

    train.extend(D1[0:l1])             # 0.8
    train.extend(D2[0:l1])
    train.extend(D3[0:l1])
    train.extend(D4[0:l1])

    valid.extend(D1[l1:l2])            # 0.1
    valid.extend(D2[l1:l2])
    valid.extend(D3[l1:l2])
    valid.extend(D4[l1:l2])

    test.extend(D1[l2:])               # 0.1
    test.extend(D2[l2:])
    test.extend(D3[l2:])
    test.extend(D4[l2:])

    x_train, y_train, y_train_color_marker = separate_DATA(train, "color")
    x_valid, y_valid = separate_DATA(valid)
    x_test, y_test, y_test_color_marker = separate_DATA(test, "color")

    return x_train, y_train, y_train_color_marker, x_test, y_test, y_test_color_marker, x_valid, y_valid

def image_show_and_compare(data1, data2, d_color):
    plt.figure(1)
    plt.scatter(data1[:, 0], data1[:, 1], c=d_color)
    plt.title("Type 1")

    plt.figure(2)
    plt.scatter(data2[:, 0], data2[:, 1], c=d_color)
    plt.title("Type 2")

    plt.show()

def Make_Sequential_model(show_summary=False):
    # relu, tanh, linear, exponential, sigmoid
    model = Sequential([
        layers.Dense(700, activation='tanh', input_shape=(144,)),
        layers.Dense(700, activation='tanh'),
        layers.Dense(700, activation='tanh'),
        layers.Dense(700, activation='tanh'),
        layers.Dense(2)
    ])
    if show_summary:
        model.summary()
    sgd = SGD(lr=0.001, decay=1e-07, momentum=0.5, nesterov=True)
    model.compile(loss='mean_squared_error', optimizer=sgd, metrics=['accuracy'])
    return model

def processing_dataset(coordinate_list, matrix_list, coo_target_list):
    DATA_SET = combine_data(coordinate_list, matrix_list, coo_target_list)

    x_train, y_train, y_train_color_marker, x_test, y_test, y_test_color_marker, x_valid, y_valid = separate_DATA_SET(DATA_SET)

    return x_train, y_train, y_train_color_marker, x_test, y_test, y_test_color_marker, x_valid, y_valid

def KNN_processing(scatter_data, scatter_target, input_data, K = 8):
    distance_list = []
    for coordinate, target in zip(scatter_data, scatter_target):
        distance = math.sqrt((input_data[0] - coordinate[0])**2 + (input_data[1] - coordinate[1])**2)
        distance_list.append([distance, coordinate, target])
    d_list = sorted(distance_list)

    classification_data ={
        "c" : 0,
        "m" : 0,
        "k" : 0,
        "y" : 0,
    }

    for time in range(K):
        classification_data[d_list[time][2]] += 1

    return classification_data

def find_Centroid(scatter_data, scatter_target, gausian_num):
    centroid = []
    each_points = {
        'c': [],
        'm': [],
        'k': [],
        'y': [],
    }

    for data, target in zip(scatter_data, scatter_target[:, 0]):
        each_points[target].append(data)

    # for key in each_points.keys():
    #     each_points[key] = np.array(each_points[key])
    #     gmm = GaussianMixture(n_components=gausian_num, random_state=0).fit(each_points[key])
    #
    #     centroid.append([gmm.means_, gmm.covariances_])

    each_points['c'] = np.array(each_points['c'])
    gmm1 = GaussianMixture(n_components=gausian_num, random_state=0).fit(each_points['c'])
    centroid.append([gmm1.means_, gmm1.covariances_])

    each_points['m'] = np.array(each_points['m'])
    gmm2 = GaussianMixture(n_components=gausian_num, random_state=0).fit(each_points['m'])
    centroid.append([gmm2.means_, gmm2.covariances_])

    each_points['y'] = np.array(each_points['y'])
    gmm3 = GaussianMixture(n_components=gausian_num, random_state=0).fit(each_points['y'])
    centroid.append([gmm3.means_, gmm3.covariances_])

    each_points['k'] = np.array(each_points['k'])
    gmm4 = GaussianMixture(n_components=gausian_num, random_state=0).fit(each_points['k'])
    centroid.append([gmm4.means_, gmm4.covariances_])

    show_img_gaussian_distribution(scatter_data, scatter_target, each_points['c'], each_points['m'], each_points['y'], each_points['k'], gmm1, gmm2, gmm3, gmm4)

    return centroid

def show_img_gaussian_distribution(scatter_data, scatter_target, point_data1, point_data2, point_data3, point_data4, gmm1, gmm2, gmm3, gmm4):
    # x_min, x_max = point_data[:, 0].min(), point_data[:, 0].max()
    # y_min, y_max = point_data[:, 1].min(), point_data[:, 1].max()
    x, y = np.meshgrid(np.linspace(-10, 15, num=30), np.linspace(-30, 20, num=30))
    pos = np.dstack((x,y))
    z1 = multivariate_normal.pdf(pos, mean=gmm1.means_[0], cov=gmm1.covariances_[0])
    z2 = multivariate_normal.pdf(pos, mean=gmm2.means_[0], cov=gmm2.covariances_[0])
    # z3 = multivariate_normal.pdf(pos, mean=gmm3.means_[0], cov=gmm3.covariances_[0])
    z4 = multivariate_normal.pdf(pos, mean=gmm4.means_[0], cov=gmm4.covariances_[0])

    plt.figure(8)
    plt.contour(x, y, z1, 5, alpha=1, linewidths=1)
    plt.contour(x, y, z2, 5, alpha=1, linewidths=1)
    # plt.contour(x, y, z3, 5, alpha=1, linewidths=1)
    plt.contour(x, y, z4, 5, alpha=1, linewidths=1)
    plt.xticks([-10, -5, 0, 5, 10, 15])
    plt.yticks([-30, -25, -20, -15, -10, -5, 0, 5, 10, 15, 20])
    # plt.axis('off')
    plt.savefig('gmm.png')

    # ax = fig.gca(projection='3d')
    # ax.set_xticks([-10, -5, 0, 5, 10, 15])
    # ax.set_yticks([-30, -25, -20, -15, -10, -5, 0, 5, 10, 15])
    # ax.set_zticks([-10, 5])
    # ax.set_xlabel("X")
    # ax.set_ylabel("Y")
    # ax.set_zlabel("Z")
    # ax.plot_surface(x, y, z1, cmap='Greens', alpha=1)
    # ax.plot_surface(x, y, z2, cmap='Greens', alpha=0.7)
    # ax.plot_surface(x, y, z3, cmap='Greens', alpha=0.5)
    # ax.plot_surface(x, y, z4, cmap='Greens', alpha=0.8)
    # plt.tight_layout()

    plt.figure(7)
    # plt.contour(x, y, z, 5, alpha=1, linewidths=1.1)

    for x, y, color, mark in zip(scatter_data[:,0], scatter_data[:,1], scatter_target[:,0], scatter_target[:,1]):
        plt.scatter(x, y, c=color, marker=mark)
    plt.xticks([-10, -5, 0, 5, 10, 15])
    plt.yticks([-30, -25, -20, -15, -10, -5, 0, 5, 10, 15, 20])
    plt.show()

def show_img_together(scatter_data, scatter_target, input_data=None, gausian_num = 1):
    plt.figure(3)
    if gausian_num == 1:
        if input_data is not None:
            plt.scatter(input_data[0], input_data[1], c='brown', marker='*')
    elif gausian_num == 2:
        print(input_data[:, 0])
        print(input_data[:, 1])
        print()
        plt.scatter(input_data[:,0], input_data[:,1], c='brown', marker='*')

    for x, y, c, s in zip(scatter_data[:,0], scatter_data[:,1], scatter_target[:,0], scatter_target[:,1]):
        plt.scatter(x, y, c=c, marker=s, alpha=0.6)
    plt.show()

def processing_confusionmatrix(confusion_matrix_material):
    base_class = ['c', 'm', 'k', 'y']
    confusion_matrix_material = sorted(confusion_matrix_material)
    # rr = 실제 값이 r인데 r로 분류한 경우.
    # rg = 실제 값이 r인데 g로 분류한 경우.
    confusion_matrix = {
        'cc': 0,
        'cm': 0,
        'ck': 0,
        'cy': 0,

        'mc': 0,
        'mm': 0,
        'mk': 0,
        'my': 0,

        'kc': 0,
        'km': 0,
        'kk': 0,
        'ky': 0,

        'yc': 0,
        'ym': 0,
        'yk': 0,
        'yy': 0,
    }
    for base in base_class:
        for con_row in confusion_matrix_material:
            if base == con_row[0]:               # 비교하고자 하는 대상이 base가 맞는 경우,
                if base in con_row[1]:           # cc
                    confusion_matrix[base+base] += 1
                elif base not in con_row[1]:     # cm
                    if 'c' in con_row[1]:
                        confusion_matrix[base+'c'] += 1
                    elif 'm' in con_row[1]:
                        confusion_matrix[base+'m'] += 1     # base가 c인데 predict로 m 인 경우
                    elif 'k' in con_row[1]:
                        confusion_matrix[base+'k'] += 1
                    elif 'y' in con_row[1]:
                        confusion_matrix[base+'y'] += 1
    return confusion_matrix

def MahalanovisCalcDistance(centroid, input_data, gausian_num):
    cov_list = []
    mean_list = []

    for group in centroid:
        mean_list.append(group[0])
        cov_list.append(group[1])

    mahal_list = []
    if gausian_num == 1:
        for cov, mean in zip(cov_list, mean_list):
            x_mu = input_data - mean
            mahal = np.dot(x_mu, np.linalg.inv(cov))
            mahal = np.dot(mahal, np.transpose(x_mu))
            mahal_list.append(mahal)

    elif gausian_num == 2:
        for cov, mean in zip(cov_list, mean_list):
            for c, m in zip(cov, mean):
                x_mu = input_data - m
                mahal = np.dot(x_mu, np.linalg.inv(c))
                mahal = np.dot(mahal, np.transpose(x_mu))
                mahal_list.append(mahal)

    return mahal_list

if __name__ == '__main__':
    # Data pre-processing
    coordinate_list, coo_target_list = open_coordinate_data()
    matrix_list, mat_target_list = open_matrix()

    # Model
    # model = Make_Sequential_model(show_summary=True)
    acc_list = []
    conf_list = []
    k = 20
    gausian_num = 1

    # test : 25
    for re in range(1):
        x_train, y_train, y_train_color_marker, x_test, y_test, y_test_color_marker, x_valid, y_valid = \
            processing_dataset(coordinate_list, matrix_list, coo_target_list)

        # show_img_together(y_train, y_train_color_marker)
        centroid = find_Centroid(y_train, y_train_color_marker, gausian_num)

        model.fit(x_train, y_train, epochs=50, batch_size=50, validation_data=(x_valid, y_valid), validation_batch_size=50)

        # Data processing
        predict_train = model.predict(x_train)
        predict_test = model.predict(x_test)

        # # Visualize
        # image_show_and_compare(y_train, predict_train, y_train_color_marker)

        check_list = {
            'success' : 0,
            'failure' : 0,
        }

        confusion_matrix_material = []

        for input_data, input_data_target in zip(predict_test, y_test_color_marker[:,0]):

            # # KNN ------------------------
            # result = KNN_processing(y_train, y_train_color_marker[:,0], input_data, K=k)
            # predict_result = [k for k, v in result.items() if max(result.values()) == v]
            # if input_data_target in predict_result:
            #     check_list['success'] += 1
            # else:
            #     check_list['failure'] += 1
            # # Visualize Processing
            # print("Answer :", input_data_target, ", Predict :", predict_result, result)
            # show_img_together(y_train, y_train_color_marker, input_data)
            #
            # confusion_matrix_material.append([input_data_target, predict_result])

            # Mahalanovis distance

            result = MahalanovisCalcDistance(centroid, input_data, gausian_num)
            min_result = min(result)

            for i, d_result in enumerate(result):
                if d_result == min_result :
                    min_index = i
                    break

            if gausian_num == 1:
                if min_index < 1 : predict_result = 'c'
                elif min_index < 2: predict_result = 'm'
                elif min_index < 3: predict_result = 'k'
                elif min_index < 4: predict_result = 'y'

            elif gausian_num == 2:
                if min_index < 2 : predict_result = 'c'
                elif min_index < 4: predict_result = 'm'
                elif min_index < 6: predict_result = 'k'
                elif min_index < 8: predict_result = 'y'

            if predict_result == input_data_target:
                check_list['success'] += 1
            else:
                check_list['failure'] += 1

            # Visualize Processing
            # print("Answer :", input_data_target, ", Predict :", predict_result, result)
            # show_img_together(y_train, y_train_color_marker, input_data)

            confusion_matrix_material.append([input_data_target, predict_result])

        confusion_matrix = processing_confusionmatrix(confusion_matrix_material)

        # print(check_list)
        acc_list.append(check_list['success'] / sum(check_list.values()))
        conf_list.append(confusion_matrix)
        # print("Accuracy : ", check_list['success'] / sum(check_list.values()))

    # print("Accuracy")
    # for acc in acc_list:
    #     print(acc)

    print()
    print("Confusion Matrix")
    for conf_dict in conf_list:
        print(conf_dict['cc'], conf_dict['cm'], conf_dict['ck'], conf_dict['cy'])
        print(conf_dict['mc'], conf_dict['mm'], conf_dict['mk'], conf_dict['my'])
        print(conf_dict['kc'], conf_dict['km'], conf_dict['kk'], conf_dict['ky'])
        print(conf_dict['yc'], conf_dict['ym'], conf_dict['yk'], conf_dict['yy'])
        print()
