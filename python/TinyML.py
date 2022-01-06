from keras.models import Sequential
from keras import layers
from keras.optimizers import SGD
import numpy as np
import os
import matplotlib.pyplot as plt
import math
from scipy.stats import multivariate_normal
from sklearn.mixture import GaussianMixture
import tensorflow as tf

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
    l1 = int(len(DATA_SET)*0.8)
    l2 = l1 + int((len(DATA_SET)-l1)*0.5)
    train = DATA_SET[0:l1]             # 0.8
    valid = DATA_SET[l1:l2]            # 0.1
    test = DATA_SET[l2:]               # 0.1

    x_train, y_train, y_train_color_marker = separate_DATA(train, "color")
    x_valid, y_valid = separate_DATA(valid)
    x_test, y_test, y_test_color_marker = separate_DATA(test, "color")

    return x_train, y_train, y_train_color_marker, x_test, y_test, y_test_color_marker, x_valid, y_valid

def image_show_and_compare(data1, data2, data3, data4, d_color):
    plt.figure(1)
    plt.title("Target")
    plt.scatter(data1[:, 0], data1[:, 1], c=d_color)

    plt.figure(2)
    plt.title("Predict")
    plt.scatter(data2[:, 0], data2[:, 1], c=d_color)

    plt.figure(3)
    plt.title("tflite model")
    plt.scatter(data3[:, 0], data3[:, 1], c=d_color)

    plt.figure(4)
    plt.title("tflite quantized")
    plt.scatter(data4[:, 0], data4[:, 1], c=d_color)

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

def processing_dataset(coordinate_list, matrix_list, coo_target_list, random=False):
    DATA_SET = combine_data(coordinate_list, matrix_list, coo_target_list)

    if random :
        np.random.shuffle(DATA_SET)       # random training, test set을 만들기 위함.

    x_train, y_train, y_train_color_marker, x_test, y_test, y_test_color_marker, x_valid, y_valid = separate_DATA_SET(DATA_SET)

    return x_train, y_train, y_train_color_marker, x_test, y_test, y_test_color_marker, x_valid, y_valid

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

    for key in each_points.keys():
        each_points[key] = np.array(each_points[key])
        gmm = GaussianMixture(n_components=gausian_num, random_state=0).fit(each_points[key])

        centroid.append([gmm.means_, gmm.covariances_])
        # show_img_gaussian_distribution(scatter_data, scatter_target, each_points[key], gmm)
    return centroid

def show_img_gaussian_distribution(scatter_data, scatter_target, point_data, gmm):
    x_min, x_max = point_data[:, 0].min(), point_data[:, 0].max()
    y_min, y_max = point_data[:, 1].min(), point_data[:, 1].max()
    x, y = np.meshgrid(np.linspace(x_min, x_max, num=10), np.linspace(y_min, y_max, num=10))
    pos = np.dstack((x,y))
    z = multivariate_normal.pdf(pos, mean=gmm.means_[0], cov=gmm.covariances_[0])
    plt.figure(7)
    plt.contour(x, y, z, 5, alpha=1, linewidths=1.1)
    for x, y, color, mark in zip(scatter_data[:,0], scatter_data[:,1], scatter_target[:,0], scatter_target[:,1]):
        plt.scatter(x, y, c=color, marker=mark)
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

def representative_dataset_generator():
    for value in x_test:
        yield [np.array(value, dtype=np.float32, ndmin=2)]

if __name__ == '__main__':
    # Data pre-processing
    coordinate_list, coo_target_list = open_coordinate_data()
    matrix_list, mat_target_list = open_matrix()

    # Model
    model = Make_Sequential_model(show_summary=True)
    acc_list = []
    conf_list = []
    gausian_num = 1
    x_train, y_train, y_train_color_marker, x_test, y_test, y_test_color_marker, x_valid, y_valid = \
        processing_dataset(coordinate_list, matrix_list, coo_target_list, random=True)

    # show_img_together(y_train, y_train_color_marker)
    # centroid = find_Centroid(y_train, y_train_color_marker, gausian_num)

    model.fit(x_train, y_train, epochs=50, batch_size=50, validation_data=(x_valid, y_valid), validation_batch_size=50)

    # 양자화 없이 모델 -> 텐서플로 라이트 형식으로 변환
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    open("TSNE.tflite", "wb").write(tflite_model)

    # 양자화하여 모델 -> 텐서플로 라이트 형식으로 변환
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset_generator

    tflite_model = converter.convert()
    open("TSNE_quantized.tflite", "wb").write(tflite_model)

    ##

    # 1. 인터프리터 객체 인스턴스화
    tsne_model = tf.lite.Interpreter('TSNE.tflite')
    tsne_model_quantized = tf.lite.Interpreter('TSNE_quantized.tflite')

    # 2. 모델에 메모리 할당하기.
    tsne_model.allocate_tensors()
    tsne_model_quantized.allocate_tensors()

    # 3. 입력 텐서에 입력 값 작성
    tsne_model_input_index = tsne_model.get_input_details()[0]["index"]
    tsne_model_output_index = tsne_model.get_output_details()[0]["index"]

    tsne_model_quantized_input_index = tsne_model_quantized.get_input_details()[0]["index"]
    tsne_model_quantized_output_index = tsne_model_quantized.get_output_details()[0]["index"]

    tsne_model_predictions = []
    tsne_model_quantized_predictions = []

    for x_value in x_train:
        x_value_tensor = tf.convert_to_tensor([x_value], dtype=np.float32)

        tsne_model.set_tensor(tsne_model_input_index, x_value_tensor)
        tsne_model.invoke()
        tsne_model_predictions.append(tsne_model.get_tensor(tsne_model_output_index)[0])

        tsne_model_quantized.set_tensor(tsne_model_quantized_input_index, x_value_tensor)
        tsne_model_quantized.invoke()
        tsne_model_quantized_predictions.append(tsne_model_quantized.get_tensor(tsne_model_quantized_output_index)[0])

    predictions = model.predict(x_train)
    tsne_model_predictions = np.array(tsne_model_predictions)
    tsne_model_quantized_predictions = np.array(tsne_model_quantized_predictions)

    image_show_and_compare(y_train, predictions, tsne_model_predictions, tsne_model_quantized_predictions, y_train_color_marker[:,0])

    basic_model_size = os.path.getsize("TSNE.tflite")
    quantized_model_size = os.path.getsize("TSNE_quantized.tflite")
    difference = basic_model_size - quantized_model_size

    print("Basic model ",basic_model_size)
    print("Quantized model ", quantized_model_size)
    print("Difference ", difference)

    # Data processing
    # predict_train = model.predict(x_train)
    # predict_test = model.predict(x_test)

    # # Visualize
    # image_show_and_compare(y_train, predict_train, y_train_color)

    # check_list = {
    #     'success' : 0,
    #     'failure' : 0,
    # }

    # for input_data, input_data_target in zip(predict_test, y_test_color_marker[:,0]):
    #
    #     # Mahalanovis distance
    #
    #     result = MahalanovisCalcDistance(centroid, input_data, gausian_num)
    #     min_result = min(result)
    #
    #     for i, d_result in enumerate(result):
    #         if d_result == min_result :
    #             min_index = i
    #             break
    #
    #     if gausian_num == 1:
    #         if min_index < 1 : predict_result = 'c'
    #         elif min_index < 2: predict_result = 'm'
    #         elif min_index < 3: predict_result = 'k'
    #         elif min_index < 4: predict_result = 'y'
    #
    #     elif gausian_num == 2:
    #         if min_index < 2 : predict_result = 'c'
    #         elif min_index < 4: predict_result = 'm'
    #         elif min_index < 6: predict_result = 'k'
    #         elif min_index < 8: predict_result = 'y'
    #
    #     if predict_result == input_data_target:
    #         check_list['success'] += 1
    #     else:
    #         check_list['failure'] += 1
    #
    #     # Visualize Processing
    #     # print("Answer :", input_data_target, ", Predict :", predict_result, result)
    #     # show_img_together(y_train, y_train_color_marker, input_data)