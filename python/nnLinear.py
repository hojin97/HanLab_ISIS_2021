import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np

def open_coordinate_data():
    file_path = os.getcwd() + '/embedding_vecs'
    file_name = os.listdir(file_path)
    fp = os.path.join(file_path, file_name[0])      # 현재는 1개뿐이라,,
    file = open(fp, "r")

    coordinate_list = []
    target_list = []

    for line in file.readlines():
        data = line.split()
        xy_pos = np.array([float(data[0]),  float(data[1])]).flatten()
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

coordinate_list, coo_target_list = open_coordinate_data()
matrix_list, mat_target_list = open_matrix()
print()