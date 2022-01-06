import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
import os

from scipy.stats import multivariate_normal
from sklearn.mixture import GaussianMixture
# import pylab as plt

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

def divide_group(coordinate_list, coo_target_list):
    each_points = {
        'bc': [],
        'hc': [],
        'rv': [],
        'tc': [],
    }

    for data, target in zip(coordinate_list, coo_target_list):
        each_points[target].append(data)

    for key in each_points.keys():
        each_points[key] = np.array(each_points[key])

    return each_points

def twoD_Gaussian(x, y, amplitude, xo, yo, sigma_x, sigma_y, theta, offset):
    xo = float(xo)
    yo = float(yo)
    a = (np.cos(theta) ** 2) / (2 * sigma_x ** 2) + (np.sin(theta) ** 2) / (2 * sigma_y ** 2)
    b = -(np.sin(2 * theta)) / (4 * sigma_x ** 2) + (np.sin(2 * theta)) / (4 * sigma_y ** 2)
    c = (np.sin(theta) ** 2) / (2 * sigma_x ** 2) + (np.cos(theta) ** 2) / (2 * sigma_y ** 2)
    return offset + amplitude * np.exp(- (a * ((x - xo) ** 2) + 2 * b * (x - xo) * (y - yo) + c * ((y - yo) ** 2)))

coordinate_list, coo_target_list = open_coordinate_data()
matrix_list, mat_target_list = open_matrix()

# rand1 = np.random.normal(loc=1, scale=2, size=1000)
coordinate_list = np.array(coordinate_list)

each_group = divide_group(coordinate_list, coo_target_list)

# GaussianMixture model
gmm = GaussianMixture(n_components=1, random_state=0).fit(each_group['bc'])
print(gmm.means_)
print(gmm.covariances_)
x, y = np.meshgrid(each_group['bc'][:,0], each_group['bc'][:,1])
pos = np.dstack((x, y))
z = multivariate_normal.pdf(pos, mean=gmm.means_[0], cov=gmm.covariances_[0])

plt.figure(1)
plt.contourf(x, y, z, levels=5)
plt.scatter(coordinate_list[:,0], coordinate_list[:,1])

plt.show()

# # Visualization
# for key in each_group.keys():
#     plt.figure(figsize=(5,3))
#     plt.title(key)
#     d = sns.kdeplot(data=each_group[key])
# plt.show()

# # Normality Test.
# for key in each_group.keys():
#     test_stat, p_val = stats.shapiro(each_group[key][:, 0])
#     print("[{}] - x, Test-statistics : {}, p-value : {}".format(key, test_stat, p_val))
#     test_stat, p_val = stats.shapiro(each_group[key][:, 1])
#     print("[{}] - y, Test-statistics : {}, p-value : {}".format(key, test_stat, p_val))
#     print()

# # Box plot
# for key in each_group.keys():
#     plt.figure()
#     plt.title(key)
#     sns.boxplot(data=each_group[key], orient='h')
# plt.show()