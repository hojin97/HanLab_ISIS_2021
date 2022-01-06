import pandas as pd
import os
from PIL import Image
import PIL
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE


def load_img_gray(type=1):
    input_base_dir = os.getcwd() + "/png"
    data_type_1 = ["AAFT_0"]  # "AAFT_0", "AAFT_2", "AAFT_3", "AAFT_4", "AAFT_5", "AAFT_10", "AAFT_20"
    if type==1:
        data_type_2 = ["I_easy_gray"]  # "I_easy_gray", "I_fair_gray"
    elif type == 2:
        data_type_2 = ["I_fair_gray"]  # "I_easy_gray", "I_fair_gray"
    data_type_3 = ["Bi", "Hammer", "Rvcurl", "Tri"]

    img_dataset = []
    img_target = []
    imgSize = 227

    for d1 in data_type_1:
        for d2 in data_type_2:
            label = 0
            for d3 in data_type_3:  # r : Bi, g : Ham, b : Rvc, y : Tri
                if d3 == "Bi":
                    label = 'r'
                elif d3 == "Hammer":
                    label = 'g'
                elif d3 == "Rvcurl":
                    label = 'b'
                elif d3 == "Tri":
                    label = 'y'

                img_path = os.path.join(input_base_dir, d1, d2, d3)
                imgs = os.listdir(img_path)
                for img_index in imgs:
                    img = os.path.join(img_path, img_index)
                    img_data = Image.open(img).convert('L')
                    img_data = img_data.resize((imgSize, imgSize), PIL.Image.ANTIALIAS)
                    img_data = (np.array(img_data))
                    img_data = img_data.flatten()

                    img_dataset.append(img_data)
                    img_target.append(label)


    return img_dataset, img_target

def load_img():
    input_base_dir = os.getcwd() + "/png"
    data_type_1 = ["AAFT_0"]  # "AAFT_0", "AAFT_2", "AAFT_3", "AAFT_4", "AAFT_5", "AAFT_10", "AAFT_20"
    data_type_2 = ["striping_midAng_color"]  # "hadamard", "striping_color", "striping_midAng_color"
    data_type_3 = ["Bi", "Hammer", "Rvcurl", "Tri"]

    img_dataset_r = []
    img_dataset_g = []
    img_dataset_b = []
    img_target = []
    imgSize = 227

    for d1 in data_type_1:
        for d2 in data_type_2:
            label = 0
            for d3 in data_type_3:      # r : Bi, g : Ham, b : Rvc, y : Tri
                if d3 == "Bi":   label = 'r'
                elif d3 == "Hammer":    label='g'
                elif d3 == "Rvcurl":    label = 'b'
                elif d3 == "Tri":    label = 'y'
                img_path = os.path.join(input_base_dir, d1, d2, d3)
                imgs = os.listdir(img_path)
                for img_index in imgs:
                    img = os.path.join(img_path, img_index)
                    img_data = Image.open(img)
                    img_data = img_data.resize((imgSize, imgSize), PIL.Image.ANTIALIAS)
                    img_data = (np.array(img_data))

                    r = img_data[:, :, 0].flatten()
                    g = img_data[:, :, 1].flatten()
                    b = img_data[:, :, 2].flatten()

                    img_dataset_r.append(r)
                    img_dataset_g.append(g)
                    img_dataset_b.append(b)
                    img_target.append(label)

    return img_dataset_r, img_dataset_g, img_dataset_b, img_target

# type : 0 => I_fair
# type : 1 => I_easy
def load_vec(type=0):
    if type == 0:
        input_base_dir = os.getcwd() + "/emg_processing/emg_result"
    elif type == 1:
        input_base_dir = os.getcwd() + "/emg_processing/emg_result_midAng"

    data_type_1 = ["AAFT(0)"]  # "AAFT(0)", "AAFT(2)", , "AAFT(3)", "AAFT(4)", "AAFT(5)", "AAFT(10)", "AAFT(20)"
    data_type_2 = ["Bi", "Hammer", "Rvcurl", "Tri"]
    vec_list = []
    vec_target = []

    for d1 in data_type_1:
        label = 0
        for d2 in data_type_2:  # 1 : Bi, 2 : Ham, 3 : Rvc, 4 : Tri
            if d2 == "Bi":   label = 'r'
            elif d2 == "Hammer":    label='g'
            elif d2 == "Rvcurl":    label = 'b'
            elif d2 == "Tri":    label = 'y'
            vec_path = os.path.join(input_base_dir, d1, d2)
            vecs = os.listdir(vec_path)
            for vec_index in vecs:
                vec = os.path.join(vec_path, vec_index)
                vec_data = open(vec, 'r')
                vec_data = vec_data.read().split()

                vec_list.append(vec_data)
                vec_target.append(label)

    return vec_list, vec_target

def save_embbeding_output(tsne_results, vecs_target, type):
    file_path = os.getcwd() + '/embedding_vecs'
    out_txt = os.path.join(file_path, type + "_embedding.txt")
    fp = open(out_txt, "w")
    for i, vecs in enumerate(tsne_results.values):
        if vecs_target[i] == "r":
            target = "bc"
        elif vecs_target[i] == "g":
            target = "hc"
        elif vecs_target[i] == "b":
            target = "rv"
        elif vecs_target[i] == "y":
            target = "tc"
        fp.write(str(vecs[0]) + "\t" + str(vecs[1]) + "\t" + target + "\n")
    fp.close()

DATA_TYPE = "AAFT(0)"
# Vecs *************************************************************
vecs_list, vecs_target = load_vec(0)
vecs_list.pop(0), vecs_target.pop(0)
vecs_list.pop(65), vecs_target.pop(65)
vecs_list.pop(100), vecs_target.pop(100)
vecs_list.pop(0), vecs_target.pop(0)
vecs_list.pop(65), vecs_target.pop(65)
vecs_list.pop(100), vecs_target.pop(100)
vecs_list.pop(0), vecs_target.pop(0)
vecs_list.pop(65), vecs_target.pop(65)
vecs_list.pop(100), vecs_target.pop(100)
tsne = TSNE(random_state=0, perplexity=50)
tsne_results = tsne.fit_transform(vecs_list)
tsne_results = pd.DataFrame(tsne_results, columns=['tsne1', 'tsne2'])
# save_embbeding_output(tsne_results, vecs_target, DATA_TYPE)
plt.figure(1)
plt.title('Vecs, TSNE')
plt.scatter(tsne_results['tsne1'], tsne_results['tsne2'], c=vecs_target)
plt.show()
#
# Color image *************************************************************
# img_dataset_r, img_dataset_g, img_dataset_b, img_target = load_img(type="color")
# tsne = TSNE(random_state=0, n_components=2, perplexity=50)
#
# tsne_results_r = tsne.fit_transform(img_dataset_r)
# tsne_results_r = pd.DataFrame(tsne_results_r, columns=['tsne1', 'tsne2'])
# plt.figure(1)
# plt.title('R channel of image, TSNE')
# plt.scatter(tsne_results_r['tsne1'], tsne_results_r['tsne2'], c=img_target)
#
# tsne_results_g = tsne.fit_transform(img_dataset_g)
# tsne_results_g = pd.DataFrame(tsne_results_g, columns=['tsne1', 'tsne2'])
# plt.figure(2)
# plt.title('G channel of image, TSNE')
# plt.scatter(tsne_results_g['tsne1'], tsne_results_g['tsne2'], c=img_target)
#
# tsne_results_b = tsne.fit_transform(img_dataset_b)
# tsne_results_b = pd.DataFrame(tsne_results_b, columns=['tsne1', 'tsne2'])
# plt.figure(3)
# plt.title('B channel of image, TSNE')
# plt.scatter(tsne_results_b['tsne1'], tsne_results_b['tsne2'], c=img_target)
# plt.show()


# GRAY image *************************************************************
# img_dataset, img_target = load_img_gray(1)
# tsne = TSNE(random_state=0, perplexity=50)
#
# tsne_results = tsne.fit_transform(img_dataset)
# tsne_results = pd.DataFrame(tsne_results, columns=['tsne1', 'tsne2'])
# print(tsne_results)
# plt.figure(1)
# plt.title('I_easy GRAY image, TSNE')
# plt.scatter(tsne_results['tsne1'], tsne_results['tsne2'], c=img_target)
#
#
# img_dataset, img_target = load_img_gray(2)
# tsne_results = tsne.fit_transform(img_dataset)
# tsne_results = pd.DataFrame(tsne_results, columns=['tsne1', 'tsne2'])
# plt.figure(2)
# plt.title('I_fair GRAY image, TSNE')
# plt.scatter(tsne_results['tsne1'], tsne_results['tsne2'], c=img_target)
#
# plt.show()
