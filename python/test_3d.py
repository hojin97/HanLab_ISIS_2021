import pandas as pd
import os
from PIL import Image
import PIL
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE

def load_img():
    input_base_dir = os.getcwd() + "/png"
    data_type_1 = ["AAFT_0"]  # "AAFT_0", "AAFT_2", "AAFT_3", "AAFT_4", "AAFT_5", "AAFT_10", "AAFT_20"
    data_type_2 = ["hadamard"]   # "hadamard", "striping_color", "striping_midAng_color"
    data_type_3 = ["Bi", "Hammer", "Rvcurl", "Tri"]
    img_dataset_r = []
    img_dataset_g = []
    img_dataset_b = []
    img_target = []
    imgSize = 227

    for d1 in data_type_1:
        for d2 in data_type_2:
            label = 0
            for d3 in data_type_3:      # 1 : Bi, 2 : Ham, 3 : Rvc, 4 : Tri
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

img_dataset_r, img_dataset_g, img_dataset_b, img_target = load_img()
#print(img_target)
tsne = TSNE(random_state=0, n_components=3, perplexity=30)

tsne_results_r = tsne.fit_transform(img_dataset_r)
print(tsne_results_r)
print('*****')
print(tsne_results_r[:,0])
print('*****')
print(tsne_results_r[:,1])
print('*****')
print(tsne_results_r[:,2])
print('*****')
#tsne_results_r = pd.DataFrame({'tsne1':tsne_results_r[:,0], 'tsne2':tsne_results_r[:1], 'tsne3':tsne_results_r[:2]})
#print(tsne_results_r)
fig1 = plt.figure(1)
ax = fig1.gca(projection='3d')
#ax.title('R channel of image, TSNE')
#plt.axes(projection = "3d")
ax.scatter(tsne_results_r[:,0], tsne_results_r[:,1], tsne_results_r[:,2], c=img_target)



# tsne_results_g = tsne.fit_transform(img_dataset_g)
# tsne_results_g = pd.DataFrame(tsne_results_g, columns=['tsne1', 'tsne2'])
# plt.figure(2)
# plt.title('G channel of image, TSNE')
# plt.scatter(tsne_results_g['tsne1'], tsne_results_g['tsne2'], c=img_target)
#
#
#
# tsne_results_b = tsne.fit_transform(img_dataset_b)
# tsne_results_b = pd.DataFrame(tsne_results_b, columns=['tsne1', 'tsne2'])
# plt.figure(3)
# plt.title('B channel of image, TSNE')
# plt.scatter(tsne_results_b['tsne1'], tsne_results_b['tsne2'], c=img_target)
plt.show()

