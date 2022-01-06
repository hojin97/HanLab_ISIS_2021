import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE

wine = load_wine()      # 13 차원 와인 데이터,,    144 차원 이미지 데이터
df = pd.DataFrame(wine.data, columns=wine.feature_names)    # pandas로 feature_name(target) 별로 나눈다.
print(df)
df = StandardScaler().fit_transform(df)         # 데이터 뜯어와서 값 조정(정규화).

df = pd.DataFrame(df,columns=wine.feature_names)    # pandas 형태로 만들기
print(df)
tsne = TSNE(random_state=0)
tsne_results = tsne.fit_transform(df)
tsne_results=pd.DataFrame(tsne_results, columns=['tsne1', 'tsne2'])
# plt.scatter(tsne_results['tsne1'], tsne_results['tsne2'], c=wine.target)
plt.scatter(tsne_results['tsne1'], tsne_results['tsne2'])
plt.show()