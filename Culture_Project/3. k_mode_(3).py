from kmodes.kmodes import KModes
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tqdm # 작업진행률 표시

df = pd.read_csv("final_data.csv", encoding="euc-kr")

Exercise_PRSCRPTN = df['Exercise_PRSCRPTN']
df = df.drop(columns=['Exercise_PRSCRPTN'])

# # 최적의 K값 찾기
# cost = []
# K = range(1, 101)
# for num in tqdm.tqdm(K):
#     kmode = KModes(n_clusters = num, init='Cao')
#     kmode.fit_predict(df)
#     cost.append(kmode.cost_)
# plt.plot(K, cost, 'bx-', color='#40E0D0')
# plt.xlabel('number of clusters')
# plt.ylabel('cost')
# plt.title('elbow method for finding optimal K')
# plt.show()

kmode = KModes(n_clusters = 60, init='Cao')
clusters = kmode.fit_predict(df)
df['clusters'] = clusters
df["Exercise_PRSCRPTN"]=Exercise_PRSCRPTN
df["Exercise_PRSCRPTN"] = df["Exercise_PRSCRPTN"].astype(str)

df.to_csv('df_clustered_kmode.csv',index=False,encoding="euc-kr")