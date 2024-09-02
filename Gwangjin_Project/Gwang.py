import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# 데이터 불러오기
pd.set_option('display.max_rows', None)  # 행의 최대 출력 제한 해제
pd.set_option('display.max_columns', None)  # 열의 최대 출력 제한 해제
pd.set_option('display.width', None)
df = pd.read_csv("datafile2.csv",nrows=15, index_col=0, encoding="euc-kr")


# # 안심택배함이 존재하지 않아 이용률이 NaN인 행을 이용률의 평균으로 보간
# df.iloc[:, 13].fillna(df.iloc[:, 13].mean(), inplace=True)

# df1 = df.iloc[:, [0, 1, 2, 9]].copy()
# df2 = df.iloc[:, [3, 4, 10, 11]].copy()
# df3 = df.iloc[:, [5, 6, 7, 8, 13]].copy()

# # 이용객 수가 0인 곳의 지하철 이용객 수 추정
# non_zero_bus = df[df['버스'] != 0]['버스'].mean()
# non_zero_subway = df[df['지하철'] != 0]['지하철'].mean()
# bus_subway_ratio = non_zero_subway / non_zero_bus

# # 버스 이용객 수를 기반으로 지하철 이용객 수 보간
# df['지하철'] = df.apply(lambda x: x['버스'] * bus_subway_ratio if x['지하철'] == 0 else x['지하철'], axis=1)
# df['지하철'] = df['지하철'].astype(int)


# 독립 변수 선택
features = ['인구수', '1인가구수', '여성인구수', 'cctv 개수', '경찰서', '버스', '지하철', '여성 길단위 상존인구', '여성 건물단위 상존인구',
            '2,30대 1인가구 여성인구', '경찰관서 접근 취약인구수']
features1 = features[:3] + [features[9]]
features2 = [features[3], features[4], features[10]]
features3 = features[5:9]

X = df[features1]
Y = df[features2]
Z = df[features3]

# 데이터 표준화
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
Y_scaled = scaler.fit_transform(Y)
Z_scaled = scaler.fit_transform(Z)

# 주성분 분석 결과 확인 함수 만들기
def pca_result_with_coefficients(pca, features):
    explained_variance_ratio = pca.explained_variance_ratio_
    components = pca.components_
    print("주성분 분석 결과:")
    for i, (ratio, component) in enumerate(zip(explained_variance_ratio, components), 1):
        print(f"주성분 {i} 설명 비율: {ratio:.2f}")
        print("주성분에 기여하는 변수 계수:")
        for j, (feature, coefficient) in enumerate(zip(features, component), 1):
            print(f"  {j}. {feature}: {coefficient:.2f}")
        print()

# 주성분 분석 및 결과 출력
pca = PCA()
X_pca = pca.fit_transform(X_scaled)
pca_result_with_coefficients(pca, features1)

Y_pca = pca.fit_transform(Y_scaled)
pca_result_with_coefficients(pca, features2)

Z_pca = pca.fit_transform(Z_scaled)
pca_result_with_coefficients(pca, features3)

# 주성분 분석 수행 결과 주성분 채택 (각각 주성분 1,2만 채택)
X_pca_selected = X_pca[:,:2]
Y_pca_selected = Y_pca[:,:2]
Z_pca_selected = Z_pca[:,:2]
combined_pca_selected = np.concatenate((X_pca_selected, Y_pca_selected, Z_pca_selected), axis=1)


# Elbow Method
sse = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=0, n_init=10)
    kmeans.fit(combined_pca_selected)
    sse.append(kmeans.inertia_)

# SSE에 대한 그래프 그리기
plt.plot(range(1, 11), sse, marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('SSE')
plt.title('Elbow Method - Optimal Number Of Clusters')
plt.show()


# KMeans 모델 학습
kmeans = KMeans(n_clusters=3, random_state=0)  # 엘보우 메소드의 결과로 3개의 클러스터를 설정 / 일관된 값을 얻기 위해 seed 지정
kmeans.fit(combined_pca_selected)

# 클러스터 할당 결과 확인
cluster_labels = kmeans.labels_

# 각 클러스터에 속한 데이터를 추출
clusters = []
for cluster_num in range(3):
    cluster_data = combined_pca_selected[cluster_labels == cluster_num]
    clusters.append(cluster_data)

# 각 군집에 속한 동 출력
for cluster_num in range(3):
    print(f"Cluster {cluster_num}:")
    for idx, label in enumerate(cluster_labels):
        if label == cluster_num:
            print(df.index[idx])


# 산점도 시각화
plt.figure(figsize=(15, 5))

# feature1에 대한 산점도
plt.subplot(1, 3, 1)
for cluster_num in range(3):
    cluster_data = clusters[cluster_num]
    plt.scatter(cluster_data[:, 0], cluster_data[:, 1], label=f'Cluster {cluster_num}')
plt.xlabel('Feature 1 - Principal Component 1')
plt.ylabel('Feature 1 - Principal Component 2')
plt.title('Clustering Results - Feature 1')
plt.legend()


# # 정규성 검정
# from scipy.stats import shapiro

# dong_names = df.index.tolist()

# # 각각의 동 데이터를 이용하여 정규성 검정 수행
# for i, data in enumerate(combined_pca_selected):
#     stat, p_val = shapiro(data)
#     print(f"{dong_names[i]}: Shapiro-Wilk test - statistic = {stat:.2f}, p-value = {p_val:.4f}")


# 등분산성 검정
from scipy.stats import levene

# 동 이름 리스트 (화양동 제외)
dong_names = df.drop("화양동").index.tolist()

# 화양동을 제외한 데이터
other_dong_data = np.concatenate((combined_pca_selected[:6], combined_pca_selected[7:]), axis=0)

# 화양동의 주성분 값 제외
hwangyeongdong_data = combined_pca_selected[6]

# 각각의 동 데이터를 이용하여 등분산성 검정 수행
for i, data in enumerate(other_dong_data):
    statistic, p_value = levene(hwangyeongdong_data, data)
    print(f"{dong_names[i]}: Levene's test - statistic = {statistic:.2f}, p-value = {p_value:.4f}")


# T 검정
from scipy.stats import ttest_ind

# 화양동의 주성분 값
hwangyeongdong_data = combined_pca_selected[6:7]  # 7번째 인덱스

# Cluster0 의 주성분 값
Cluster0_dong_data = np.concatenate((combined_pca_selected[1:2], combined_pca_selected[4:6], combined_pca_selected[7:8], combined_pca_selected[10:14]), axis=0)

# Cluster1 의 주성분 값
Cluster1_dong_data = np.concatenate((combined_pca_selected[0:1], combined_pca_selected[2:4], combined_pca_selected[8:10], combined_pca_selected[14:15]), axis=0)

# 화양동과 Cluster 0 간의 t-검정
for i in range(Cluster0_dong_data.shape[1]):
    statistic, p_value = ttest_ind(hwangyeongdong_data[:, i], Cluster0_dong_data[:, i])
    print(f"주성분 {i+1}: t-statistic = {statistic:.2f}, p-value = {p_value:.4f}")

# 화양동과 Cluster 1 간의 t-검정
for i in range(Cluster1_dong_data.shape[1]):
    statistic, p_value = ttest_ind(hwangyeongdong_data[:, i], Cluster1_dong_data[:, i])
    print(f"주성분 {i+1}: t-statistic = {statistic:.2f}, p-value = {p_value:.4f}")

# 유의수준을 기준으로 p-value가 작은 주성분을 찾아 차이가 유의한 주성분을 확인할 수 있습니다.