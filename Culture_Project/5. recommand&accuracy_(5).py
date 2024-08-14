import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_csv('seperate_exercise.csv', encoding='euc-kr', index_col=0)
prep_exercises_matrix = pd.read_csv('prep_exercises_matrix.csv', encoding='cp949', index_col=0)
main_exercises_matrix = pd.read_csv('main_exercises_matrix.csv', encoding='cp949', index_col=0)
cool_exercises_matrix = pd.read_csv('cool_exercises_matrix.csv', encoding='cp949', index_col=0)

# 최적의 잠재요인수(n_components) 찾기
def find_n_components(user_item_matrix):
    svd = TruncatedSVD(n_components=60, random_state=42)
    svd.fit(user_item_matrix)

    explained_variance = svd.explained_variance_ratio_
    cumulative_explained_variance = np.cumsum(explained_variance)

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(cumulative_explained_variance) + 1), cumulative_explained_variance, marker='o')
    plt.title('Cumulative Explained Variance by Number of Components')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.grid(True)
    plt.show()
# find_n_components(prep_exercises_matrix) # 약 95% 기준으로 3개의 component 선택 
# find_n_components(main_exercises_matrix) # 약 95% 기준으로 3개의 component 선택 
# find_n_components(cool_exercises_matrix) # 약 95% 기준으로 3개의 component 선택 

# User(Cluster)/Item latent matrix(잠재 요인 행렬) 생성 : TruncatedSVD 적용
def latent_matrix(user_item_matrix):
    svd = TruncatedSVD(n_components=3, random_state=42)
    U = svd.fit_transform(user_item_matrix)
    Vt = svd.components_.T
    return U, Vt

# Cluster 잠재요인행렬과 Item 잠재요인행렬 생성
prep_cluster_latent_matrix, prep_item_latent_matrix = latent_matrix(prep_exercises_matrix)
main_cluster_latent_matrix, main_item_latent_matrix = latent_matrix(main_exercises_matrix)
cool_cluster_latent_matrix, cool_item_latent_matrix = latent_matrix(cool_exercises_matrix)

# 각 운동별 데이터 딕셔너리 생성
prep_data = {'item_user_matrix': prep_exercises_matrix, 'cluster_latent_matrix': prep_cluster_latent_matrix, 'item_latent_matrix': prep_item_latent_matrix}
main_data = {'item_user_matrix': main_exercises_matrix, 'cluster_latent_matrix': main_cluster_latent_matrix, 'item_latent_matrix': main_item_latent_matrix}
cool_data = {'item_user_matrix': cool_exercises_matrix, 'cluster_latent_matrix': cool_cluster_latent_matrix, 'item_latent_matrix': cool_item_latent_matrix}

# 원-핫 인코딩
encoder = OneHotEncoder()
sex_encoded = encoder.fit_transform(df[['SEX']]).toarray()
encoded_columns = encoder.get_feature_names_out(['SEX'])
df_encoded = df.drop(columns=['SEX']).join(pd.DataFrame(sex_encoded, index=df.index, columns=encoded_columns))

# 사용자 특성 데이터 표준화 - 클러스터 중심 계산
columns_to_drop = ['clusters', '준비운동', '본운동', '마무리운동']
user_features = df_encoded.drop(columns=columns_to_drop)
scaler = StandardScaler()
user_features_scaled = scaler.fit_transform(user_features)

# 동일한 스케일로 클러스터 중심 계산
cluster_centers = user_features.groupby(df['clusters']).mean()
cluster_centers_scaled = scaler.transform(cluster_centers)

## 새로운 사용자에게 추천

# 사용자 입력
age = int(input("나이를 입력하세요: "))
sex = int(input("성별을 입력하세요 (남성: 1, 여성: 0): "))
height = float(input("키를 입력하세요 (cm): "))
weight = float(input("체중을 입력하세요 (kg): "))
bmi = weight/((height/100)**2)
body_fat = float(input("체지방률을 입력하세요 (%) (유소년인 경우 0을 입력하세요): "))
waist = float(input("허리둘레를 입력하세요 (cm): "))
right_grip_strength = float(input("오른손 악력을 입력하세요: "))
left_grip_strength=float(input("왼손 악력을 입력하세요: "))
Low_BP = float(input("이완기최저혈압: "))
High_BP = float(input("수축기최고혈압: "))
grip_strength=(max(right_grip_strength, left_grip_strength)/ weight) * 100

def categorize_age(AGE):
    if AGE < 15:
        return 0
    elif AGE < 20:
        return 1
    elif AGE < 25:
        return 2
    elif AGE < 30:
        return 3
    elif AGE < 35:
        return 4
    elif AGE < 40:
        return 5
    elif AGE < 50:
        return 6
    elif AGE < 65:
        return 7
    else:
        return 8

def categorize_height(Height):
    if Height < 140:
        return 0
    elif Height < 150:
        return 1
    elif Height < 160:
        return 2
    elif Height < 170:
        return 3
    elif Height < 180:
        return 4
    else:
        return 5

def categorize_weight(Weight):
    if Weight < 40:
        return 0
    elif Weight < 60:
        return 1
    elif Weight < 80:
        return 2
    else:
        return 3

def categorize_bmi(BMI):
    if BMI < 18.4:
        return 0
    elif BMI < 22.9:
        return 1
    elif BMI < 24.9:
        return 2
    elif BMI < 29.9:
        return 3
    else:
        return 4

def categorize_bodyfat(Body_Fat):
    if Body_Fat < 1:
        return 0
    elif Body_Fat < 15:
        return 1
    elif Body_Fat < 30:
        return 2
    else:
        return 3

def categorize_waist(Waist):
    if Waist < 60:
        return 0
    elif Waist < 70:
        return 1
    elif Waist < 80:
        return 2
    elif Waist < 90:
        return 3
    else:
        return 4

# low_bp/high_bp 칼럼
def categorize_blood_pressure(low_bp, high_bp):
    if low_bp < 60 or high_bp < 90:
        return 0  # 저혈압
    elif low_bp >= 90 or high_bp >= 140:
        return 2  # 고혈압
    else:
        return 1  # 정상 혈압

def categorize_grap_strength(Relative_Grap_strength):
    if Relative_Grap_strength < 30:
        return 0
    elif Relative_Grap_strength < 50:
        return 1
    elif Relative_Grap_strength < 70:
        return 2
    else:
        return 3

# 새로운 사용자의 특성 데이터프레임 생성
new_user = pd.DataFrame({
    'AGE': [age],
    'SEX': [sex],
    'Height': [height],
    'Weight': [weight],
    'BMI': [bmi],
    'Body_Fat': [body_fat],
    'Waist': [waist],
    'Relative_Grap_strength': [grip_strength],
    'Low_BP': [Low_BP],
    'High_BP': [High_BP]
})

# train 데이터와 동일한 전처리 수행 : 범주화, 칼럼 재배열, 표준화
new_user['AGE'] = new_user['AGE'].apply(categorize_age)
new_user['Height'] = new_user['Height'].apply(categorize_height)
new_user['Weight'] = new_user['Weight'].apply(categorize_weight)
new_user['BMI'] = new_user['BMI'].apply(categorize_bmi)
new_user['Body_Fat'] = new_user['Body_Fat'].apply(categorize_bodyfat)
new_user['Waist'] = new_user['Waist'].apply(categorize_waist)
new_user['Blood_Pressure_Status'] = new_user.apply(lambda row: categorize_blood_pressure(row['Low_BP'], row['High_BP']), axis=1)
new_user['Relative_Grap_strength'] = new_user['Relative_Grap_strength'].apply(categorize_grap_strength)

new_user = new_user.drop(columns=['Low_BP','High_BP'])
col = new_user.pop('Blood_Pressure_Status')
new_user.insert(7, 'Blood_Pressure_Status', col)

new_user_encoded = new_user.drop(columns=['SEX']).join(pd.DataFrame(encoder.transform(new_user[['SEX']]).toarray(), columns=encoded_columns))
new_user_scaled = scaler.transform(new_user_encoded)

# 가장 유사한 클러스터 찾기 : 유사도 계산(코사인 유사도)
similarities = cosine_similarity(new_user_scaled, cluster_centers_scaled)
most_similar_cluster = similarities.argmax()
#print(f"가장 유사한 클러스터: {most_similar_cluster}")

# 운동 추천 함수
def recommand_exercise(dict):
    # 새로운 사용자의 잠재 요인 벡터 계산
    new_user_latent_vector = dict['cluster_latent_matrix'][most_similar_cluster].reshape(1, -1)

    # 유사도 계산 (코사인 유사도)
    similarities = cosine_similarity(new_user_latent_vector, dict['item_latent_matrix'])

    # 상위 5개의 유사한 운동 처방
    top_5_similar_exercise_indices = similarities.argsort()[0, -5:][::-1]
    top_5_similar_exercises_prep = [dict['item_user_matrix'].columns[idx] for idx in top_5_similar_exercise_indices]
    return top_5_similar_exercises_prep

# 운동 추천
top_5_similar_exercises_prep = recommand_exercise(prep_data)
top_5_similar_exercises_main = recommand_exercise(main_data)
top_5_similar_exercises_cool = recommand_exercise(cool_data)

print("준비운동 추천: ", top_5_similar_exercises_prep)
print("본운동 추천: ", top_5_similar_exercises_main)
print("마무리운동 추천: ", top_5_similar_exercises_cool)


# 모델 평가
# 클러스터별 랜덤 10명이 실제 처방받은 운동 데이터프레임 생성 : 19번 클러스터에는 준비/본/마무리운동을 모두 5개 이상씩 처방받은 사용자가 적으므로 분리
df_19 = df[df['clusters'] == 19]
df_not_19 = df[df['clusters'] != 19]

def check_five_values(value):
    return len(str(value).split(',')) >= 5
def check_four_values(value):
    return len(str(value).split(',')) >= 4
filtered_df = df_not_19[df_not_19.apply(lambda row: check_five_values(row['준비운동']) and check_five_values(row['본운동']) and check_five_values(row['마무리운동']), axis=1)]
filtered_19_df = df_19[df_19.apply(lambda row: check_four_values(row['준비운동']) and check_five_values(row['본운동']) and check_five_values(row['마무리운동']), axis=1)]
combined_df = pd.concat([filtered_df, filtered_19_df], ignore_index=True)
combined_df = combined_df.sort_values(by='clusters')
combined_df = combined_df.iloc[:, -4:]

def expand_exercises(df, column_name, prefix):
    split_exercises = df[column_name].str.split(',', expand=True).apply(lambda x: [item.strip() if isinstance(item, str) else '' for item in x])
    split_exercises = split_exercises.apply(pd.Series)
    split_exercises.columns = [f'{prefix}{i+1}' for i in range(split_exercises.shape[1])]
    return split_exercises

# 준비운동, 본운동, 마무리운동 확장
prep_exercises = expand_exercises(combined_df, '준비운동', 'PREP')
main_exercises = expand_exercises(combined_df, '본운동', 'MAIN')
cool_exercises = expand_exercises(combined_df, '마무리운동', 'COOL')

# 원래의 데이터프레임에 확장된 운동 데이터 추가
combined_df = combined_df.drop(columns=['준비운동', '본운동', '마무리운동'])
df_expanded = pd.concat([combined_df, prep_exercises, main_exercises, cool_exercises], axis=1)

# 클러스터별로 10명씩 랜덤으로 선택
grouped = df_expanded.groupby('clusters')
true_df = grouped.apply(lambda x: x.sample(min(len(x), 10), random_state=42)).reset_index(drop=True)
true_df = true_df.replace('', np.nan)
true_df = true_df.dropna(axis=1, how='all')
#print(true_df)

# 클러스터별 추천 운동 처방 데이터프레임 생성
def create_recommendation_df(prefix, cluster_latent, exercise_latent, user_item_matrix):
    result = []
    for most_similar_cluster in range(60):
        # 잠재 요인 벡터 계산 : 유사도 계산 (코사인 유사도)
        new_user_latent_vector = cluster_latent[most_similar_cluster].reshape(1, -1)
        similarities = cosine_similarity(new_user_latent_vector, exercise_latent)

        # 상위 5개의 유사한 운동 처방 찾기
        top_5_similar_exercise_indices = similarities.argsort()[0, -5:][::-1]
        top_5_similar_exercises = [user_item_matrix.columns[idx] for idx in top_5_similar_exercise_indices]
        result.append({
            "clusters": most_similar_cluster,
            f"{prefix}1": top_5_similar_exercises[0],
            f"{prefix}2": top_5_similar_exercises[1],
            f"{prefix}3": top_5_similar_exercises[2],
            f"{prefix}4": top_5_similar_exercises[3],
            f"{prefix}5": top_5_similar_exercises[4],
        })
    return pd.DataFrame(result)
results_df1 = create_recommendation_df('PREP', prep_cluster_latent_matrix, prep_item_latent_matrix, prep_exercises_matrix)
results_df2 = create_recommendation_df('MAIN', main_cluster_latent_matrix, main_item_latent_matrix, main_exercises_matrix)
results_df3 = create_recommendation_df('COOL', cool_cluster_latent_matrix, cool_item_latent_matrix, cool_exercises_matrix)
recommend_df = pd.merge(pd.merge(results_df1, results_df2, on='clusters', how='right'), results_df3, on='clusters', how='right')
# print(recommend_df)

# accuracy 평가
def check_values(true_row, recommend_row):
    true_values = set(true_row[1:])
    recommend_values = set(recommend_row[1:])
    return int(bool(true_values & recommend_values))

result = {
    cluster: [
        check_values(true_row, recommend_row.iloc[0])
        for _, true_row in true_cluster_rows.iterrows()
    ]
    for cluster, true_cluster_rows in true_df.groupby('clusters')
    for recommend_row in [recommend_df[recommend_df['clusters'] == cluster]]
}

accuracy = 0
for val in result.values():
    accuracy += sum(val)
accuracy /= 600
print(f'정확도: {accuracy:.3%}')