import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("combined_data.csv", encoding='cp949')

# 사용할 칼럼만 저장 및 열이름 변경
df.rename(columns={
    'AGRDE_FLAG_NM': 'AGE_FLAG',
    'MESURE_AGE_CO': 'AGE',
    'SEXDSTN_FLAG_CD': 'SEX',
    'MESURE_IEM_001_VALUE': 'Height',
    'MESURE_IEM_002_VALUE': 'Weight',
    'MESURE_IEM_018_VALUE': 'BMI',
    'MESURE_IEM_003_VALUE': 'Body_Fat',
    'MESURE_IEM_004_VALUE': 'Waist',
    'MESURE_IEM_005_VALUE': 'Low_BP',
    'MESURE_IEM_006_VALUE': 'High_BP',
    'MESURE_IEM_007_VALUE': 'Left_Grap_Strength',
    'MESURE_IEM_008_VALUE': 'Right_Grap_Strength',
    'MESURE_IEM_028_VALUE': 'Relative_Grap_strength',
    'MVM_PRSCRPTN_CN': 'Exercise_PRSCRPTN'
}, inplace=True)

df = df[[
    'AGE_FLAG', 'AGE', 'SEX', 'Height', 'Weight', 'BMI',
    'Body_Fat', 'Waist', 'Low_BP', 'High_BP', 'Left_Grap_Strength',
    'Right_Grap_Strength', 'Relative_Grap_strength','Exercise_PRSCRPTN'
]]

# 유소년 분리 : bodyfat 범주화 목적(0으로)
df_normal = df.loc[df['AGE_FLAG'] != "유소년"].copy()
df_youth = df.loc[df['AGE_FLAG'] == "유소년"].copy()
df_youth['Body_Fat'] = df_youth['Body_Fat'].fillna(0)

# 결측치 처리 및 box-plot 출력
df_normal.dropna(inplace=True)
df_youth.dropna(inplace=True)

def print_boxplot(df):
    numeric_cols = df.select_dtypes(include=[np.number]).iloc[:, 3:]
    plt.figure(figsize=(15, 10))  
    for i, column in enumerate(numeric_cols.columns):
        plt.subplot(2, 4, i + 1)  
        sns.boxplot(y=numeric_cols[column], color='#40E0D0')
        plt.title(f'Boxplot of {column}')
        plt.grid(True)
    plt.tight_layout()
    plt.show()
# print_boxplot(df_normal)

def remove_outliers(df, columns):
    for column in columns:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    return df

columns_to_filter = ['Body_Fat', 'BMI', 'Waist', 'Low_BP', 'High_BP', 'Left_Grap_Strength', 'Right_Grap_Strength', 'Relative_Grap_strength']
df_normal = remove_outliers(df_normal, columns_to_filter)
df_youth = remove_outliers(df_youth, columns_to_filter[1:])
# print_boxplot(df_normal)

# 병합 및 기초통계량 출력
df = pd.concat([df_normal, df_youth], axis=0)
# print(df.describe())

# 범주화
# AGE 칼럼
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
df['AGE'] = df['AGE'].apply(categorize_age)

# SEX 칼럼
def categorize_sex(SEX):
    if SEX == 'F':
        return 0
    else :
        return 1
df['SEX'] = df['SEX'].apply(categorize_sex)

# HEIGHT 칼럼
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
df['Height'] = df['Height'].apply(categorize_height)

# WEIGHT 칼럼
def categorize_weight(Weight):
    if Weight < 40:
        return 0
    elif Weight < 60:
        return 1
    elif Weight < 80:
        return 2
    else:
        return 3
df['Weight'] = df['Weight'].apply(categorize_weight)

# BMI 칼럼
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
df['BMI'] = df['BMI'].apply(categorize_bmi)

# Body_Fat 칼럼
def categorize_bodyfat(Body_Fat):
    if Body_Fat < 1:
        return 0
    elif Body_Fat < 15:
        return 1
    elif Body_Fat < 30:
        return 2
    else:
        return 3
df['Body_Fat'] = df['Body_Fat'].apply(categorize_bodyfat)

# Waist 칼럼
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
df['Waist'] = df['Waist'].apply(categorize_waist)

# low_bp/high_bp 칼럼
def categorize_blood_pressure(low_bp, high_bp):
    if low_bp < 60 or high_bp < 90:
        return 0  # 저혈압
    elif low_bp >= 90 or high_bp >= 140:
        return 2  # 고혈압
    else:
        return 1  # 정상 혈압
df['Blood_Pressure_Status'] = df.apply(lambda row: categorize_blood_pressure(row['Low_BP'], row['High_BP']), axis=1)

# 악력 칼럼
def categorize_grap_strength(Relative_Grap_strength):
    if Relative_Grap_strength < 30:
        return 0
    elif Relative_Grap_strength < 50:
        return 1
    elif Relative_Grap_strength < 70:
        return 2
    else:
        return 3
df['Relative_Grap_strength'] = df['Relative_Grap_strength'].apply(categorize_grap_strength)

# 사용하지 않는 칼럼, 범주형 타입으로 변환
df = df.drop(columns=['AGE_FLAG','Low_BP','High_BP','Left_Grap_Strength','Right_Grap_Strength'])

# 새로운 순서로 데이터프레임 재배열
col = df.pop('Blood_Pressure_Status')
df.insert(7, 'Blood_Pressure_Status', col)

# csv 파일로 저장
df.to_csv('final_data.csv', index=False, encoding='euc-kr')

