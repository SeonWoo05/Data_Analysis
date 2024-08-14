import pandas as pd
df1 = pd.read_csv('data/combined_data_22.csv', encoding='euc-kr')
df2 = pd.read_csv('data/combined_data_23.csv', encoding='euc-kr')
df3 = pd.read_csv('data/combined_data_24.csv', encoding='euc-kr')
df = pd.concat([df1, df2, df3]).reset_index(drop=True)
df = df.drop('strd_yymm', axis=1)

# 설명변수 추가 : X1(기온), X2(강수량)
temp_2022 = pd.read_csv('data/temperature_2022.csv', encoding='euc-kr')
temp_2023 = pd.read_csv('data/temperature_2023.csv', encoding='euc-kr')
temp_2024 = pd.read_csv('data/temperature_2024.csv', encoding='euc-kr')

# '일시'열을 날짜와 시간으로 분리 및 24시간 형식으로 변환
def preprocess_temperature(df):
    df_filtered = df.loc[df['지점'] == 159].copy()
    df_filtered.loc[:, 'date'] = pd.to_datetime(df_filtered['일시'], errors='coerce')
    df_filtered = df_filtered.dropna(subset=['date'])
    df_filtered.loc[:, 'strd_date'] = df_filtered['date'].dt.strftime('%Y%m%d').astype(int)
    df_filtered.loc[:, 'strd_tizn_val'] = df_filtered['date'].dt.hour
    df_filtered.loc[:, 'rainfall'] = df_filtered['강수량(mm)'].fillna(0) # Nan값은 비 안 온 거니까 0으로 보간
    return df_filtered[['strd_date', 'strd_tizn_val', '기온(°C)', 'rainfall']].rename(columns={'기온(°C)': 'temperature'})

temp_2022 = preprocess_temperature(temp_2022)
temp_2023 = preprocess_temperature(temp_2023)
temp_2024 = preprocess_temperature(temp_2024)
temperature_data = pd.concat([temp_2022, temp_2023, temp_2024]).reset_index(drop=True)

merged_data = pd.merge(df, temperature_data, on=['strd_date', 'strd_tizn_val'], how='left')

# 20240313일 14시의 기온 데이터 결측이므로 당일 13시와 15시의 평균값인 12.2로 보간
merged_data.loc[(merged_data['strd_date'] == 20240313) & (merged_data['strd_tizn_val'] == 14), 'temperature'] = 12.2

# 설명변수 추가 : X3(환승역(범주형)) -> 동해선/김해경전철과의 환승역은 비교적 인원이 적으므로 3가지 범주로 나눔
transfer_stations_1 = ['교대', '벡스코', '사상', '거제', '대저']
transfer_stations_2 = ['동래',  '연산', '서면', '수영', '덕천', '미남']
merged_data['is_transfer'] = merged_data['swst_nm'].apply(lambda x: 1 if any(station in x for station in transfer_stations_1) else (2 if any(station in x for station in transfer_stations_2) else 0))

# 설명변수 추가 : X4(공휴일 및 주말(범주형))
holidays = [
    '20220101', '20220131', '20220201', '20220202', '20220301', '20220309',
    '20220505', '20220601', '20220606', '20220815', 
    '20220909', '20220912', '20221003', '20221010',
    '20230123', '20230124', '20230301', '20230515', '20230529', '20230606', '20230815',
    '20230928', '20230929', '20231002', '20231003', '20231009', '20231225',
    '20240101', '20240209', '20240212', '20240301', '20240410', '20240506', '20240515', '20240606'
]

def is_weekend_or_holiday(date):
    date_str = str(date)
    date_obj = pd.to_datetime(date_str, format='%Y%m%d')
    if date_str in holidays or date_obj.weekday() >= 5:
        return 1
    return 0

# 'strd_date' 열에서 주말 또는 공휴일 여부 확인
merged_data['is_holiday'] = merged_data['strd_date'].apply(is_weekend_or_holiday)

# 설명변수 추가 : X5(출퇴근 시간대(범주형))
def is_peak(row):
    if row['is_holiday'] == 0 and row['strd_tizn_val'] in [7, 8, 18, 19]:
        return 1
    return 0
merged_data['rush_hour'] = merged_data.apply(is_peak, axis=1)

# 설명변수 추가 : X6(계절(범주형))
def get_season(date):
    month = str(date)[4:6]
    if month in ['12', '01', '02']:
        return 1  # 겨울
    elif month in ['03', '04', '05']:
        return 2  # 봄
    elif month in ['06', '07', '08']:
        return 3  # 여름
    elif month in ['09', '10', '11']:
        return 4  # 가을
    else:
        raise ValueError(f"Invalid month value extracted from {date}")
merged_data['season'] = merged_data['strd_date'].apply(get_season)

print(merged_data)
merged_data.to_pickle('merged_data.pkl')
