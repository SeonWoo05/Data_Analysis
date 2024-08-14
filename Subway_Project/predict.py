import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from pyproj import Transformer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from category_encoders import TargetEncoder
import warnings
warnings.filterwarnings('ignore')

data = pd.read_pickle('merged_data2.pkl')

# 위도/경도를 중부원점 좌표계로 변환
# EPSG:5174는 한국 중부원점 좌표계
transformer = Transformer.from_crs("epsg:4326", "epsg:5174")

# 변환 적용
data['swst_ltd_cdn_val'], data['swst_lgd_cdn_val'] = transformer.transform(data['swst_ltd_cdn_val'].values, data['swst_lgd_cdn_val'].values)

# 1/1000 스케일링
data['swst_ltd_cdn_val'] = data['swst_ltd_cdn_val'] / 1000
data['swst_lgd_cdn_val'] = data['swst_lgd_cdn_val'] / 1000

# 수영역(3호선)/지하철역 이름 drop, index를 시일로
data = data[data['swst_nm'] != '수영역(3호선)']
data['strd_date'] = pd.to_datetime(data['strd_date'], format='%Y%m%d')


# 시계열 처리 -> cos/sin 주기성 활용
data['year'] = data['strd_date'].dt.year
data['month'] = data['strd_date'].dt.month
data['day'] = data['strd_date'].dt.day

# 요일과 월을 주기적 특성으로 변환
data['day_of_week_sin'] = np.sin(2 * np.pi * data['strd_date'].dt.dayofweek / 7)
data['day_of_week_cos'] = np.cos(2 * np.pi * data['strd_date'].dt.dayofweek / 7)
data['month_sin'] = np.sin(2 * np.pi * data['strd_date'].dt.month / 12)
data['month_cos'] = np.cos(2 * np.pi * data['strd_date'].dt.month / 12)

# 시간의 주기성 반영
data['time_sin'] = np.sin(2 * np.pi * data['strd_tizn_val'] / 24)
data['time_cos'] = np.cos(2 * np.pi * data['strd_tizn_val'] / 24)

data['datetime'] = data['strd_date'] + pd.to_timedelta(data['strd_tizn_val'], unit='h')
data.drop(['strd_date', 'swst_nm', 'season'], axis=1, inplace=True)
data.set_index('datetime', inplace=True)

# # 탑승객 수의 정규성 확인
# data['usr_num'].hist(bins=100)
# plt.title('Distribution of usr_num')
# plt.xlabel('usr_num')
# plt.ylabel('Frequency')
# plt.show()

# 왜곡이 심하므로 로그 변환 -> 추후 예측값을 expm1()으로 복구 예정
y_log = np.log1p(data['usr_num'])
# y_log.hist(bins=100)
# plt.title('Distribution of usr_num')
# plt.xlabel('usr_num')
# plt.ylabel('Frequency')
# plt.show()

# 범주형 데이터 변환
categorical_features = ['year', 'month', 'day', 'strd_tizn_val', 'swst_id', 'is_transfer', 'is_holiday', 'rush_hour', 'event']
data[categorical_features] = data[categorical_features].astype('category')

# 독립변수/종속변수 설정
X = data.drop(['usr_num'], axis=1)
# y = data['usr_num']

# # 1) OnehotEncoding을 통해 범주형 변수 처리
# X = pd.get_dummies(X, columns=categorical_features)

# 2) Target Encoding을 통해 범주형 변수 처리
# X_train, X_test, y_train, y_test = train_test_split(X, y_log, test_size=0.3, random_state=94)
encoder = TargetEncoder(cols=categorical_features)
train_encoded = encoder.fit_transform(X, y_log)

# 예측
# RMSE 계산
def evaluate_regr(y,pred):
    rmse_val = np.sqrt(mean_squared_error(y,pred))
    print(f'RMSE: {rmse_val:.3F}')

def get_model_predict(model, X_train, X_test, y_train, y_test, is_expm1=False):
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    if is_expm1 :
        y_test = np.expm1(y_test)
        pred = np.expm1(pred)
    print('###',model.__class__.__name__,'###')
    evaluate_regr(y_test, pred)

# XGBoost 적용 -> 미완성 (시계열 처리 미적용, y 로그 변환 미적용, 파라미터 값 수정 필요)
#X_train, X_test, y_train, y_test = train_test_split(X, y_log, test_size=0.2, random_state=94)

# GridSearchCV를 통해 최적의 하이퍼 파라미터값 찾기 (굉장히 오래 걸려요)
# param_grid = {
#     'n_estimators': [500, 700, 1000, 1500, 3000],
#     'learning_rate': [0.01, 0.03, 0.05, 0.08,  0.1],
#     'subsample' : [0.5, 0.8, 1.0]
# }
# xgb_reg = XGBRegressor()
# grid_search = GridSearchCV(xgb_reg, param_grid, scoring='neg_mean_squared_error', cv=3, verbose=1) #cv = 3, 5, 10
# grid_search.fit(X_train, y_train)
# print(grid_search.best_params_)

xgb_reg = XGBRegressor(n_estimators=1000, learning_rate=0.05, colsample_bytree=0.5, subsample=0.8, max_depth=4) # learning_rate=0.05, colsample_bytree=0.5, subsample=0.8, max_depth=4
#get_model_predict(xgb_reg, X_train.values, X_test.values, y_train.values, y_test.values, is_expm1=True)


## 2024년 7월 예측
future_data = pd.read_csv('result_raw.csv', encoding='euc-kr')
result = future_data.copy()

# 'strd_yymm', 'usr_num'열 제거 -> 이후 추가
future_data = future_data.drop(columns=['V1', 'V8'])

# 열 이름 변경
future_data = future_data.rename(columns={
    'V2': 'strd_date',
    'V3': 'strd_tizn_val',
    'V4': 'swst_id',
    'V5': 'swst_nm',
    'V6': 'swst_lgd_cdn_val',
    'V7': 'swst_ltd_cdn_val'
})

# 경위도 중부원점 좌표계 변환 적용
future_data['swst_ltd_cdn_val'], future_data['swst_lgd_cdn_val'] = transformer.transform(future_data['swst_ltd_cdn_val'].values, future_data['swst_lgd_cdn_val'].values)

# 1/1000 스케일링
future_data['swst_ltd_cdn_val'] = future_data['swst_ltd_cdn_val'] / 1000
future_data['swst_lgd_cdn_val'] = future_data['swst_lgd_cdn_val'] / 1000

# 설명변수 추가 : X1(기온), X2(강수량)
temp_2024_07 = pd.read_csv('data/temperature_2024_07.csv', encoding='euc-kr')
def preprocess_temperature(df):
    df_filtered = df.loc[df['지점'] == 159].copy()
    df_filtered.loc[:, 'date'] = pd.to_datetime(df_filtered['일시'], errors='coerce')
    df_filtered = df_filtered.dropna(subset=['date'])
    df_filtered.loc[:, 'strd_date'] = df_filtered['date'].dt.strftime('%Y%m%d').astype(int)
    df_filtered.loc[:, 'strd_tizn_val'] = df_filtered['date'].dt.hour
    df_filtered.loc[:, 'rainfall'] = df_filtered['강수량(mm)'].fillna(0) # Nan값은 비 안 온 거니까 0으로 보간
    return df_filtered[['strd_date', 'strd_tizn_val', '기온(°C)', 'rainfall']].rename(columns={'기온(°C)': 'temperature'})
temp_2024_07 = preprocess_temperature(temp_2024_07)
future_data = pd.merge(future_data, temp_2024_07, on=['strd_date', 'strd_tizn_val'], how='left')

# 설명변수 추가 : X3(환승역(범주형))
transfer_stations_1 = ['교대', '벡스코', '사상', '거제', '대저']
transfer_stations_2 = ['동래',  '연산', '서면', '수영', '덕천', '미남']
future_data['is_transfer'] = future_data['swst_nm'].apply(lambda x: 1 if any(station in x for station in transfer_stations_1) else (2 if any(station in x for station in transfer_stations_2) else 0))

# 설명변수 추가 : X4(공휴일 및 주말(범주형))
future_data['strd_date'] = pd.to_datetime(future_data['strd_date'], format='%Y%m%d')
future_data['is_holiday'] = future_data['strd_date'].apply(lambda x: 1 if x.weekday() >= 5 else 0)

# 설명변수 추가 : X5(출퇴근 시간대(범주형))
def is_peak(row):
    if row['is_holiday'] == 0 and row['strd_tizn_val'] in [7, 8, 18, 19]:
        return 1
    return 0
future_data['rush_hour'] = future_data.apply(is_peak, axis=1)

# 설명변수 추가 : X6(계절(범주형))
future_data['season'] = 3  # 여름

# 설명변수 추가 : X7(행사/축제 등 이벤트(범주형))
future_data['event'] = 0
future_data['strd_date'] = future_data['strd_date'].dt.strftime('%Y%m%d').astype(int)
event = pd.read_csv('data/event.csv', encoding='euc-kr')
merged = future_data.merge(event, on=['strd_date', 'strd_tizn_val', 'swst_nm'], how='left', indicator=True)
future_data.loc[merged['_merge'] == 'both', 'event'] = 1
future_data['strd_date'] = pd.to_datetime(future_data['strd_date'], format='%Y%m%d')

# 시계열 처리 -> cos/sin 주기성 활용
# 날짜를 년, 월, 일로 분리
future_data['year'] = future_data['strd_date'].dt.year
future_data['month'] = future_data['strd_date'].dt.month
future_data['day'] = future_data['strd_date'].dt.day

# 요일과 월을 주기적 특성으로 변환
future_data['day_of_week_sin'] = np.sin(2 * np.pi * future_data['strd_date'].dt.dayofweek / 7)
future_data['day_of_week_cos'] = np.cos(2 * np.pi * future_data['strd_date'].dt.dayofweek / 7)
future_data['month_sin'] = np.sin(2 * np.pi * future_data['strd_date'].dt.month / 12)
future_data['month_cos'] = np.cos(2 * np.pi * future_data['strd_date'].dt.month / 12)

# 시간의 주기성 반영
future_data['time_sin'] = np.sin(2 * np.pi * future_data['strd_tizn_val'] / 24)
future_data['time_cos'] = np.cos(2 * np.pi * future_data['strd_tizn_val'] / 24)

future_data['datetime'] = future_data['strd_date'] + pd.to_timedelta(future_data['strd_tizn_val'], unit='h')
future_data.drop(['strd_date', 'swst_nm', 'season'], axis=1, inplace=True)
future_data.set_index('datetime', inplace=True)

# 범주형 데이터 변환
categorical_features = ['year', 'month', 'day', 'strd_tizn_val', 'swst_id', 'is_transfer', 'is_holiday', 'rush_hour', 'event']
future_data[categorical_features] = future_data[categorical_features].astype('category')

# 독립변수 설정 -> 버스 제외
X_future = future_data

# # 1) OnehotEncoding을 통해 범주형 변수 처리
# X_future = pd.get_dummies(X_future, columns=categorical_features)
# X_future = X_future.reindex(columns=X.columns, fill_value=0)

# 2) Target Encoding을 통해 범주형 변수 처리 -> 얘는 train/target 분할 먼저 하고 인코딩 적용
test_encoded = encoder.transform(X_future)

# 예측
xgb_reg.fit(X.values, y_log.values)

# # 피처 중요도 시각화
# feature_importances = xgb_reg.feature_importances_
# features = X.columns
# importances = pd.Series(feature_importances, index=features)
# importances = importances.sort_values(ascending=False)

# plt.figure(figsize=(10, 6))
# importances.plot(kind='barh', fontsize=10)
# plt.title('Feature Importances')
# plt.xlabel('Importance')
# plt.ylabel('Features')
# plt.show()

y_future = xgb_reg.predict(X_future.values)
y_future = np.expm1(y_future)

# 예측 결과 후처리: 음수를 0으로 (반올림하여 정수로 변환)
result['V8'] = np.round(np.maximum(0, y_future)).astype(int)

result.to_csv('result.csv', index=False, encoding='euc-kr')