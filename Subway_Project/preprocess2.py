import pandas as pd

# 설명변수 추가 : X7(행사/축제 등 이벤트(범주형))
data = pd.read_pickle('merged_data.pkl')

# 'datetime'을 인덱스로 설정
data['strd_date'] = pd.to_datetime(data['strd_date'], format='%Y%m%d')
data['datetime'] = data['strd_date'] + pd.to_timedelta(data['strd_tizn_val'], unit='h')
data.set_index('datetime', inplace=True)

# 요일 칼럼 추가
data['dayofweek'] = data.index.dayofweek
data.reset_index(inplace=True)
data.drop(['datetime'], axis=1, inplace=True)

# 각 역, 요일, 시간별 평균과 표준편차 계산
def calculate_mean_std(group):
    mean = group['usr_num'].mean()
    std = group['usr_num'].std()
    return pd.Series([mean, std], index=['mean_usr_num', 'std_usr_num'])

mean_std_df = data.groupby(['swst_id', 'dayofweek', 'strd_tizn_val']).apply(calculate_mean_std).reset_index()

# 원본 데이터와 평균, 표준편차 데이터를 병합
data = data.merge(mean_std_df, on=['swst_id', 'dayofweek', 'strd_tizn_val'], how='left').reset_index()

# 평균과 표준편차를 이용한 이상치 탐지 : 평균 + 표준편차의 2배 기준
data['event'] = data['usr_num'] > (data['mean_usr_num'] + 2 * data['std_usr_num'])

# mean_usr_num가 100보다 큰 데이터만 선택
data = data[data['mean_usr_num'] >= 100]

# 이상치 데이터프레임 생성 (strd_date, swst_nm, usr_num, mean_usr_num, std_usr_num, strd_tizn_val 컬럼 포함)
event = data[data['event']][['strd_date','strd_tizn_val', 'swst_nm', 'usr_num', 'mean_usr_num', 'std_usr_num']]

# 이상치와 평균의 차이 계산
event['diff'] = event['usr_num'] - event['mean_usr_num']

# 이상치와 평균의 차이가 큰 행만 선택
event = event[event['diff'] >= 500]
event['strd_date'] = event['strd_date'].dt.strftime('%Y%m%d').astype(int)

merged_data = pd.read_pickle('merged_data.pkl')
merged_data['strd_date'] = pd.to_datetime(merged_data['strd_date'], format='%Y%m%d').dt.strftime('%Y%m%d').astype(int)

# 'event' 변수 추가
event['event'] = 1

# 'strd_date', 'strd_tizn_val', 'swst_nm' 기준으로 병합
merged_data = merged_data.merge(event[['strd_date', 'strd_tizn_val', 'swst_nm', 'event']], on=['strd_date', 'strd_tizn_val', 'swst_nm'], how='left')

# NaN인 경우 0으로 보간
merged_data['event'] = merged_data['event'].fillna(0).astype(int)

merged_data.to_pickle('merged_data2.pkl')