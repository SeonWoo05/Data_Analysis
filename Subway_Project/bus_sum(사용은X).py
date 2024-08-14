import pandas as pd
from geopy.distance import geodesic

# 설명변수 추가 : X7(근처 3개 버스정류장 탑승객수)
merged_data = pd.read_pickle('merged_data.pkl')
bus_usage_df = pd.read_csv('2022.01_버스 정류소별 이용객 통계 데이터.csv', encoding='euc-kr')
subway_usage_df = pd.read_csv('combined_data_22.csv', encoding='euc-kr')

# 지하철, 버스의 이름과 경위도 추출
unique_bus_stops = bus_usage_df[['stps_nm', 'stps_lgd_cdn_val', 'stps_ltd_cdn_val']].drop_duplicates()
unique_subway_stops = subway_usage_df[['swst_nm', 'swst_lgd_cdn_val', 'swst_ltd_cdn_val']].drop_duplicates()

# 각 지하철역과 가장 가까운 3개의 버스 정류장을 찾기
# 보고서 : geodesic 거리에 대한 설명 간략하게 필요
def find_nearest_bus_stops(subway_row, unique_bus_stops, n=3):
    subway_location = (subway_row['swst_ltd_cdn_val'], subway_row['swst_lgd_cdn_val'])
    unique_bus_stops['dist'] = unique_bus_stops.apply(lambda row: geodesic(subway_location, (row['stps_ltd_cdn_val'], row['stps_lgd_cdn_val'])).meters, axis=1)
    nearest_buses = unique_bus_stops.nsmallest(n, 'dist')
    return nearest_buses

nearest_buses_list = []
for _, subway_row in unique_subway_stops.iterrows():
    nearest_buses = find_nearest_bus_stops(subway_row, unique_bus_stops)
    for i in range(3):
        subway_row[f'stps_nm{i+1}'] = nearest_buses.iloc[i]['stps_nm']
        subway_row[f'stps_lgd_cdn_val{i+1}'] = nearest_buses.iloc[i]['stps_lgd_cdn_val']
        subway_row[f'stps_ltd_cdn_val{i+1}'] = nearest_buses.iloc[i]['stps_ltd_cdn_val']
    nearest_buses_list.append(subway_row)
nearest_buses_df = pd.DataFrame(nearest_buses_list)

# 지하철 역별 인접 버스 정류장의 시간별 탑승객 수 합산
def get_nearby_buses_usage(subway_row, bus_df):
    nearby_stops = [
        subway_row['stps_nm1'],
        subway_row['stps_nm2'],
        subway_row['stps_nm3']
    ]
    filtered_bus_df = bus_df[bus_df['stps_nm'].isin(nearby_stops)]
    filtered_bus_df = filtered_bus_df.groupby(['strd_date', 'strd_tizn_val'])['usr_num'].sum().reset_index()
    filtered_bus_df['swst_nm'] = subway_row['swst_nm']
    return filtered_bus_df

# 2022년 1월부터 2024년 6월까지 적용
bus_sum = pd.DataFrame()
for year in range(22, 25):
    end_month = 6 if year == 24 else 12
    for month in range(1, end_month + 1):
        bus_file_path = '20{:02d}.{:02d}_버스 정류소별 이용객 통계 데이터.csv'.format(year, month)
        bus_df = pd.read_csv(bus_file_path, encoding='euc-kr')
        for _, subway_row in nearest_buses_df.iterrows():
            nearby_buses_usage = get_nearby_buses_usage(subway_row, bus_df)
            bus_sum = pd.concat([bus_sum, nearby_buses_usage], ignore_index=True)
bus_sum.rename(columns={'usr_num': 'bus_usr_sum'}, inplace=True)

# 병합하여 최종 데이터 생성
merged_data = pd.merge(merged_data, bus_sum[['strd_date', 'strd_tizn_val', 'swst_nm', 'bus_usr_sum']],
                     on=['strd_date', 'strd_tizn_val', 'swst_nm'], how='left')

# bus_usr_num이 없는 행을 0으로 보간
merged_data['bus_usr_sum'] = merged_data['bus_usr_sum'].fillna(0).astype(int)
merged_data['usr_num'] = merged_data.pop('usr_num')

print(merged_data.head().to_string())
merged_data.to_pickle('merged_data1.pkl')
