'''
import pulp
import numpy as np
import requests

#구글맵 API를 이용한 거리 측정
def get_distance(origin, destination, api_key):
    base_url = 'https://maps.googleapis.com/maps/api/distancematrix/json?'
    params = {
        'origins': origin,
        'destinations': destination,
        'key': api_key
    }
    response = requests.get(base_url, params=params)
    data = response.json()
    # 거리 정보는 JSON 응답에서 추출하여 사용합니다.
    distance_text = data['rows'][0]['elements'][0]['distance']['text']
    return distance_text

# origin과 destination은 도시 이름 또는 위도/경도를 포함한 문자열로 표현됩니다.
origin = 'New York, NY'
destination = 'Los Angeles, CA'
api_key = 'YOUR_API_KEY'

distance = get_distance(origin, destination, api_key)
print('Distance between', origin, 'and', destination, 'is', distance)

# 거리 행렬 설정 (예: 미국 30개 도시)
distances = np.array([[0, 1, 2, 3, 4],
                      [1, 0, 5, 6, 7],
                      [2, 5, 0, 8, 9],
                      [3, 6, 8, 0, 10],
                      [4, 7, 9, 10, 0]])

# p-median 문제를 위한 PuLP 문제 설정
prob = pulp.LpProblem("p-median", pulp.LpMinimize)

# 설비(시설) 변수 설정
n = distances.shape[0]  # 도시의 수
p = 2  # 설비(시설)의 수 (원하는 값으로 수정)
x = pulp.LpVariable.dicts("x", [(i, j) for i in range(n) for j in range(n)], cat="Binary")

# 목적 함수 설정 (설비와 각 도시 사이의 거리 합 최소화)
prob += pulp.lpSum(distances[i, j] * x[(i, j)] for i in range(n) for j in range(n))

# 각 도시에서 설비까지의 거리 합이 p 이상이어야 함 (모든 도시가 설비에 할당되어야 함)
for i in range(n):
    prob += pulp.lpSum(x[(i, j)] for j in range(n)) >= p

# 설비(시설)의 수가 p개 이하여야 함
prob += pulp.lpSum(x[(i, j)] for i in range(n) for j in range(n)) == p

# PuLP를 사용하여 최적화 문제 해결
prob.solve()

# 결과 출력
print("설비(시설) 위치:")
for v in prob.variables():
    if v.varValue == 1:
        print(v)
'''

import requests
import numpy as np
import pulp

# Google Maps API 키
API_KEY = 'YOUR_API_KEY'

# 도시 목록
cities = ["New York, NY", "Los Angeles, CA", "Chicago, IL", "Houston, TX", "Phoenix, AZ"]

# 도시 간 거리 행렬 초기화
distances = np.zeros((len(cities), len(cities)))

# Google Maps Distance Matrix API를 사용하여 도시 간 거리 행렬 채우기
for i in range(len(cities)):
    for j in range(len(cities)):
        if i != j:
            origin = cities[i]
            destination = cities[j]
            url = f'https://maps.googleapis.com/maps/api/distancematrix/json?origins={origin}&destinations={destination}&key={API_KEY}'
            response = requests.get(url)
            data = response.json()
            distance = data['rows'][0]['elements'][0]['distance']['value'] / 1000  # 거리를 킬로미터로 변환
            distances[i][j] = distance

# p-median 문제를 위한 PuLP 모델 설정
prob = pulp.LpProblem("p-median", pulp.LpMinimize)

# 설비(시설) 변수 설정
n = len(cities)  # 도시의 수
p = 2  # 설비(시설)의 수 (원하는 값으로 수정)
x = pulp.LpVariable.dicts("x", [(i, j) for i in range(n) for j in range(n)], cat="Binary")

# 목적 함수 설정 (설비와 각 도시 사이의 거리 합 최소화)
prob += pulp.lpSum(distances[i, j] * x[(i, j)] for i in range(n) for j in range(n))

# 각 도시에서 설비까지의 거리 합이 p 이상이어야 함 (모든 도시가 설비에 할당되어야 함)
for i in range(n):
    prob += pulp.lpSum(x[(i, j)] for j in range(n)) >= p

# 설비(시설)의 수가 p개 이하여야 함
prob += pulp.lpSum(x[(i, j)] for i in range(n) for j in range(n)) == p

# PuLP를 사용하여 최적화 문제 해결
prob.solve()

# 결과 출력
print("설비(시설) 위치:")
for v in prob.variables():
    if v.varValue == 1:
        print(cities[v.name[2]], end=", ")