import numpy as np
import pandas as pd
import pulp

#Kakao local API와 Kakao mobility API를 통해 경위도 좌표와 거리를 포함한 데이터 파일 불러오기
pd.set_option('display.width', None)
df = pd.read_csv("distance.csv", encoding='euc-kr')

# 화양동 블록화 목록
cities = ["이룸빌리지", "로얄빌라", "세환원룸빌", "건국대학교병원", "모진당경로당"]

# 설비 후보지 목록
candidates = ["한아름공원", "인애노인복지센터"]

# 도시와 후보지 간 거리 행렬 초기화
distances = np.zeros((len(cities), len(candidates)))

for i in range(len(cities)):
    for j in range(len(candidates)):
        mask = (df['origin'] == cities[i]) & (df['destination'] == candidates[j])
        if not df[mask].empty:
            distances[i][j] = df[mask]['distance'].values[0]

# p-median 문제를 위한 PuLP 모델 설정
prob = pulp.LpProblem("p-median", pulp.LpMinimize)

# 설비(시설) 변수 설정
p = 1  # 설비(시설)의 수 (원하는 값으로 수정)
y = pulp.LpVariable.dicts("y", range(len(candidates)), cat="Binary")

# 목적 함수 설정 (설비 후보지와 각 도시 사이의 거리 합 최소화)
prob += pulp.lpSum(distances[i][j] * y[j] for i in range(len(cities)) for j in range(len(candidates)))

# 각 도시에서 적어도 하나의 설비 후보지에 연결되어야 함
for i in range(len(cities)):
    prob += pulp.lpSum(y[j] for j in range(len(candidates))) >= p

# 설비(시설)의 수가 p개 여야 함
prob += pulp.lpSum(y[j] for j in range(len(candidates))) == p

# PuLP를 사용하여 최적화 문제 해결
prob.solve()

# 결과 출력
print("최적 설비 위치:")
for j in range(len(candidates)):
    if y[j].varValue == 1:
        print(candidates[j])