import pandas as pd

df = pd.read_csv('df_clustered_kmode.csv', encoding='euc-kr')

# 운동 처방 칼럼 텍스트 처리
def process_exercises(exercise_str):
    exercise_str = exercise_str.replace('준비운동:', '준비운동').replace('본운동:', '본운동').replace('마무리운동:', '마무리운동')
    parts = exercise_str.split(' / ')
    exercise_dict = {'준비운동': [], '본운동': [], '마무리운동': []}
    for part in parts:
        if '준비운동' in part:
            exercises = part.replace('준비운동', '').split(',')
            exercise_dict['준비운동'].extend([exercise.strip() for exercise in exercises])
        elif '본운동' in part:
            exercises = part.replace('본운동', '').split(',')
            exercise_dict['본운동'].extend([exercise.strip() for exercise in exercises])
        elif '마무리운동' in part:
            exercises = part.replace('마무리운동', '').split(',')
            exercise_dict['마무리운동'].extend([exercise.strip() for exercise in exercises])
    return exercise_dict
df['Exercise_PRSCRPTN'] = df['Exercise_PRSCRPTN'].apply(process_exercises)

# 운동 처방 칼럼을 준비/본/마무리운동 칼럼으로 분할
df['준비운동'] = df['Exercise_PRSCRPTN'].apply(lambda x: None if not x['준비운동'] else ', '.join(x['준비운동']))
df['본운동'] = df['Exercise_PRSCRPTN'].apply(lambda x: None if not x['본운동'] else ', '.join(x['본운동']))
df['마무리운동'] = df['Exercise_PRSCRPTN'].apply(lambda x: None if not x['마무리운동'] else ', '.join(x['마무리운동']))
df = df.drop(columns=['Exercise_PRSCRPTN'])
df.index.name = "id"

# 준비/본/마무리운동 칼럼 확장
def expand_exercises(df, column_name, prefix):
    split_exercises = df[column_name].str.split(',', expand=True)
    split_exercises.columns = [f'{prefix}_{i+1:02d}' for i in range(split_exercises.shape[1])]
    return split_exercises
prep_exercises = expand_exercises(df, '준비운동', '준비운동')
main_exercises = expand_exercises(df, '본운동', '본운동')
cool_exercises = expand_exercises(df, '마무리운동', '마무리운동')

# 클러스터와 확장된 운동 칼럼 데이터프레임 생성
prep_exercises = pd.concat([df['clusters'], prep_exercises], axis=1)
main_exercises = pd.concat([df['clusters'], main_exercises], axis=1)
cool_exercises = pd.concat([df['clusters'], cool_exercises], axis=1)

# 각 운동의 item-user(cluster-user) matrix 생성
def create_item_user_matrix(exercises_df):
    melted_data = exercises_df.melt(
        id_vars = 'clusters', 
        value_vars = exercises_df.columns[1:],
        var_name = 'Exercise_Type', 
        value_name = 'Exercise')
    melted_data.dropna(subset=['Exercise'], inplace=True)
    melted_data['Exercise'] = melted_data['Exercise'].str.strip()
    
    cleaned_exercise_counts = melted_data.groupby(['clusters', 'Exercise']).size().reset_index(name='Count')
    pivot_table = cleaned_exercise_counts.pivot_table(index='clusters', columns='Exercise', values='Count', fill_value=0)
    return pivot_table

# 준비운동, 본운동, 마무리운동 매트릭스 생성
prep_exercises_matrix = create_item_user_matrix(prep_exercises)
main_exercises_matrix = create_item_user_matrix(main_exercises)
cool_exercises_matrix = create_item_user_matrix(cool_exercises)

df.to_csv('seperate_exercise.csv', index=True, encoding="euc-kr")
prep_exercises_matrix.to_csv('prep_exercises_matrix.csv', index=True, encoding="euc-kr")
main_exercises_matrix.to_csv('main_exercises_matrix.csv', index=True, encoding="euc-kr")
cool_exercises_matrix.to_csv('cool_exercises_matrix.csv', index=True, encoding="euc-kr")


