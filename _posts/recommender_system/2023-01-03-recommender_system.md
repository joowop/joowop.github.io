---
title: "추천 시스템 모델링 test"
excerpt: "사용자 설문 기반 추천 시스템 모델링 프로젝트 시작 "

categories:
  - recommend_system

toc: true
toc_sticky: true
# date: 2020-05-25
# last_modified_at: 2020-05-25
---

# 사용자 설문 기반 중고트럭 추천 서비스 모델링

### 기술 스택

- python, sklearn, tensor

---

- 유저들이 구입하려는 중고트럭을 쉽게 탐색할 수 있도록 차량 구매자 요구에 따른 근접매물 추천 서비스를 제공하고자 개발 목표 수립

- 서로 다른 단위의 차량 데이터 정보를 Min-Max Scaling하여 정규화

- 기준을 잡아 차량의 톤수, 차량의 브랜드, 딜러의 신뢰도, 차량상태, 유저의 보유자산 상태를 측정

- 5개의 축을 통해 0~1사이의 점수를 도출

- 최근접 이웃(KNN)을 통해 유저가 선택한 차량의 총 점수와 DB에 저장되어 있는 차량의 점수와 비교
  - 가장 가까운 차량 6대를 추천

```python
from sklearn.preprocessing import MinMaxScaler
import sys
import numpy as np
import pandas as pd
sys.path.insert(0, '../../assets/modules')
import re
from truckEDA import EDAHelper
from datetime import datetime as dt
from dotenv import dotenv_v
alues
import random
from IPython.core.interactiveshell import InteractiveShell
eda = EDAHelper()
now = pd.to_datetime(dt.today())
```

### 데이터 로드

DB 차량 data를 import하기 위해 sys를 통해 moudules 접근.
EDAHelper라는 모듈을 통해 DB의 data를 가독성 있는 data로 변환.
data 정규화를 위한 라이브러리인 MinMaxScaler 채택.

```python
def call_mycar(df):
    df = pd.merge(left=dfs['GROUP01']['trucks'], right=dfs['GROUP01']['dealer_cnt'], on='DAL_ID', how='left')
    group2 = pd.merge(left=dfs['GROUP02']['trucks'], right=dfs['GROUP02']['dealer_cnt'], on='DAL_ID', how='left')
    group3 = pd.merge(left=dfs['GROUP03']['trucks'], right=dfs['GROUP03']['dealer_cnt'], on='DAL_ID', how='left')
    return df
```

### 데이터 전처리

추천 시스템에서 사용할 data를 점수로 도출(정규화)하기 위해 바이너리한 data로 전처리.
ex\_ dataframe의 '실매물 확인여부'라는 컬럼의 값이 'Y', 'N'의 str 형태로 저장 -> 각각 1, 0의 값으로 replace

```python
def preprocess_mycar(df):
    df = eda.rename_column(df, ["TB_MYCAR", "TB_DEALER", "TB_CLICK"]).rename(columns={
        'DAL_TOTAL_UPT_CNT': '딜러 등록대수',
        '종사원증 번호': '종사원증 번호여부',
        '상세설명': '상세설명 길이'
    })

    df['가입일자'] = df['가입일자'].str.replace(pat=r'[a-zA-Z]+', repl= r' ', regex=True)
    df['마지막 로그인 시간'] = df['마지막 로그인 시간'].str.replace(pat=r'[a-zA-Z]+', repl= r' ', regex=True)
    df['등록일자'] = df['등록일자'].str.replace(pat=r'[a-zA-Z]+', repl= r' ', regex=True)

    df["상세설명 길이"] = df["상세설명 길이"].str.replace(pat=r'[^\w]', repl=r'', regex=True)
    df["상세설명 길이"] = df["상세설명 길이"].str.len().fillna(0)

    df['마지막 로그인 시간'] = pd.to_datetime(df['마지막 로그인 시간'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
    df['가입일자'] = pd.to_datetime(df['가입일자'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
    df['등록일자'] = pd.to_datetime(df['등록일자'], format='%Y-%m-%d %H:%M:%S', errors='coerce')

    df["종사원증 번호여부"] = df["종사원증 번호여부"].str.len()

    df.loc[df["종사원증 번호여부"] <= 5, '종사원증 번호여부'] = 0
    df.loc[df["종사원증 번호여부"] > 5, '종사원증 번호여부'] = 1

    # MCR_IMG1 ~ MCR_IMG20 이미지 파일은 존재할경우 1, Null일경우 0으로 변환
    for i in range(1, 21):
        df[['이미지{}'.format(i)]] = df[['이미지{}'.format(i)]].where(df[['이미지{}'.format(i)]].isnull(), 1).fillna(0)

    # 변환한 값에 대해 모두 더한값을 저장하는 MCR_IMG 컬럼 생성
    df['이미지 수'] = 0

    # 연산(Count MCR_IMG1 ~ MCR_IMG20) & drop 이미지 columns
    for i in range(1, 21):
        df['이미지 수'] += df[f'이미지{i}']
        df.drop(columns=[f"이미지{i}"], inplace=True)

    df['연식'] = pd.to_datetime(df['연식'], format='%Y-%m-%d', errors='coerce')
    df.drop(df[df['연식'] > now].index, inplace=True)

    df['번호판 종류'] = df['번호판 종류'].replace('0402', 1).replace('0401', 0)

    df['실매물 확인여부'] = df['실매물 확인여부'].replace('N', 0).replace('Y', 1)
    df['직거래 매물여부'] = df['직거래 매물여부'].replace('N', 0).replace('Y', 1)

    df['헛걸음보상제공여부'] = df['헛걸음보상제공여부'].replace('N', 0).replace('Y', 1)

    df['마지막 로그인 시간'].fillna(df['가입일자'], inplace=True)

    df['톤수'] = df['톤수'].replace('기타', '0').astype(float)
    df['톤수'] = eda.ton_masking(df, '톤수')
    df['주행거리'] = df['주행거리'].astype("int64")
    df['가격'] = df['가격'].astype("int64")

    df['일평균등록매물'] = (df['마지막 로그인 시간'] - df['가입일자']).dt.days / df['딜러 등록대수']
    df['일평균조회수'] = (df['클릭횟수'] / ((now - df['등록일자']).dt.days + 1)).fillna(0)
    df['일평균조회수'] = -df['일평균조회수']

    df = df.drop(columns=[ '등록일자', '가입일자', '마지막 로그인 시간', '클릭횟수', '딜러 아이디', '딜러 등록대수'])
    # 일평균 주행거리가 600km 초과인 매물 drop
    df = df.drop(df[df['주행거리'] / ((now - df['연식']).dt.days) > 600].index)
    # 차량상태 점수를 위한 주행거리 뒤집기
    df['주행거리'] = -df['주행거리']

    # 100만원 이하, 3억 이상인 차량 drop
    min_drop = 100
    df = df.drop(df[df['가격'] <= min_drop].index)
    max_drop = 30000
    df = df.drop(df[df['가격'] >= max_drop].index)

    df.set_index('내차사기 아이디', inplace=True)

    return df
```

### 각 축(라벨)에 대한 점수 도출

설정한 축(라벨) 톤, 브랜드, 신뢰도 높음, 차량 상태, 보유자산 총 5개의 축에 대한 점수 도출.
점수는 MinMaxScale을 통해 각각 0~1값으로 도출
도출한 점수는 각 차량마다의 고유 점수 부여
ex -> ID가 123인 차량의 총합 점수(톤, 브랜드, 신뢰도, 차량상태, 보유자산을 MinMax하고 각 점수를 계산) : 0.8123...

```python
def get_car_bias_points(df, user_points):
    scaler = MinMaxScaler()

    car_ton = [
            (1, 1.2),
            (1.3, 4.5),
            (4.5, 5),
            (5, 11.5),
            (11.5, 27),
        ]

    car_brand = [
        ('C'),
        ('D')
    ]

    # 신뢰도 높음 점수
    df_reliability = df[[
        "실매물 확인여부",
        "종사원증 번호여부",
        "일평균등록매물",
        "헛걸음보상제공여부",
       ]]
    reliability = scaler.fit_transform(df_reliability)

    # 차량상태 좋음 점수
    df_condition = df[[
        "주행거리",
        "가격",
        "상세설명 길이",
        "이미지 수",
       ]]

    condition = scaler.fit_transform(df_condition)
    condition = np.concatenate((condition, scaler.fit_transform(df[["연식"]])), axis=1)
    # 보유자산 낮음 점수
    df_asset = df[[
        "직거래 매물여부",
        "번호판 종류",
        "일평균조회수",
    ]]

    asset = scaler.fit_transform(df_asset)

     # 톤 점수
    dfa = df.copy()
    dfa['톤수_점수'] = None
    for i, t in enumerate(car_ton):
        dfa.loc[
            (dfa['톤수'] >= t[0]) & (dfa['톤수'] <= t[1]), '톤수_점수'
        ] = (len(car_ton) / (len(car_ton) * (len(car_ton) - 1))) * ((i + 1) - 1)
    ton = scaler.fit_transform(pd.DataFrame(-abs(dfa['톤수_점수'] - user_points['TON']['Q1'])))

    # 브랜드 점수
    dfa_2 = df.copy()
    dfa_2['브랜드_점수'] = None
    for i, b in enumerate(car_brand):
        globals()["i_temp"] = i
        k = dfa_2.loc[
            (dfa_2['브랜드'].str.contains(b)), '브랜드_점수'
        ] = (len(car_brand) / (len(car_brand) * (len(car_brand) - 1))) * ((i + 1) - 1)
        l = dfa_2['브랜드_점수']
    brand = scaler.fit_transform(pd.DataFrame(-abs(dfa_2['브랜드_점수'] - user_points['BRAND']['Q1'])))

    # 각 차량의 점수 도출 후 dataframe으로 저장
    df_knn = pd.DataFrame({
        'RELIABILITY': (reliability.sum(axis=1) / reliability.shape[1]),
        'CONDITION': (condition.sum(axis=1) / condition.shape[1]),
        'ASSET': (asset.sum(axis=1) / asset.shape[1]),
        'TON': (ton.sum(axis=1) / ton.shape[1]),
        'BRAND' : (brand.sum(axis=1) / brand.shape[1]),
    }, index=df.index,)
    globals()["df_knn_temp_2"] = df_knn

    return df_knn
```

### User 차량 설문 선택 점수

User가 총 5개의 축에 대한 설문을 진행
ex -> 톤: 1톤, 브랜드: 국산, 신뢰도: 높음, 차량상태 : 좋음, 보유자산 : 낮음 등...
설문을 진행하여 얻은 값은 1~5까지의 int형 정수를 확보
도출된 정수형 data를 바탕으로 차량의 점수와 마찬가지로 0~1의 값의 점수로 도출할 수 있도록 계산 및 설계

```python
def get_user_bias_points(questions, user_points):
    user_bias_points = dict()
    for (axis_question, quests_question), (axis_answer, quests_answer) in zip(questions.items(), user_points.items()):
        user_bias_points[axis_answer] = list()
        for (q_num, quest_question), (a_num, quest_answer) in zip(quests_question.items(), quests_answer.items()):
            user_bias_points[axis_answer].append((quest_question["QUESTION_NUMBER"] / (quest_question["QUESTION_NUMBER"] * (quest_question["QUESTION_NUMBER"] - 1))) * (quest_answer - 1))
        user_bias_points[axis_answer] = sum(user_bias_points[axis_answer]) / len(user_bias_points[axis_answer])

    return user_bias_points
```

### Knn 공식을 이용한 거리 계산 및 추천 차량 선택

Knn 공식을 이용하여 도출된 해당 차량의 점수, User가 선택한 차량의 점수를 계산하여
가장 가까운 거리의 차량 6대를 slicing 해준 뒤 추천

```python
def sort_by_distance(user_bias_points, df_knn):
    X, Y, Z, T, B = np.array(df_knn['RELIABILITY']), np.array(df_knn['CONDITION']), np.array(df_knn['ASSET']), np.array(df_knn['TON']), np.array(df_knn['BRAND'])

    distance = np.sqrt(
                ((X - user_bias_points['RELIABILITY']) ** 2) +
                ((Y - user_bias_points['CONDITION']) ** 2) +
                ((Z - user_bias_points['ASSET']) ** 2) +
                ((T - user_bias_points['TON']) ** 2) +
                ((B - user_bias_points['BRAND']) ** 2)
            )
    globals()["distance"] = distance
    df_knn['거리'] = distance

    result_ID = df_knn.sort_values(by='거리').index[:6]
    globals()["result_ID"] = result_ID
    return result_ID
```

### main 동작 함수

위의 함수가 동작할 수 있도록 함수 설계.
User가 선택할 설문을 test할 수 있도록 random으로 설계 (현재 User가 선택할 설문을 정리해 놓은 Nosql형식의 Json파일이 있음)
KNN(df, questions, user_points) 실행 시 추천 시스템 모델 동작
-> 6개의 차량 ID 도출

```python
def KNN(df, questions, user_points):
    # get raw data
#     df_mcr = call_mycar(df)

    # 전처리
    df_mcr_cleansed = df.copy()
    df_mcr_cleansed = preprocess_mycar(df_mcr_cleansed)
    globals()["df_mcr_cleansed"] = df_mcr_cleansed

    # 매물 feature points
    try:
        user_bias_points = get_user_bias_points(questions, user_points)
        globals()["user_bias_points_temp"] = user_bias_points
        df_knn = get_car_bias_points(df_mcr_cleansed, user_points)
        globals()["df_knn_temp"] = pd.merge(df_mcr_cleansed.copy().reset_index(), df_knn.copy().reset_index(), how='left', on='내차사기 아이디')
        recommend_truck = sort_by_distance(user_bias_points, df_knn)
        globals()["recommend_truck_temp"] = recommend_truck
    except ValueError:
        recommend_truck = ''
# get_car_bias_points(get_survey_brand(preprocess_mycar(call_mycar(df).copy()), BRAND_NUM)))
    print_recommendation(recommend_truck)

if __name__ == '__main__':
    # 메타 질문 데이터는 꼭 json화 돼 있어야 됨
    questions = {
        "RELIABILITY": {
            f"Q{i+1}": {"QUESTION": "질문 예시", "QUESTION_NUMBER": random.randint(2, 5)} for i in range(random.randint(3, 7))
        },
        "CONDITION": {
            f"Q{i+1}": {"QUESTION": "질문 예시", "QUESTION_NUMBER": random.randint(2, 5)} for i in range(random.randint(3, 7))
        },
        "ASSET": {
            f"Q{i+1}": {"QUESTION": "질문 예시", "QUESTION_NUMBER": random.randint(2, 5)} for i in range(random.randint(3, 7))
        },
        "TON": {
            "Q1":{
                "QUESTION": "질문 예시",
                "QUESTION_NUMBER": 5,
            }
        },
        "BRAND": {
            "Q1":{
                "QUESTION": "질문 예시",
                "QUESTION_NUMBER": 2,
            }
        }
    }
    user_points = dict() # 이거는 파라미터로 받아야 됨
    for axis, quests in questions.items():
        user_points[axis] = dict()
        for q_num, quest in quests.items():
            user_points[axis][q_num] = random.randint(1, quest["QUESTION_NUMBER"])
    df = pd.merge(left=df_truck, right=df_dealer_cnt, on='DAL_ID', how='left')
    KNN(df, questions, user_points)
```

### 추천 시스템 모델링 끝!
