---
title: "추천 시스템 평가"
excerpt: "F1-Score 평가"

categories:
  - recommend_system

toc: true
toc_sticky: true
date: 2023-02-21
---

# 추천 시스템 검증

- precision@k와 recall@k를 활용 f1_Score 진행
  - 0.8 이상의 성과 도출

```python
def main():
    try :
        result_ID = sort_by_distance()
    except ValueError:
         result_ID = ''
    recommend = re.sub(' ', '', str(list(result_ID))[1:-1])

    return recommend


def predict():
    recommend = main()
    pre = recommend.split(',')
    pre = list(map(int, pre))
    return pre

def f1_score():
    try:
        pre = predict()
        user_bias_points = get_user_bias_points()
        df_knn = get_car_bias_points()

        if 'TON' in df_knn.columns:
            df_knn_a = df_knn[(df_knn['TON'] >= user_bias_points['TON'] - 0.2) & (df_knn['TON']<= user_bias_points['TON'] + 0.2) & (df_knn['BRAND'] >= user_bias_points['BRAND'] - 0.3) & (df_knn['BRAND']<= user_bias_points['BRAND'] + 0.3) & (df_knn['RELIABILITY'] >= user_bias_points['RELIABILITY'] - 0.4) & (df_knn['RELIABILITY']<= user_bias_points['RELIABILITY'] + 0.4) & (df_knn['CONDITION'] >= user_bias_points['CONDITION'] - 0.4) & (df_knn['CONDITION']<= user_bias_points['CONDITION'] + 0.4) & (df_knn['ASSET'] >= user_bias_points['ASSET'] - 0.4) & (df_knn['ASSET']<= user_bias_points['ASSET'] + 0.4)]
        else:
            df_knn_a = df_knn[(df_knn['BRAND'] >= user_bias_points['BRAND'] - 0.5) & (df_knn['BRAND']<= user_bias_points['BRAND'] + 0.5) & (df_knn['RELIABILITY'] >= user_bias_points['RELIABILITY'] - 0.3) & (df_knn['RELIABILITY']<= user_bias_points['RELIABILITY'] + 0.3) & (df_knn['CONDITION'] >= user_bias_points['CONDITION'] - 0.3) & (df_knn['CONDITION']<= user_bias_points['CONDITION'] + 0.3) & (df_knn['ASSET'] >= user_bias_points['ASSET'] - 0.33) & (df_knn['ASSET']<= user_bias_points['ASSET'] + 0.33)]

        if len(df_knn_a) > 6:
            df_knn_a = df_knn[(df_knn['TON'] >= user_bias_points['TON'] - 0.2) & (df_knn['TON']<= user_bias_points['TON'] + 0.2) & (df_knn['BRAND'] >= user_bias_points['BRAND'] - 0.3) & (df_knn['BRAND']<= user_bias_points['BRAND'] + 0.3) & (df_knn['RELIABILITY'] >= user_bias_points['RELIABILITY'] - 0.33) & (df_knn['RELIABILITY']<= user_bias_points['RELIABILITY'] + 0.33) & (df_knn['CONDITION'] >= user_bias_points['CONDITION'] - 0.33) & (df_knn['CONDITION']<= user_bias_points['CONDITION'] + 0.33) & (df_knn['ASSET'] >= user_bias_points['ASSET'] - 0.33) & (df_knn['ASSET']<= user_bias_points['ASSET'] + 0.33)]
        else :
            df_knn_a = df_knn[(df_knn['TON'] >= user_bias_points['TON'] - 0.5) & (df_knn['TON']<= user_bias_points['TON'] + 0.5) & (df_knn['BRAND'] >= user_bias_points['BRAND'] - 0.3) & (df_knn['BRAND']<= user_bias_points['BRAND'] + 0.3) & (df_knn['RELIABILITY'] >= user_bias_points['RELIABILITY'] - 0.5) & (df_knn['RELIABILITY']<= user_bias_points['RELIABILITY'] + 0.5) & (df_knn['CONDITION'] >= user_bias_points['CONDITION'] - 0.5) & (df_knn['CONDITION']<= user_bias_points['CONDITION'] + 0.5) & (df_knn['ASSET'] >= user_bias_points['ASSET'] - 0.5) & (df_knn['ASSET']<= user_bias_points['ASSET'] + 0.5)]

        if len(df_knn_a) < 6:
            df_knn_a = df_knn[(df_knn['TON'] >= user_bias_points['TON'] - 0.2) & (df_knn['TON']<= user_bias_points['TON'] + 0.2) & (df_knn['BRAND'] >= user_bias_points['BRAND'] - 0.3) & (df_knn['BRAND']<= user_bias_points['BRAND'] + 0.3) & (df_knn['RELIABILITY'] >= user_bias_points['RELIABILITY'] - 0.5) & (df_knn['RELIABILITY']<= user_bias_points['RELIABILITY'] + 0.5) & (df_knn['CONDITION'] >= user_bias_points['CONDITION'] - 0.5) & (df_knn['CONDITION']<= user_bias_points['CONDITION'] + 0.5) & (df_knn['ASSET'] >= user_bias_points['ASSET'] - 0.5) & (df_knn['ASSET']<= user_bias_points['ASSET'] + 0.5)]

        if df_knn_a.empty:
            df_knn_a = df_knn[(df_knn['TON'] >= user_bias_points['TON'] - 0.5) & (df_knn['TON']<= user_bias_points['TON'] + 0.5) & (df_knn['BRAND'] >= user_bias_points['BRAND'] - 0.5) & (df_knn['BRAND']<= user_bias_points['BRAND'] + 0.5) & (df_knn['RELIABILITY'] >= user_bias_points['RELIABILITY'] - 0.55) & (df_knn['RELIABILITY']<= user_bias_points['RELIABILITY'] + 0.55) & (df_knn['CONDITION'] >= user_bias_points['CONDITION'] - 0.55) & (df_knn['CONDITION']<= user_bias_points['CONDITION'] + 0.55) & (df_knn['ASSET'] >= user_bias_points['ASSET'] - 0.57) & (df_knn['ASSET']<= user_bias_points['ASSET'] + 0.584)]
        else :
            df_knn_a = df_knn_a

        if len(df_knn) < 6:
            df_knn_a = df_knn

        X, Y, Z, T, B = np.array(df_knn_a['RELIABILITY']), np.array(df_knn_a['CONDITION']), np.array(df_knn_a['ASSET']), np.array(df_knn_a['TON']), np.array(df_knn_a['BRAND'])
        distance_a = np.sqrt(
                    ((X - user_bias_points['RELIABILITY']) ** 2) +
                    ((Y - user_bias_points['CONDITION']) ** 2) +
                    ((Z - user_bias_points['ASSET']) ** 2) +
                    ((T - user_bias_points['TON']) ** 2) +
                    ((B - user_bias_points['BRAND']) ** 2)
                )
        df_knn_a['거리'] = distance_a
        result = df_knn_a.sort_values(by='거리').index[:6]

        anw = re.sub(' ', '', str(list(result))[1:-1])
        anw = anw.split(',')
        anw = list(map(int, anw))

    except:
        anw = []
        pre = []

    if len(anw) != 0:
        usr_anw = 0
        for idx, val in enumerate(pre):
            if val in anw:
                usr_anw += 1
            else :
                continue
        precision = usr_anw / len(pre)

        re_anw = 0
        for r_idx, r_val in enumerate(anw):
            if r_val in pre:
                re_anw += 1
            else :
                continue
        recall = re_anw / len(anw)

        f1_score = 2.0 / (1/precision + 1/recall)

    else :
        f1_score = 0

    return f1_score
```
