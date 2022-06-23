import pandas as pd
import itertools
import requests
import random
import re
import matplotlib.pyplot as plt
import numpy as np
import xgboost
from bs4 import BeautifulSoup

imported_df = pd.read_excel("./lotto_number.xlsx")
df = imported_df.iloc[:, [1, 2, 3, 4, 5, 6, 7]]
df = df.values.tolist()  # 모든 7개의 로또 번호 리스트

'''
    무작위 숫자 6개를 출력합니다.
'''
def random_pick():
    result = list()
    while len(result) != 6:
        result.append(random.randint(1, 45))
        result = list(set(result))
    return result

'''
    역대 가장 많이/적게 나온 번호 6개를 출력합니다.
'''
def most_picked_num(reverse=False, pick=6):  # reverse가 True인 경우, 적게 나온 순으로 출력.
    count_lotto = dict()

    for i in range(1, 46):
        count_lotto[i] = 0

    for numbers in df:
        for number in numbers:
            count_lotto[number] += 1

    if not reverse:
        count_lotto = sorted(count_lotto.items(), key=lambda x: x[1], reverse=True)
    else:
        count_lotto = sorted(count_lotto.items(), key=lambda x: x[1])

    result = list()
    for i in range(pick):
        result.append(count_lotto[i][0])

    return result

'''
    역대 로또 번호 통계를 출력합니다.
'''
def print_stastics():
    count_lotto = dict()

    for i in range(1, 46):
        count_lotto[i] = 0

    for numbers in df:
        for number in numbers:
            count_lotto[number] += 1

    keys = np.array(list(count_lotto.keys())); values = np.array(list(count_lotto.values()))
    is_max = values == max(values)
    is_min = values == min(values)
    plt.figure(figsize=(14, 10))
    plt.xlabel('numbers')
    plt.ylabel('picked_count')
    plt.bar(keys, values)
    plt.bar(keys[is_max], values[is_max], color='red')
    plt.bar(keys[is_min], values[is_min], color='yellow')
    plt.xticks(keys)
    plt.yticks(values)
    plt.xlim(min(keys) - 1, max(keys) + 1)
    plt.ylim(min(values) - 1, max(values) + 1)

    for i, v in enumerate(keys):
        plt.text(v, values[i], values[i],
                 fontsize=12, color='black',
                 horizontalalignment='center',
                 verticalalignment='top')
        plt.text(v, values[i], keys[i],
                 fontsize=12, color='red',
                 horizontalalignment='center',
                 verticalalignment='bottom')
    plt.show()

'''
    xgboost 머신러닝 알고리즘을 통해 로또 번호를 예측합니다.
    계산 방식: 각 회차의 합을 구하여, 이를 기반으로 다음 회차의 번호 합을 예측.
    이 합을 기반으로 가장 많이/적게 나온 숫자들을 바탕으로 combination을 생성.
'''
def predict_with_xgboost(reverse=False):  # reverse가 True인 경우, 적게 나온 combination을 출력.
    count_in = 500  # 입력으로 할 회차의 수
    values = [sum(i) for i in df]
    df_values = pd.DataFrame(values)
    slp = list()  # Supervised Learning Problem 데이터를 저장하기 위함.

    for i in range(count_in, 0, -1):
        slp.append(df_values.shift(i))
    slp.append(df_values)

    slp = pd.concat(slp, axis=1)
    slp.dropna(inplace=True)
    train = slp.values
    train_X, train_Y = train[:, :-1], train[:, -1]  # split into input and output columns.

    # fit model
    model = xgboost.XGBRegressor()
    model.fit(train_X, train_Y)

    # construct an input for a new prediction
    data_input = values[-count_in:]

    # make a one-step prediction
    predicted_sum = model.predict([data_input])[0]
    print(f'Predict: {predicted_sum}')
    predicted_sum = int(predicted_sum)

    sort_by_freq = most_picked_num(reverse=reverse, pick=45)
    nCr = itertools.combinations(sort_by_freq, 6)

    for i in nCr:
        if sum(i) == predicted_sum:
            return i
    return "combination not exists."

'''
    최신 로또 번호를 갱신하여 excel 파일에 저장합니다.
'''
def append_new_pick():
    lotto_num_url = 'https://search.naver.com/search.naver?where=nexearch&sm=top_hty&fbm=1&ie=utf8&query=%EB%A1%9C%EB%98%90+%EB%B2%88%ED%98%B8'
    req = requests.get(lotto_num_url)
    raw = req.text

    html = BeautifulSoup(raw, 'html.parser')

    inning_date = html.select_one('#main_pack > div.sc_new.cs_lotto._lotto > div > div.content_area > div > div > div.tab_area > div.type_flick_select._custom_select > div.select_tab > a.text._select_trigger._text').get_text()
    inning = int(inning_date[:4])

    if len(df) == inning:
        print("already recorded inning! No.%d" % inning)
    else:
        new_row = [inning]
        numbers_with_tag = html.select('#main_pack > div.sc_new.cs_lotto._lotto > div > div.content_area > div > div > div:nth-child(2) > div.win_number_box > div > div.winning_number > span')
        for text in numbers_with_tag:
            new_row.append(text.get_text())  # 일반 번호 추가

        bonus_num = html.select_one('#main_pack > div.sc_new.cs_lotto._lotto > div > div.content_area > div > div > div:nth-child(2) > div.win_number_box > div > div.bonus_number > span').get_text()
        new_row.append(bonus_num)  # 보너스 번호

        prize_amount = html.select_one('#main_pack > div.sc_new.cs_lotto._lotto > div > div.content_area > div > div > div:nth-child(2) > div.win_number_box > p > strong').get_text()
        prize_amount = int(prize_amount.replace(",", ""))
        winners_count = html.select_one('#main_pack > div.sc_new.cs_lotto._lotto > div > div.content_area > div > div > div:nth-child(2) > div.win_number_box > p').get_text()
        winners_count = winners_count[winners_count.index('('):]
        winners_count = int(re.findall(r'\d', winners_count)[0])
        new_row.append(winners_count)  # 1등 당첨 수
        new_row.append(prize_amount)   # 1등 당첨 금액
        new_row.append('')             # 로또 총 구매 금액

        date = inning_date[inning_date.index('('):]
        date = date.strip('()')
        date = date.replace('.', '-')
        new_row.append(date)  # 발표일

        new_df = imported_df.append(pd.Series(new_row, index=imported_df.columns), ignore_index=True)

        writer = pd.ExcelWriter('lotto_number.xlsx', engine='openpyxl')
        new_df.to_excel(writer, index=False)
        writer.save()

if __name__ == "__main__":
    append_new_pick()
    print(random_pick())
    print(most_picked_num())
    print(most_picked_num(reverse=True))
    print(predict_with_xgboost())
    print(predict_with_xgboost(reverse=True))
    exit()
