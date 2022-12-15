import pandas as pd
import itertools
import requests
import random
import re
import matplotlib.pyplot as plt
import numpy as np
import xgboost
from bs4 import BeautifulSoup

imported_df = pd.read_csv("./lotto_number.csv")
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
    역대 가장 많이/적게 나온 번호 n개를 출력합니다.
    reverse: reverse가 True인 경우, 적게 나온 순으로 출력.
    pick: 번호 n개 출력.
    rank: 순위 n개 이하의 번호 중 랜덤 출력.
'''
def most_picked_num(reverse=False, pick=6, random_from_rank=0):
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
    if random_from_rank:  # 임의 추출 옵션
        count_lotto = [i[0] for i in count_lotto[:random_from_rank]]
        while len(result) < 6:
            result.append(random.choice(count_lotto))
            result = list(set(result))
        return result

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
    # print(f'Predict: {predicted_sum}')  // for check
    predicted_sum = int(predicted_sum)

    sort_by_freq = most_picked_num(reverse=reverse, pick=45)
    #region 가급적 빈도수 높은걸로 나올 수 있도록 개선했으나 차후 최적화가 필요함.
    cnt = 6
    while cnt <= 45:
        temp = [i for i in range(cnt)]
        nCr = itertools.combinations(temp, 6)

        for c in nCr:
            result = 0
            for index in c:
                result += sort_by_freq[index]
            if result == predicted_sum:
                # print(c)  // indexes
                return [sort_by_freq[i] for i in c]
        cnt += 1
    #endregion
    return "combination does not exist!"

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

def menu():
    while True:
        print("-----Select a menu-----\n1. 무작위 번호 6개 추출\n2. 최다 숫자 기준 추출\n3. 최저 숫자 기준 추출")
        print("4. 머신러닝을 통한 6개 추출\n5. 머신러닝을 통한 안나올법한 6개 추출\n6. 로또번호 DB갱신\n7. 통계 확인\n0. 종료")
        sel = input("골라주세요!: ")
        try:
            sel = int(sel)
            if sel == 0:
                print("프로그램을 종료합니다.")
                return

            elif sel == 1:  # 6개의 무작위 번호
                print(random_pick())

            elif sel == 2:  # 가장 많이 나온 숫자 10위 이하의 임의의 6개 번호
                print("역대 가장 많이 나온 숫자 n위 이하의 번호를 추출합니다.")
                position = int(input("n값을 입력해주세요: "))
                print(most_picked_num(random_from_rank=position))

            elif sel == 3:  # 뒤에서 10위 까지의 임의의 6개 번호
                print("역대 가장 적게 나온 숫자 n위 이하의 번호를 추출합니다.")
                position = int(input("n값을 입력해주세요: "))
                print(most_picked_num(reverse=True, random_from_rank=position))

            elif sel == 4:  # 머신러닝을 통한 많이 나올법한 6개 번호 예측
                print(predict_with_xgboost())

            elif sel == 5:  # 머신러닝을 통한 가장 안나올법한 6개 번호 예측
                print(predict_with_xgboost(reverse=True))

            elif sel == 6:
                append_new_pick()

            elif sel == 7:
                print_stastics()

            else:
                print("잘못 입력하셨습니다.")

        except Exception as e:
            print(e)



if __name__ == "__main__":
    menu()
    exit()
