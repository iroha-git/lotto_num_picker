import pandas as pd
import requests
import re
import tensorflow as tf
from bs4 import BeautifulSoup

imported_df = pd.read_excel("./lotto_number.xlsx")
df = imported_df.iloc[:, [1, 2, 3, 4, 5, 6, 7]]
df = df.values.tolist()  # 모든 7개의 로또 번호 리스트

'''
    역대 가장 많이 나온 번호 6개를 출력합니다.
'''
def max_picked_num():
    count_lotto = dict()

    for i in range(1, 46):
        count_lotto[i] = 0

    for numbers in df:
        for number in numbers:
            count_lotto[number] += 1

    count_lotto = sorted(count_lotto.items(), key=lambda x: x[1], reverse=True)

    result = list()
    for i in range(6):
        result.append(count_lotto[i][0])

    print(result)

'''
    딥러닝 기반으로 학습된 새로운 모델을 생성합니다.
'''
def train_new_model():
    pass


'''
    딥러닝 모델을 통해 로또 번호를 예측합니다.
'''
def predict_with_model():
    pass

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

        bonus_num = html.select_one('#main_pack > div.sc_new.cs_lotto._lotto > div > div.content_area > div > div > div:nth-child(2) > div.win_number_box > div > div.winning_number > span').get_text()
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