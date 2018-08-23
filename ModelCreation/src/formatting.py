# -*- coding: utf-8 -*-
from sqlalchemy import create_engine
import pandas as pd


"""
1. (Local) MySQL에 접속해 결과를 도출하는 클래스이다. 따로 연결을 끊는게 없다.. 리팩 필요!
"""

class ResultDataFrame:

    def __init__(self):
        # DB URL 설정
        self.db_uri = "mysql+pymysql://root:1234@127.0.0.1:3306/db_recommendation_test"

    def connect_db(self):
        return create_engine(self.db_uri)

        # 쿼리 날려서 데이터를 'DataFrame'형태로 만들기.
    def result_table(self, query):
        return pd.read_sql_query(query, con=self.connect_db())


"""
2. Json 형태의 데이터를 읽고, 저장하는 함수
"""

def load_to_json(load_path):
    import json
    with open(load_path, 'r', encoding="utf-8") as file:
        return json.load(file)

def save_to_json(data, save_root, file_name):
    import json
    import time
    save_path = "{}\\{}.{}.json".format(save_root, file_name, int(time.time()))
    with open(save_path, 'w', encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent="\t")
    file.close()
    print(save_path + "에 저장 완료")


"""
3. 도출된 결과를 특정 포맷에 맞춰 저장하는 함수 패키지.
"""

def related_tags_format(created_time, related_tags_dict):
    return dict({"created_time": created_time, 'related_tags': related_tags_dict})

def related_items_format(created_time, related_items_dict):
    return dict({"created_time": created_time, 'related_items': related_items_dict})
