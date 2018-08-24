# -*- coding: utf-8 -*-
"""
pandas에서 sql을 사용하려면 sqlalchemy 패키지를 사용해야 한다.
"""
from sqlalchemy import create_engine
import pandas as pd


class ResultDataFrame:
    """
    1. (Local) MySQL에 접속해 결과를 도출하는 클래스이다. 따로 연결을 끊는게 없다.. 리팩 필요!
    """
    def __init__(self):
        # DB URL 설정
        self.db_uri = "mysql+pymysql://root:1234@127.0.0.1:3306/db_recommendation_test"

    def connect_db(self):
        return create_engine(self.db_uri)

        # 쿼리 날려서 데이터를 'DataFrame'형태로 만들기.
    def result_table(self, query):
        return pd.read_sql_query(query, con=self.connect_db())


