# -*- coding: utf-8 -*-
"""
random, numpy 패키지를 사용하여 사용자 정보를 만들어준다.
"""
import random
import numpy as np


class UserInfo:
    """
    사용자 정보를 만들어볼까!
    """
    def __init__(self, user_id):
        self.user_id = user_id

    @staticmethod
    def get_influencer(total_posts_count, total_like_count, total_comment_count, follower_count):
        """
        사용자의 영향력을 계산하는 함수
        """
        molecular = ((total_like_count * 0.9) + (total_comment_count * 0.1)) / total_posts_count
        denominator = follower_count
        influencer = (molecular / denominator) if denominator != 0 else 0
        return influencer

    @staticmethod
    def get_following_list(users_list, tags_list):
        """
        사용자가 최근 팔로우한 10명의 사용자, 10개의 태그 목록을 반환하는 함수
        """
        following_users = random.sample(users_list, 10)
        following_tags = random.sample(tags_list, 10)
        return (following_users, following_tags)

    @staticmethod
    def get_like_posts_list(posts_list):
        """
        사용자가 좋아요를 누른 게시물 10개의 목록을 반환하는 함
        """
        return random.sample(posts_list, 10)

    @staticmethod
    def set_user_basic_info():
        """
        연산에 필요한 변수들을 정의한다.
        최근 30일간 포스팅한 게시물 수, 사용자가 좋아요한 게시물,
        각 게시물에 포함된 각 좋아요/코멘트 수의 합, 팔로워, 팔로잉 수,
        * 사용자가 좋아요한 게시물: 최근 12개 게시물에 대한 것.
        """
        total_posts_count = np.random.randint(100, size=1)[0]
        total_like_count = np.random.randint(20, size=1)[0]
        total_comment_count = np.random.randint(150, size=1)[0]
        follower_count = np.random.randint(300, size=1)[0]
        user_basic_info = (total_posts_count, total_like_count,
                           total_comment_count, follower_count)
        return user_basic_info

    def get_user_info(self, users_list, posts_list, tags_list):
        """
        influencer: total_posts_count, total_like_count, total_comment_count, follower_count 연산 필요.
        user_info: 최종 반환될 사용자 정보.
        """
        import time
        created_time = int(time.time()) #사용자 정보를 생성한 시간
        user_basic_info = self.set_user_basic_info()
        following_list = self.get_following_list(users_list, tags_list)
        follow_users = following_list[0]
        follow_tags = following_list[1]
        like_posts = self.get_like_posts_list(posts_list)
        influencer = self.get_influencer(user_basic_info[0], user_basic_info[1],
                                         user_basic_info[2], user_basic_info[3])
        user_info = (self.user_id, created_time, follow_users,
                     follow_tags, like_posts, influencer)
        return user_info


def format_user_info_for_recommendation(user_information):
    """
    최종 반환될 추천시 사용되는 사용자 정보 포맷으로 바꾸는 함수
    """
    user_information_dict = {'user': user_information[0],
                             'createdTime': user_information[1],
                             'follow_users': user_information[2],
                             'follow_tags': user_information[3],
                             'like_posts': user_information[4],
                             'influencer': user_information[5]}
    return user_information_dict


def deduplicated_all_tags(df):
    """
    태그 
    """
    t_list= []
    for row in df:
        t_list.append(row.split(" "))
      
    n_t_list = []   
    for i in t_list:
        n_t_list += i
        
    return list(set(n_t_list))