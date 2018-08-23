# -*- coding: utf-8 -*-
import logging
import time
import json

from gensim.models import Word2Vec
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from numpy.core.defchararray import lower


"""
1. 모델 생성을 위한 단어장을 만들어보자!
"""

# Item2Vec용 단어장 생성을 위한 클래스 #리팩 필요!
class TaggedDocuments:

    def __init__(self, df):
        self.df = df

    # space로 구분된 추출 태그를 단어장 생성을 위한 input 형태로 만들어주는 함수이다.
    # {1111111: ["안녕 나는 지금 퇴근 중이야"]}
    def define_data_set(self, col_name_idx, col_name_extracted_tags):
        tmp = self.df.groupby(col_name_idx)[col_name_extracted_tags].agg(lambda col: ' '.join(col))
        defined_data_set = dict()
        for item in tmp.items():
            # 혹시 모를 공백을 위해. 공백을 가진 item에 대해서는 모델 학습을 시키지 않는다.
            if item[1] != "":
                defined_data_set[item[0]] = item[1]
            else:
                pass
        return defined_data_set

    # Item2Vec 모델 생성시 사용될 단어장인 tagged_documents를 만든다.
    def get_custom_tagged_document(self, data_set):
        custom_tagged_document = []
        for idx in data_set.keys():
            split_tags = data_set[idx].split(" ")
            custom_tagged_document.append(TaggedDocument(split_tags, [str(idx)]))
        return custom_tagged_document


# Tag2Vec용 단어장 생성을 위한 클래스 #리팩 필요!
class SentencesReader:
    def __init__(self, df):
        self.df = df

    # 문장 리스트로 읽기
    def read_rows(self, col_name_extracted_tags):
        custom_sentences = []
        for row in self.df[col_name_extracted_tags].iteritems():
            if row[1] != "":
                custom_sentences.append([w for w in row[1].split(' ')])
        return custom_sentences


# 생성된 단어장 저장을 위한 것. # DB를 사용하니깐 굳이.. 필요할까?
def save_vocab(vocab_name, save_root, vocab_type):
    vocab_types = ['user2vec', 'post2vec', 'tag2vec', 'relatedtag2vec']
    lower_vocab_type = lower(vocab_type)
    if lower_vocab_type in vocab_types:
        save_path = "{}\\{}.{}.vocab".format(save_root, int(time.time()), lower_vocab_type)
        with open(save_path, 'w', encoding="utf-8") as data_file:
            json.dump(vocab_name, data_file, ensure_ascii=False, indent="\t")
        data_file.close()
        print(save_path + "에 저장 완료")

    else:
        print("Check your vocab type. You can use only 4 types \
              like user2vec, post2vec, tag2Vec, relatedtag2vec")


"""
2. 생성된 단어장을 이용하여 모델을 생성해보자!
"""

class Item2Vec:

    def __init__(self, tagged_documents):
        self.tagged_documents = tagged_documents

    # Doc2Vec 학습모델 생성
    def train(self, config):
        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                            level=logging.INFO)
        start = time.time()
        model = Doc2Vec(**config)
        model.build_vocab(self.tagged_documents)

        # epoch 설정 후 학습
        for epoch in range(model.epochs):
            model.train(self.tagged_documents, total_examples=len(self.tagged_documents),
                        epochs=model.epochs)
            model.alpha -= 0.002  # decrease the learning rate
            model.min_alpha = model.alpha  # fix the learning rate, no decay
        end = time.time()
        print("During Time: {}".format(end - start))
        return model

    # updated_tagged_documents #솔찍히 필요 없을듯 -_-
    def update(self, model):
        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                            level=logging.INFO)
        model.build_vocab(self.tagged_documents, update=True)
        # epoch 설정 후 학습
        start = time.time()
        for epoch in range(model.epochs):
            model.train()
            model.alpha -= 0.002  # decrease the learning rate
            model.min_alpha = model.alpha  # fix the learning rate, no decay
        end = time.time()
        print("During Time: {}".format(end - start))
        return model

class Tag2Vec:

    def __init__(self, sentences):
        self.sentences = sentences

    # Word2Vec 학습모델 생성
    def train(self, config):
        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                            level=logging.INFO)
        start = time.time()
        model = Word2Vec(**config)
        model.build_vocab(self.sentences)
        model.train(self.sentences, total_examples=len(self.sentences), epochs=model.iter)
        end = time.time()
        print("During Time: {}".format(end - start))
        return model

    def update(self, model):
        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',\
                            level=logging.INFO)
        # model = Word2Vec.load(model_load_path)
        start = time.time()
        model.build_vocab(self.sentences, update=True)
        model.train()
        end = time.time()
        print("During Time: {}".format(end - start))
        return model


# model.wv.vocab <- 중복 제거된 단어장을 얻을 수 있다. 반환시 list로 반환하면 편하다. 원래는 dict
def save_model(model, save_root, model_type):
    model_types = ['user2vec', 'post2vec', 'tag2vec', 'relatedtag2vec']
    lower_model_type = lower(model_type)
    if lower_model_type in model_types:
        save_path = "{}\\{}.{}.model".format(save_root, int(time.time()), lower_model_type)
        model.save(save_path)
        print(save_path + "에 저장 완료")
    else:
        print("Check your model type. You can use only 4 types \
              like user2vec, post2vec, tag2Vec, relatedtag2vec")


"""
3. 모델을 이용하여 결과를 도출해보자!
"""
# 저장 모델 불러오기. 공통적으로 사용된다.
def load_model(load_path):
    if load_path.find('user2vec') > 0 or load_path.find('post2vec') > 0:
        model = Doc2Vec.load(load_path)
        return model
    elif load_path.find('tag2vec') > 0 or load_path.find('relatedtags2vec') > 0:
        model = Word2Vec.load(load_path)
        return model
    else:
        return print("Check your path! The current path is {}".format(load_path))


# {사용자 번호: {유사 사용자 1 : 유사도, 유사 사용자 2 : 유사도, , etc}}
# {게시물 번호: {유사 게시물 1 : 유사도, 유사 게시물 2 : 유사도, , etc}}
# {'8156815234': {'3056895722': 0.6550565958023071, '8145042343': 0.5274348258972168}
def get_most_similar_items(model, item_id, n):
    item_list = model.docvecs.most_similar(str(item_id), topn=n)
    item_dict = dict(item_list)
    similar_items_dict = {item_id: item_dict}
    return similar_items_dict


def get_most_similar_items_without_sim(model, item_id, n):
    item_list = [item for item, sims in model.docvecs.most_similar(str(item_id), topn=n)]
    similar_items_dict = {item_id: item_list}

    #item_list = model.docvecs.most_similar(str(item_id), topn=n)
    #item_dict = dict(item_list)
    #similar_items_dict = {item_id: list(item_dict.keys())}
    return similar_items_dict

# 태그: {유사 태그 1: 유사도, 유사 태그 2: 유사도}
def get_most_similar_tags(model, tag, n):
    tags_list = model.wv.most_similar(tag, topn=n)
    tags_dict = dict(tags_list)
    similar_tags_dict = {tag: tags_dict}
    return similar_tags_dict

# 태그: {유사 태그 1, 유사 태그 2}
def get_most_similar_tags_without_sim(model, tag, n):
    tags_list = [tag for tag, sims in model.wv.most_similar(tag, topn=n)]
    similar_tags_dict = {tag: tags_list}
    return similar_tags_dict

# 생성된 모델을 이용해 결과를 도출시 사용하는 함수 (특정 tag_list해 해당하는 내용에 대해서 연관 태그를 추출한다.)
def recommend_tags(model, tags_list, n):
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                        level=logging.INFO)
    start = time.time()
    recommendation_tags_dict = dict()
    cnt = 0
    for tag in tags_list:
        cnt += 1
        recommendation_tags_dict.update(get_most_similar_tags_without_sim(model, tag, n))
    end = time.time()
    print("During Time: {}, Total Length: {}".format((end - start), cnt))
    return recommendation_tags_dict

# 생성된 모델을 이용해 결과를 도출시 사용하는 함수 (특정 tag_list해 해당하는 내용에 대해서 연관 아이템을 추출한다.)
def recommend_items(model, items_list, n):
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                        level=logging.INFO)
    start = time.time()
    recommendation_items_dict = dict()
    cnt = 0
    for item in items_list:
        cnt += 1
        recommendation_items_dict.update(get_most_similar_items_without_sim(model, item, n))
    end = time.time()
    print("During Time: {}, Total Length: {}".format((end - start), cnt))
    return recommendation_items_dict

# 생성된 모델을 이용해 연관 태그 결과를 도출시 사용하는 함수 (특정 tag_list해 해당하는 내용에 대해서 연관 아이템을 추출한다.)
def get_related_tags(model, tags_list):
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                        level=logging.INFO)
    start = time.time()

    related_tags_dict = dict()
    cnt = 0
    # 연관 태그는 항상 20개를 뽑는다. 20개가 없떡하지? #리팩 필요?
    for tag in tags_list:
        cnt += 1
        related_tags_dict.update(get_most_similar_tags_without_sim(model, tag, 20))
    end = time.time()
    print("During Time: {}, Total Length: {}".format((end - start), cnt))
    return related_tags_dict


"""
4. 생성된 모델을 이용하여 신규 생성된 아이침 & 태그에 대한 추론을 하겠다. (***진행중**)
"""
# 추가해야 할 함수 1. 아이템 추론하기.
def infer_items(model, labeled_document):
    pass

# 추가해야 할 함수 2. 태그 추론하기.
def infer_tags(model, labeled_document):
    pass
