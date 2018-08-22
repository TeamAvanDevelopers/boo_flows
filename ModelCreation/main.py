import time
from src.formatting import *
from src.modelCofig import *
from src.recommendationModel import *


rt = ResultDataFrame()
model_save_root = "C:\\Users\\User\\Desktop\\공동_작업_깃헙\\boo_flows\\ModelCreation\\res\\Model"
vocab_save_root = "C:\\Users\\User\\Desktop\\공동_작업_깃헙\\boo_flows\\ModelCreation\\res\\Vocab"
result_save_root = "C:\\Users\\User\\Desktop\\공동_작업_깃헙\\boo_flows\\ModelCreation\\res\\Result"

def main_tag2vec_and_realted2vec(result_idx):
    # 1. 데이터 불러오기
    df = rt.result_table("SELECT post_id, extracted_tags FROM db_recommendation_test.new_table where extracted_tags != \"\"")

    # 2. Vocab 생성
    sentencesReader = SentencesReader(df)

    # 모델 생성에 사용됨
    myVocab_list = sentencesReader.read_rows('extracted_tags')

    # 3. Voacb 저장
    save_vocab(myVocab_list, vocab_save_root, 'tag2vec')

    # 4-1. Tag2vec Model 생성
    tag2vec = Tag2Vec(myVocab_list)
    model = tag2vec.train(tag2vec_config)

    # 4-2. Tag2vec Model 저장
    save_model(model, model_save_root, 'tag2vec')

    # 4-3. Tag2vec Model을 이용한 결과 도출
    tags_list = tag2vec.wv.index2word
    recommend_tags = recommend_tags(tag2vec_model, tags_list, 10)
    print("recommend_tags length:", len(recommend_tags))

    tag2vec_result_create_time = time.time()
    tag2vec_result = related_tags_format(1, tag2vec_result_create_time, recommend_tags)
    save_to_json(tag2vec_result, result_save_root, "tag2vec_result")

    # 5-1. RelatedTag2vec Model 생성
    relatedTag2vec = Tag2Vec(myVocab_list)
    model = relatedTag2vec.train(relatedTags2Vec_config)

    # 5-2. RelatedTag2vec Model 저장
    save_model(model, model_save_root, 'relatedtag2vec')

    # 5-3. RealtedTag2Vec 모델을 이용한 결과 도출
    tags_list2 = relatedTag2vec.wv.index2word
    related_tags = get_related_tags(relatedtag2vec_model, tags_list2, 10)
    print("related_tags length:", len(related_tags))

    relatedtag2vec_result_create_time = time.time()
    relatedtag2vec_result = related_tags_format(result_idx, relatedtag2vec_result_create_time, related_tags)

    # 6. 도출 결과 저장
    save_to_json(relatedtag2vec_result, result_save_root, "relatedtag2vec_result")


def main_post2vec(result_idx):
    # 1. 데이터 불러오기
    df = rt.result_table("SELECT post_id, extracted_tags FROM db_recommendation_test.new_table where extracted_tags != \"\"")

    # 2. Vocab 생성
    taggedDocuments = TaggedDocuments(df)
    define_data_set = taggedDocuments.define_data_set('post_id', 'extracted_tags')

    # 모델 생성에 사용됨
    myVocab_list = taggedDocuments.get_custom_tagged_document(define_data_set)

    # 3. Voacb 저장
    save_vocab(myVocab_list, vocab_save_root, 'Post2vec')

    # 4. Post2Vec Model 생성
    post2Vec = Item2Vec(myVocab_list)
    model = post2Vec.train(post2vec_config)

    # 5. Model 저장
    save_model(model, model_save_root, 'Post2vec')

    # 6. 생성된 모델을 이용한 결과 도출
    post_id_key_list = df.post_id.tolist()
    recommend_posts = recommend_items(post2Vec, post_id_key_list, 10)
    print("recommend_posts length:", len(recommend_posts))
    print(type(recommend_posts))

    post2vec_result_create_time = time.time()
    post2vec_result = related_items_format(result_idx, post2vec_result_create_time, recommend_posts)

    # 7. 도출 결과 저장
    save_to_json(post2vec_result, result_save_root, "post2vec_result")


def main_user2vec(result_idx):
    # 1. 데이터 불러오기
    df = rt.result_table("SELECT user_id, extracted_tags FROM db_recommendation_test.new_table where extracted_tags != \"\"")

    # 2. Vocab 생성
    taggedDocuments = TaggedDocuments(df)
    define_data_set = taggedDocuments.define_data_set('user_id', 'extracted_tags')

    # 모델 생성에 사용됨
    myVocab_list = taggedDocuments.get_custom_tagged_document(define_data_set)

    # 3. Voacb 저장
    save_vocab(myVocab_list, vocab_save_root, 'User2vec')

    # 4. Model 생성
    user2Vec = Item2Vec(myVocab_list)
    model = user2Vec.train(user2vec_config)

    # 5. model 저장
    save_model(model, model_save_root,'User2vec')

    # 6. 생성된 모델을 이용한 결과 도출
    # df = rt.result_table("SELECT user_id FROM db_recommendation_test.new_table where extracted_tags != \"\"")
    user_id_key_list = df.user_id.tolist()
    recommend_users = recommend_items(user2Vec, user_id_key_list, 10)
    print("recommend_users length:", len(recommend_users))

    user2vec_result_create_time = time.time()
    user2vec_result = related_items_format(result_idx, user2vec_result_create_time, recommend_users)

    # 7. 도출 결과 저장
    save_to_json(user2vec_result, result_save_root, "user2vec_result")


def main(result_idx):
    main_tag2vec_and_realted2vec(result_idx)
    main_post2vec(result_idx)
    main_user2vec(result_idx)

main(1)
