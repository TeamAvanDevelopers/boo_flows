import time
import src.formatting as fm
import src.model_config as mc
import src.vector_model as rm

from src.formatting import ResultDataFrame
from src.vector_model import SentencesReader, TaggedDocuments, Tag2Vec, Item2Vec



MODEL_SAVE_ROOT = "C:\\Users\\User\\Desktop\\공동_작업_깃헙\\boo_flows\\ModelCreation\\res\\Model"
VOCAB_SAVE_ROOT = "C:\\Users\\User\\Desktop\\공동_작업_깃헙\\boo_flows\\ModelCreation\\res\\Vocab"
RESULT_SAVE_ROOT = "C:\\Users\\User\\Desktop\\공동_작업_깃헙\\boo_flows\\ModelCreation\\res\\Result"


def main_tag2vec_and_realted2vec(create_time):
    # 1. 데이터 불러오기
    rdf = ResultDataFrame()
    query = "SELECT extracted_tags FROM db_recommendation_test.new_table \
            where extracted_tags != \"\""
    df = rdf.result_table(query)

    # 2. Vocab 생성
    sentences_reader = SentencesReader(df)

    # 모델 생성에 사용됨
    vocab = sentences_reader.read_rows('extracted_tags')

    # 3. Voacb 저장
    rm.save_vocab(vocab, VOCAB_SAVE_ROOT, 'tag2vec+relatedtag2vec')


    # 태그와 연관태그 모델을 공통적으로 Tag2Vec클래스를 공유함. 단어장도 동일함.
    tag_and_related_tags = Tag2Vec(vocab)

    # 4-1. Tag2vec Model 생성
    tag2vec_model = tag_and_related_tags.train(mc.TAG2VEC_CONFIG)

    # 4-2. Tag2vec Model 저장
    rm.save_model(tag2vec_model, MODEL_SAVE_ROOT, 'tag2vec')

    # 4-3. Tag2vec Model을 이용한 결과 도출
    tag2vec_word_list = tag2vec_model.wv.index2word
    recommend_tags = rm.recommend_tags(tag2vec_model, tag2vec_word_list, 10)
    #print("recommend_tags length:", len(recommend_tags))

    tag2vec_result = fm.related_tags_format(create_time, recommend_tags)
    fm.save_to_json(tag2vec_result, RESULT_SAVE_ROOT, "tag2vec_result")

    # 5-1. RelatedTag2vec Model 생성
    relatedtag2vec_model = tag_and_related_tags.train(mc.REALTEDTAG2VEC_CONFIG)

    # 5-2. RelatedTag2vec Model 저장
    rm.save_model(relatedtag2vec_model, MODEL_SAVE_ROOT, 'relatedtag2vec')

    # 5-3. RealtedTag2Vec 모델을 이용한 결과 도출
    relatedtag2vec_word_list = relatedtag2vec_model.wv.index2word
    related_tags = rm.get_related_tags(relatedtag2vec_model, relatedtag2vec_word_list)
    #print("related_tags length:", len(related_tags))

    relatedtag2vec_result = fm.related_tags_format(create_time, related_tags)
    fm.save_to_json(relatedtag2vec_result, RESULT_SAVE_ROOT, "relatedtag2vec_result")


def main_post2vec(create_time):
    # 1. 데이터 불러오기
    rdf = ResultDataFrame()
    query = "SELECT post_id, extracted_tags FROM db_recommendation_test.new_table \
            where extracted_tags != \"\""
    df = rdf.result_table(query)

    # 2. Vocab 생성
    tagged_documents = TaggedDocuments(df)
    define_data_set = tagged_documents.define_data_set('post_id', 'extracted_tags')

    # 모델 생성에 사용됨
    vocab = tagged_documents.get_custom_tagged_document(define_data_set)

    # 3. Voacb 저장
    rm.save_vocab(vocab, VOCAB_SAVE_ROOT, 'Post2vec')

    # 4. Post2Vec Model 생성
    post2Vec = Item2Vec(vocab)
    post2Vec_model = post2Vec.train(mc.POST2VEC_CONFIG)

    # 5. Model 저장
    rm.save_model(post2Vec_model, MODEL_SAVE_ROOT, 'Post2vec')

    # 6. 생성된 모델을 이용한 결과 도출
    post_id_key_list = df['post_id'].tolist()
    recommend_posts = rm.recommend_items(post2Vec_model, post_id_key_list, 10)
    #print("recommend_posts length:", len(recommend_posts))

    # 7. 도출 결과 저장
    post2vec_result = fm.related_items_format(create_time, recommend_posts)
    fm.save_to_json(post2vec_result, RESULT_SAVE_ROOT, "post2vec_result")


def main_user2vec(create_time):
    # 1. 데이터 불러오기
    rdf = ResultDataFrame()
    query = "SELECT user_id, extracted_tags FROM db_recommendation_test.new_table \
            where extracted_tags != \"\""
    df = rdf.result_table(query)

    # 2. Vocab 생성
    tagged_documents = TaggedDocuments(df)
    define_data_set = tagged_documents.define_data_set('user_id', 'extracted_tags')

    # 모델 생성에 사용됨
    vocab = tagged_documents.get_custom_tagged_document(define_data_set)

    # 3. Voacb 저장
    rm.save_vocab(vocab, VOCAB_SAVE_ROOT, 'User2vec')

    # 4. Model 생성
    user2Vec = Item2Vec(vocab)
    user2Vec_model = user2Vec.train(mc.USER2VEC_CONFIG)

    # 5. model 저장
    rm.save_model(user2Vec_model, MODEL_SAVE_ROOT, 'User2vec')

    # 6. 생성된 모델을 이용한 결과 도출
    user_id_key_list = df['user_id'].tolist()
    recommend_users = rm.recommend_items(user2Vec_model, user_id_key_list, 10)
    #print("recommend_users length:", len(recommend_users))

    # 7. 도출 결과 저장
    user2vec_result = fm.related_items_format(create_time, recommend_users)
    fm.save_to_json(user2vec_result, RESULT_SAVE_ROOT, "user2vec_result")


def main():
    create_time = int(time.time())
    main_tag2vec_and_realted2vec(create_time)
    main_post2vec(create_time)
    main_user2vec(create_time)


main()
