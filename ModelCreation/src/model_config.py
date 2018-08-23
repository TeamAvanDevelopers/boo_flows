# -*- coding: utf-8 -*-
import multiprocessing


# model별 config이다.
USER2VEC_CONFIG= {
    'dm': 1,  # PV-DBOW / default 1
    'dbow_words': 1,  # w2v simultaneous with DBOW d2v / default 0
    'window': 7,  # distance between the predicted word and context words
    'vector_size': 200,  # vector size
    'alpha': 0.025,  # learning-rate
    'sample': 1e-4,
    'seed': 1234,
    'min_count': 0,  # ignore with freq lower
    'min_alpha': 0.025,  # min learning-rate
    'workers': multiprocessing.cpu_count(),  # multi cpu
    'hs': 1,  # hierarchical softmax / default 0
    'negative': 5,  # negative sampling / default 5
    'epochs': 10  # 보통 딥러닝에서 말하는 epoch과 비슷한, 반복 횟수. 또한 모델 학습시 반복 횟수 만큼 반복된다!!!
    }


POST2VEC_CONFIG = {
    'dm': 1,  # PV-DBOW / default 1
    'dbow_words': 1,  # w2v simultaneous with DBOW d2v / default 0
    'window': 4,  # distance between the predicted word and context words
    'vector_size': 200,  # vector size
    'alpha': 0.025,  # learning-rate
    'sample': 1e-4,
    'seed': 1234,
    'min_count': 0,  # ignore with freq lower
    'min_alpha': 0.025,  # min learning-rate
    'workers': multiprocessing.cpu_count(),  # multi cpu
    'hs': 1,  # hierarchical softmax / default 0
    'negative': 5,  # negative sampling / default 5
    'epochs': 10  # 보통 딥러닝에서 말하는 epoch과 비슷한, 반복 횟수. 또한 모델 학습시 반복 횟수 만큼 반복된다!!!
    }


TAG2VEC_CONFIG = {
    'sg': 0, # CBOW
    'window': 6,  # distance between the predicted word and context words
    'size': 200,  # vector size
    'batch_words': 10000,
    'iter': 10,  # 보통 딥러닝에서 말하는 epoch와 비슷한, 반복 횟수
    'min_count': 2,  # ignore with freq lower
    'workers': multiprocessing.cpu_count()  # multi cpu. 1이 속도가 느리지만 그나마.. 메모리를 적게 먹는다.
    }


REALTEDTAG2VEC_CONFIG = {
    'sg': 1, # skip-gram
    'window': 6,  # distance between the predicted word and context words
    'size': 200,  # vector size
    'batch_words': 10000,
    'iter': 10,  # 보통 딥러닝에서 말하는 epoch와 비슷한, 반복 횟수
    'min_count': 0,  # ignore with freq lower
    'workers': multiprocessing.cpu_count() # multi cpu. 1이 속도가 느리지만 그나마.. 메모리를 적게 먹는다.
    }
