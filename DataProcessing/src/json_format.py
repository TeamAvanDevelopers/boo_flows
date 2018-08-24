# -*- coding: utf-8 -*-
import json

def load_to_json(load_path):
    """
    json 파일을 로드한다.
    """

    with open(load_path, 'r', encoding="utf-8") as file:
        return json.load(file)

def save_to_json(data, save_root, file_name):
    """
    데이터를 json 파일로 저장한다.
    """
    import time
    save_path = "{}\\{}.{}.json".format(save_root, file_name, int(time.time()))
    with open(save_path, 'w', encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent="\t")
    file.close()
    print(save_path + "에 저장 완료")