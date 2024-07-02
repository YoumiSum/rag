from transformers import AutoTokenizer, AutoModel
from transformers import BertTokenizer, BertModel, BertForSequenceClassification


import numpy as np
import pickle
import json

import torch


device = torch.device("cuda:0")

with open(r"train_similar_model\id_vector", "rb") as f:
    faiss_index = pickle.load(f)


model_path = r"train_similar_model\my_model"
tokenizer = BertTokenizer.from_pretrained(model_path)
recall_model = BertModel.from_pretrained(model_path)
rank_model = BertForSequenceClassification.from_pretrained(r'train_similar_model\rank_model')

chat_tokenizer = AutoTokenizer.from_pretrained(r"train_similar_model\models\ChatGLM-6B", trust_remote_code=True)
chat_model = AutoModel.from_pretrained(r"train_similar_model\models\ChatGLM-6B", trust_remote_code=True).cuda()
chat_model = chat_model.to(device)
chat_model = chat_model.half()

rank_model = rank_model.to(device)
recall_model = recall_model.to(device)


def get_similar_query(query, num=3):
    results = []
    for _ in range(0, num):
        # 大模型进行改写
        response, _ = chat_model.chat(chat_tokenizer, query+"。改写上述话", history=[], do_sample=True, num_beams=3, temperature=2.0)
        results.append(response)

    return results


def read_knowledge(path):
    with open(path, encoding="utf-8") as f:
        lines = f.readlines()

    data = [json.loads(line.strip()) for line in lines]
    id_desc = {}
    for s in data:
        id = s['id']
        id_desc[id] = s

    return id_desc


def normal(vector):
    vector = vector.tolist()[0]
    ss = sum([s**2 for s in vector])**0.5

    return [s/ss for s in vector]


def get_vector(sentence):
    encoded_input = tokenizer([sentence], padding=True, truncation=True, return_tensors='pt')
    encoded_input = encoded_input.to(device)
    # Compute token embeddings
    with torch.no_grad():
        model_output = recall_model(**encoded_input)

    # Perform pooling. In this case, mean pooling.
    sentence_embeddings = normal(model_output.pooler_output)
    sentence_embeddings = np.array([sentence_embeddings])

    return sentence_embeddings


def get_candidate(input, num=20):
    vector = get_vector(input)
    D, I = faiss_index.search(vector, num)
    D = D[0]
    I = I[0]
    indexs = []
    for d, i in zip(D, I):
        indexs.append(i)

    return indexs


def rank_sentence(query, sentences):
    X = [[query[0:200], sentence[0:200]] for sentence in sentences]
    X = tokenizer(X, padding=True, truncation=True, max_length=512, return_tensors='pt')
    X = X.to(device)
    scores = rank_model(**X).logits
    scores = torch.softmax(scores, dim=-1).tolist()
    scores = [round(s[1], 3) for s in scores]
    return scores


def rag_recall(query):
    similar_querys = get_similar_query(query)
    index_score = {}
    for input in [query] + similar_querys:
        indexs = get_candidate(input, num=30)
        sentences = [id_knowledge[index]['病情描述'] for index in indexs]
        scores = rank_sentence(input, sentences)
        for index, score in zip(indexs, scores):
            if score < 0.9:
                continue
            index_score[index] = index_score.get(index, 0.0) + score

    results = sorted(index_score.items(), key=lambda s: s[1], reverse=True)

    return results[0:3]


def get_prompt(recall_result):

    prompt = ""
    # 知识的id，召回的分数
    for i, [recall_id, recall_score] in enumerate(recall_result):
        prompt += "案例{}：".format(i) + "病情描述：" + id_knowledge[recall_id]['病情描述'] + "治疗方案:" + id_knowledge[recall_id]['治疗方案'] + "。"

    return prompt
 

id_knowledge = read_knowledge("knowledge")

while True:
    query = input("输入症状: ")
    recall_result = rag_recall(query)
    # 参考经验
    prompt = get_prompt(recall_result)
    response, _ = chat_model.chat(chat_tokenizer, prompt + "根据上述治疗方案，给出下述病情的治疗方案" + query, history=[])
    print(response)

