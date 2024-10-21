import os 
import VectorSpace
import tfidf
import math
import string
import numpy as np

from tqdm import tqdm 
from Parser import Parser

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob as tb
from collections import OrderedDict
import csv




lemmatizer = WordNetLemmatizer()

# def normalize_doc(doc):
#     parser = Parser()
#     vocabularyString = " ".join(doc)
#     #將字符串分割成單詞
#     vocabularyList = parser.tokenise(vocabularyString)
#     vocabularyList = parser.removeStopWords(vocabularyList)
#     return vocabularyList

def creatQuery(query,keys_string):      
    query_blob = tb(query)
    query_lower = query_blob.lower()
    query_vector = [tfidf.tf(word, query_lower) for word in keys_string]
    return query_vector

# def normalize_words(words):
#     lemmatized_words = []
#     for word in words:
#         # 使用 'v' 來指示單詞是動詞
#         lemmatized_word = lemmatizer.lemmatize(word, pos='v')
#         lemmatized_word = lemmatizer.lemmatize(word, pos='a')
#         lemmatized_words.append(lemmatized_word)
#     return lemmatized_words

def cosine_similarity(vecA, vecB):
    dot_product = sum(a * b for a, b in zip(vecA, vecB))
    magnitudeA = math.sqrt(sum(a ** 2 for a in vecA))
    magnitudeB = math.sqrt(sum(b ** 2 for b in vecB))
    if not magnitudeA or not magnitudeB:
        return 0.0
    return dot_product / (magnitudeA * magnitudeB)

def euclidean_distance(vecA, vecB):
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(vecA, vecB)))

def lemmatize_lower_sentence(sentence,stopwords):
    # Tokenize sentence
    words = word_tokenize(sentence)
    words = [word for word in words if word not in string.punctuation]
    words = [word.lower() for word in words if word.lower() not in stopwords]
    # 進行詞性還原
    lemmatized_words = []
    for word in words:
        lemmatized_word = lemmatizer.lemmatize(word, pos='v')
        lemmatized_words.append(lemmatized_word)
    return ' '.join(lemmatized_words)

def print_top_10(documents,smilarity_results):

    results = [(filename, sim) for (filename, sim) in zip(documents.keys(), smilarity_results)]
    top_10_results = sorted(results, key=lambda x: x[1], reverse=True)[:10]
    for filename, sim in top_10_results:
        print(f"{filename}: {sim:.4f}")
    converted_data = [int(item[0].replace('d', '').replace('.txt', '')) for item in top_10_results]
    return converted_data


# 1. 定義 MRR@10 函數
def calculate_mrr10(top10_predictions, relevant_docs, k=10): 
    rr_sum = 0
    num_queries = len(top10_predictions)  # 记录查询的数量
    for query, preds in top10_predictions.items():
        print("Query: ", query)
        print("Top 10 Predictions: ", preds)

        # 检查相关文档是否存在
        relevant_docs_for_query = relevant_docs.get(query, [])
        for rank, doc in enumerate(preds[:k]):  # 只考虑前 k 个预测
            if doc in relevant_docs_for_query:  # 检查文档是否是相关的
                print(f"Found relevant document: {doc} at rank {rank + 1}")
                rr_sum += 1 / (rank + 1)  # 加 1 是为了调整为 1-based rank
                break  # 找到第一个相关文档后就跳出

    # 避免除以零
    return rr_sum / num_queries if num_queries > 0 else 0
            

    # for i, pred in enumerate(predictions):
    #     for rank, doc in enumerate(pred[:k], 1):
    #         if doc in relevant_docs[i]:
    #             rr_sum += 1 / rank
    #             break
    

# 2. 定義 MAP@10 函數
def calculate_map(predictions, relevant_docs, k=10):
    ap_sum = 0
    for i, pred in enumerate(predictions):
        relevant = 0
        precision_sum = 0
        for rank, doc in enumerate(pred[:k], 1):
            if doc in relevant_docs[i]:
                relevant += 1
                precision_sum += relevant / rank
        if relevant > 0:
            ap_sum += precision_sum / relevant
    return ap_sum / len(predictions)

# 3. 定義 Recall@10 函數
def calculate_recall(predictions, relevant_docs, k=10):
    recall_sum = 0
    for i, pred in enumerate(predictions):
        relevant = len(relevant_docs[i])
        hits = len([doc for doc in pred[:k] if doc in relevant_docs[i]])
        recall_sum += hits / relevant
    return recall_sum / len(predictions)

def main():
    #TODO: 將 EnglishNews 中的文件讀取出來
    documents = {}
    for filename in os.listdir('smaller_dataset/collections'):
        if filename.endswith('.txt'):
            with open('smaller_dataset/collections/'+filename, 'r', encoding="utf-8") as f:
                documents[filename] = f.read()
    querys =  OrderedDict()
    for filename in os.listdir('smaller_dataset/queries'):
        if filename.endswith('.txt'):
            with open('smaller_dataset/queries/'+filename, 'r', encoding="utf-8") as f:
                querys[filename] = f.read()

    # 指定 TSV 文件的路徑
      # 替換為你的文件路徑

    # 打開 TSV 文件並讀取內容
    rel = {}
    with open('smaller_dataset/rel.tsv', 'r', encoding='utf-8') as file:
        for line in file:
            # 去掉行末的換行符號，並將行拆分為兩部分
            line = line.strip()
            if line:  # 確保行不為空
                key, value = line.split('\t')
                # 轉換字符串的列表形式為實際的列表
                # 去掉中括號並轉換為列表
                value_list = eval(value)  # 注意：eval() 可能有安全隱患，確保你的輸入是可信的
                # 將結果存入字典
                rel[key] = value_list

    print(rel)
    stopwords = open('english.stop', 'r').read().split()

    #TODO: (fixed) 處理文件 stemming and lower (文件應該也要處理 stop words?)
    lemmatized_lower_documents = [lemmatize_lower_sentence(doc,stopwords) for doc in documents.values()]
    # print("Lemmatized_Lower_Documents: ",lemmatized_lower_documents)

    #TODO: 處理 keys_target
    lemmatized_lower_documents_join = " ".join(lemmatized_lower_documents)
    all_words = lemmatized_lower_documents_join.split()

    # 組合回字符串形式
    keys_target =set( " ".join(all_words).split())
    # print("Keys_Target: ", keys_target)

    #TODO: 創建 Querys 向量
    # query = "Typhoon Taiwan war"
    normalize_querys = [lemmatize_lower_sentence(query,stopwords) for query in querys.values()]
    # for query in normalize_querys:
    #     print("Query: ",query)

    result_querys = [creatQuery(normalize_query,keys_target) for normalize_query in normalize_querys]
    # for result_query in result_querys:
    #     print("Result_Query: ",result_query)

    #TODO:  TF Weighting (Raw TF in course PPT) + Cosine Similarity
    tf_vectors = []
    for doc in tqdm(lemmatized_lower_documents, desc="Processing TF-Vectors"): #如何加速
        tf_vector = [tfidf.tf(word, doc) for word in keys_target]
        tf_vectors.append(tf_vector)
    # for vector in tf_vectors:
    #     print("tf_vector: ",vector)
    idf_vectors = []
    idf_vectors = [tfidf.idf(word,lemmatized_lower_documents) for word in keys_target]
        
    tfidf_vectors = []
    for tf_vector in  tqdm(tf_vectors, desc="Processing TFIDF-Vectors"):  #如何加速
        tfidf_vector = [tf * idf for tf, idf in zip(tf_vector, idf_vectors)]
        tfidf_vectors.append(tfidf_vector)

    tfidf_similarities_matrix = []
    tfidf_similarities = []
    for result_query in  tqdm(result_querys, desc="Processing TFIDF-Vectors similiarity"): #如何加速
        for tfidf_vector in tfidf_vectors:
            tfidf_similarity = cosine_similarity(result_query, tfidf_vector)
            tfidf_similarities.append(tfidf_similarity)
        tfidf_similarities_matrix.append(tfidf_similarities)
    # print(tfidf_similarities_matrix)

    # sorted_queries = sorted(querys.keys(), key=lambda x: int(x[1:]))  # 如果鍵是 q0, q1 這樣的格式，按數字排序
    predictions = {}
    for i ,query  in enumerate(querys.keys()):
        print("Query: ",query)
        print("TF-IDF Cosine")
        print("NewsID   Score")
        predictions[query.replace(".txt", "")] = print_top_10(documents,tfidf_similarities_matrix[i])

    print(predictions)
    
    # for i ,rel in enumerate(rel.keys()):
    #     print("rel",rel)

    # 現在有每個query的結果
    # 再來要找出每個query的前10名
    # 然後再找出每個query的 rank 1 的文件
    # 再根據這個文件做 MMR
    




    print("MRR@10")
    print(calculate_mrr10(predictions,rel))
    # print("MAP@10")
    # calculate_map()
    # print("RECALL@10")
    # calculate_recall()

if __name__ == "__main__":
    main()