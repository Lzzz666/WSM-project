import os 
import tfidf
import math
import string
import numpy as np
import csv

from nltk.stem import PorterStemmer
from tqdm import tqdm 
from nltk.tokenize import word_tokenize
from textblob import TextBlob as tb


ps = PorterStemmer()

import nltk
nltk.download('punkt_tab')
nltk.download('wordnet')


def creatQuery(query,keys_string):      
    query_blob = tb(query)
    query_lower = query_blob.lower()
    query_vector = [tfidf.tf(word, query_lower) for word in keys_string]
    return query_vector

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
    stem_docs = []
    for word in words:
        stem_word = ps.stem(word)
        stem_docs.append(stem_word)
    return ' '.join(stem_docs)

def return_top_10(documents,smilarity_results):

    results = [(filename, sim) for (filename, sim) in zip(documents.keys(), smilarity_results)]
    top_10_results = sorted(results, key=lambda x: x[1], reverse=True)[:10]
    converted_data = [int(item[0].replace('d', '').replace('.txt', '')) for item in top_10_results]
    return converted_data


# 1. 定義 MRR@10 函數
def calculate_mrr10(top10_predictions, relevant_docs, k=10): 
    rr_sum = 0
    num_queries = len(top10_predictions)
    for query, preds in top10_predictions.items():
        relevant_docs_for_query = relevant_docs.get(query, [])
        for rank, doc in enumerate(preds[:k]):
            if doc in relevant_docs_for_query:
                rr_sum += 1 / (rank + 1)  
                break  

    # 避免除以零
    return rr_sum / num_queries if num_queries > 0 else 0
            


# 2. 定義 MAP@10 函數
def calculate_map(top10_predictions, relevant_docs, k=10):
    ap_sum = 0.0  # 平均精度总和
    num_queries = len(top10_predictions)  # 记录查询的数量

    for query, preds in top10_predictions.items():
        relevant_docs_for_query = relevant_docs.get(query, [])
        relevant_count = 0
        precision_sum = 0.0
        for rank, doc in enumerate(preds[:k]):  
            if doc in relevant_docs_for_query:  
                relevant_count += 1
                precision_sum += relevant_count / (rank + 1)  
        ap = precision_sum / relevant_count if relevant_count > 0 else 0.0
        ap_sum += ap
    return ap_sum / num_queries if num_queries > 0 else 0.0

# 3. 定義 Recall@10 函數
def calculate_recall(top10_predictions, relevant_docs, k=10):
    recall_sum = 0.0 
    num_queries = len(top10_predictions) 

    for query, preds in top10_predictions.items():
        relevant_docs_for_query = relevant_docs.get(query, [])
        total_relevant_count = len(relevant_docs_for_query) 
        retrieved_relevant_count = 0  
        for doc in preds[:k]:  
            if doc in relevant_docs_for_query: 
                retrieved_relevant_count += 1
        recall = (retrieved_relevant_count / total_relevant_count) if total_relevant_count > 0 else 0.0
        recall_sum += recall

    return recall_sum / num_queries if num_queries > 0 else 0.0

def main():
    #TODO: 將 EnglishNews 中的文件讀取出來
    documents = {}
    for filename in os.listdir('smaller_dataset/collections'):
        if filename.endswith('.txt'):
            with open('smaller_dataset/collections/'+filename, 'r', encoding="utf-8") as f:
                documents[filename] = f.read()
    querys =  {}
    for filename in os.listdir('smaller_dataset/queries'):
        if filename.endswith('.txt'):
            with open('smaller_dataset/queries/'+filename, 'r', encoding="utf-8") as f:
                querys[filename] = f.read()

    rel = {}
    with open('smaller_dataset/rel.tsv', 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if line: 
                key, value = line.split('\t')
                value_list = eval(value) 
                rel[key] = value_list

    stopwords = open('english.stop', 'r').read().split()

    #TODO: (fixed) 處理文件 stemming and lower (文件應該也要處理 stop words?)
    lemmatized_lower_documents = [lemmatize_lower_sentence(doc,stopwords) for doc in documents.values()]

    #TODO: 處理 keys_target
    lemmatized_lower_documents_join = " ".join(lemmatized_lower_documents)
    all_words = lemmatized_lower_documents_join.split()

    # 組合回字符串形式
    keys_target =set( " ".join(all_words).split())

    #TODO: 創建 Querys 向量
    
    normalize_querys = [lemmatize_lower_sentence(query,stopwords) for query in querys.values()]

    result_querys = [creatQuery(normalize_query,keys_target) for normalize_query in normalize_querys]

    #TODO:  TF Weighting (Raw TF in course PPT) + Cosine Similarity
    tf_vectors = []  # 每篇文件的 TF 向量
    for doc in tqdm(lemmatized_lower_documents, desc="Processing TF-Vectors"): #如何加速
        tf_vector = [tfidf.tf(word, doc) for word in keys_target]
        tf_vectors.append(tf_vector)

    idf_vectors = []  # 每個字的 IDF 值
    idf_vectors = [tfidf.idf(word,lemmatized_lower_documents) for word in keys_target]
        
    tfidf_vectors = [] # 每篇文件的 TF-IDF 向量
    for tf_vector in  tqdm(tf_vectors, desc="Processing TFIDF-Vectors"):  #如何加速
        tfidf_vector = [tf * idf for tf, idf in zip(tf_vector, idf_vectors)]
        tfidf_vectors.append(tfidf_vector)

    tfidf_similarities_matrix = []
    
    for result_query in  tqdm(result_querys, desc="Processing TFIDF-Vectors similiarity"): #如何加速
        tfidf_similarities = []
        for tfidf_vector in tfidf_vectors:
            tfidf_similarity = cosine_similarity(result_query, tfidf_vector)
            tfidf_similarities.append(tfidf_similarity)
        tfidf_similarities_matrix.append(tfidf_similarities)


    predictions = {}
    for i ,query  in enumerate(querys.keys()):
        predictions[query.replace(".txt", "")] = return_top_10(documents,tfidf_similarities_matrix[i])
    print("TF-IDF Cosine")
    print("-------------------")
    print(f"MRR@10:    {calculate_mrr10(predictions, rel):<10.6f}")
    print(f"MAP@10:    {calculate_map(predictions, rel):<10.6f}")
    print(f"RECALL@10: {calculate_recall(predictions, rel):<10.6f}")
    print("-------------------")

if __name__ == "__main__":
    main()