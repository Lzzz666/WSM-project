import os 
import tfidf
import math
import string
import numpy as np
import jieba

from tqdm import tqdm 
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from textblob import TextBlob as tb
import argparse
from collections import defaultdict
ps = PorterStemmer()

def creatQuery(query,keys_string):      
    query_lower = tb(query).lower()
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

def preprocess(sentence,stopwords):
    # Tokenize sentence
    words = word_tokenize(sentence)
    words = [word for word in words if word not in string.punctuation]
    words = [word.lower() for word in words if word.lower() not in stopwords]
    # 進行詞性還原
    stem_docss = []
    for word in words:
        stem_docs = ps.stem(word)
        stem_docss.append(stem_docs)
    return ' '.join(stem_docss)

def print_top_10(documents,smilarity_results,reversed=True):
    if reversed:
        results = [(filename, sim) for (filename, sim) in zip(documents.keys(), smilarity_results)]
        top_10_results = sorted(results, key=lambda x: x[1], reverse=True)[:10]
        for filename, sim in top_10_results:
            print(f"{filename}: {sim:.4f}")
    else:
        results = [(filename, sim) for (filename, sim) in zip(documents.keys(), smilarity_results)]
        top_10_results = sorted(results, key=lambda x: x[1], reverse=False)[:10]
        for filename, sim in top_10_results:
            print(f"{filename}: {sim:.4f}")

def return_first(documents,smilarity_results,reversed=True):
    top_10_results = []
    if reversed:
        results = [(filename, sim) for (filename, sim) in zip(documents.keys(), smilarity_results)]
        top_10_results = sorted(results, key=lambda x: x[1], reverse=True)[:10]
    else:
        results = [(filename, sim) for (filename, sim) in zip(documents.keys(), smilarity_results)]
        top_10_results = sorted(results, key=lambda x: x[1], reverse=False)[:10]
    first_filename = top_10_results[0][0]
    return documents[first_filename]

def creatKeyTarget(stem_docs):
    stem_docs_join = " ".join(stem_docs)
    all_words = stem_docs_join.split()
    return set( " ".join(all_words).split())


def task3(args):
    documents = {}
    for filename in os.listdir('ChineseNews'):
        if filename.endswith('.txt'):
            with open("ChineseNews/"+filename, 'r', encoding="utf-8") as f:
                orginal_file = f.read()
            documents[filename]= " ".join(jieba.cut(orginal_file))

    stopwords = open('chinese.stop', 'r',encoding='utf-8').read().split()

    #TODO: (fixed) 處理文件 stemming and lower
    stem_docs = [preprocess(doc,stopwords) for doc in documents.values()]

    #TODO: 處理 keys_target

    keys_target = creatKeyTarget(stem_docs)

    #TODO: 創建 Query 向量
    
    chi_query = args.Chi_query
    # print("Query: ",chi_query)

    normalize_query = preprocess(chi_query,stopwords)
    # print("Query: ",normalize_query)
    result_query = creatQuery(normalize_query,keys_target)
    # print("Result_Query: ",result_query)

    #TODO:  TF Weighting (Raw TF in course PPT) + Cosine Similarity
    tf_vectors = []
    for doc in tqdm(stem_docs, desc="Processing TF-Vectors"): #如何加速
        # Calculate TF for the document only once
        tf_count = calculate_tf(doc)
        # Create a vector for the keys_target words
        tf_vector = [tf_count[word] for word in keys_target]  # Get count from tf_count dictionary
        tf_vectors.append(tf_vector)

    tf_similarities = []
    for tf_vector in tqdm(tf_vectors, desc="Processing TF-Similarities"): #如何加速
        tf_similarity = cosine_similarity(result_query, tf_vector)
        tf_similarities.append(tf_similarity)

    print("TF Cosine")
    print("NewsID   Score")
    print_top_10(documents,tf_similarities)

    #TODO:  TF-IDF Weighting + Cosine Similarity

    idf_vectors = []
    idf_vectors = [tfidf.idf(word,stem_docs) for word in keys_target]
    
    tfidf_vectors = []
    for tf_vector in  tqdm(tf_vectors, desc="Processing TFIDF-Vectors"):  #如何加速
        tfidf_vector = [tf * idf for tf, idf in zip(tf_vector, idf_vectors)]
        tfidf_vectors.append(tfidf_vector)
    # print(tfidf_vectors)

    tfidf_similarities = []
    for tfidf_vector in tqdm(tfidf_vectors, desc="Processing TFIDF-Similarities"): #如何加速
        tfidf_similarity = cosine_similarity(result_query, tfidf_vector)
        tfidf_similarities.append(tfidf_similarity)

    print("TF-IDF Cosine")
    print("NewsID   Score")
    print_top_10(documents,tfidf_similarities)

def calculate_tf(doc):
    words = doc.split()
    tf_count = defaultdict(int)
    for word in words:
        tf_count[word] += 1
    return tf_count

def main():
    #TODO: 將 EnglishNews 中的文件讀取出來
    documents = {}
    for filename in os.listdir('EnglishNews/EnglishNews'):
        if filename.endswith('.txt'):
            with open('EnglishNews/EnglishNews/'+filename, 'r', encoding="utf-8") as f:
                documents[filename] = f.read()

    stopwords = open('english.stop', 'r').read().split()

    #TODO: (fixed) 處理文件 stemming and lower (文件應該也要處理 stop words?)
    stem_docs = [preprocess(doc,stopwords) for doc in documents.values()]

    #TODO: 處理 keys_target
    keys_target = creatKeyTarget(stem_docs)

    #TODO: 創建 Query 向量
    parser = argparse.ArgumentParser(description="Process English query")
    parser.add_argument('--Eng_query', type=str, required=True, help='English query')
    parser.add_argument('--Chi_query', type=str, required=True, help='Chinese query')
    args = parser.parse_args()

    query = args.Eng_query

    normalize_query = preprocess(query,stopwords)

    result_query = creatQuery(normalize_query,keys_target)

    #TODO:  TF Weighting (Raw TF in course PPT) + Cosine Similarity
    tf_vectors = []
    for doc in tqdm(stem_docs, desc="Processing TF-Vectors"): #如何加速
        # Calculate TF for the document only once
        tf_count = calculate_tf(doc)
        # Create a vector for the keys_target words
        tf_vector = [tf_count[word] for word in keys_target]  # Get count from tf_count dictionary
        tf_vectors.append(tf_vector)
    print(tf_vectors)
    

    tf_similarities = []
    for tf_vector in tqdm(tf_vectors, desc="Processing TF-Similarities"): #如何加速
        tf_similarity = cosine_similarity(result_query, tf_vector)
        tf_similarities.append(tf_similarity)

    print("=== Task 1 ===")
    print("TF Cosine")
    print("NewsID   Score")
    print_top_10(documents,tf_similarities)

    #TODO:  TF-IDF Weighting + Cosine Similarity

    idf_vectors = []
    idf_vectors = [tfidf.idf(word,stem_docs) for word in keys_target]
    
    tfidf_vectors = []
    for tf_vector in  tqdm(tf_vectors, desc="Processing TFIDF-Vectors"):  #如何加速
        tfidf_vector = [tf * idf for tf, idf in zip(tf_vector, idf_vectors)]
        tfidf_vectors.append(tfidf_vector)

    tfidf_similarities = []
    for tfidf_vector in tqdm(tfidf_vectors, desc="Processing TFIDF-Similarities"): #如何加速
        tfidf_similarity = cosine_similarity(result_query, tfidf_vector)
        tfidf_similarities.append(tfidf_similarity)

    print("TF-IDF Cosine")
    print("NewsID   Score")
    print_top_10(documents,tfidf_similarities)

    #TODO: TF Weighting (Raw TF in course PPT) + Euclidean Distance
    tf_eulidean_distances = []
    for tf_vector in tqdm(tf_vectors, desc="Processing TF-Euclidean"): #如何加速
        tf_eulidean_distance = euclidean_distance(result_query, tf_vector)
        tf_eulidean_distances.append(tf_eulidean_distance)

    print("TF Euclidean")  
    print("NewsID   Score")
    print_top_10(documents,tf_eulidean_distances,False)


    #TODO: TF-IDF Weighting + Euclidean Distance
    tfidf_eulidean_distances = []
    for tfidf_vector in tqdm(tfidf_vectors, desc="Processing TFIDF-Euclidean"): #如何加速
        tfidf_eulidean_distance = euclidean_distance(result_query, tfidf_vector)
        tfidf_eulidean_distances.append(tfidf_eulidean_distance)

    print("TF-IDF Euclidean")
    print("NewsID   Score")
    print_top_10(documents,tfidf_eulidean_distances,False)

    first_docs = return_first(documents,tfidf_similarities)

    feedback_query = preprocess(first_docs,stopwords)

    feedback_query = creatQuery(feedback_query,keys_target)
    
    new_query = 1 * np.array(result_query) + 0.5 * np.array(feedback_query)

    re_ranked_tfidf_similarities = []
    for tfidf_vector in tqdm(tfidf_vectors, desc="Processing Re-Ranked TFIDF-Similarities"): #如何加速
        tfidf_similarity = cosine_similarity(new_query, tfidf_vector)
        re_ranked_tfidf_similarities.append(tfidf_similarity)
    print("=== Task 2 ===")
    print("TF-IDF Cosine re-ranked")
    print("NewsID   Score")
    print_top_10(documents,re_ranked_tfidf_similarities)

    # task 3
    task3(args)
    

if __name__ == "__main__":
    main()
