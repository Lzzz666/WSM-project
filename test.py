import os 
import VectorSpace
import tfidf
import math
import string
import numpy as np
import jieba

from tqdm import tqdm 
from Parser import Parser

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob as tb


lemmatizer = WordNetLemmatizer()

def normalize_doc(doc):
    parser = Parser()
    vocabularyString = " ".join(doc)
    #將字符串分割成單詞
    vocabularyList = parser.tokenise(vocabularyString)
    vocabularyList = parser.removeStopWords(vocabularyList)
    return vocabularyList

def creatQuery(query,keys_string):      
    query_blob = tb(query)
    query_lower = query_blob.lower()
    query_vector = [tfidf.tf(word, query_lower) for word in keys_string]
    return query_vector

def normalize_words(words):
    lemmatized_words = []
    for word in words:
        # 使用 'v' 來指示單詞是動詞
        lemmatized_word = lemmatizer.lemmatize(word, pos='v')
        lemmatized_word = lemmatizer.lemmatize(word, pos='a')
        lemmatized_words.append(lemmatized_word)
    return lemmatized_words

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

def main():
    #TODO: 將 ChineseNews 中的文件讀取出來
    # jieba.add_word('資安')
    documents = {}
    for filename in os.listdir('ChineseNews'):
        if filename.endswith('.txt'):
            with open("ChineseNews/"+filename, 'r', encoding="utf-8") as f:
                orginal_file = f.read()
            # print("Orginal_File: ",orginal_file)
            documents[filename]= " ".join(jieba.cut(orginal_file))

    print("Orginal_File:",documents)
    
    stopwords = open('chinese.stop', 'r',encoding='utf-8').read().split()

    #TODO: (fixed) 處理文件 stemming and lower
    lemmatized_lower_documents = [lemmatize_lower_sentence(doc,stopwords) for doc in documents.values()]
    # print("Lemmatized_Lower_Documents: ",lemmatized_lower_documents)
    print("Lemmatized_Lower_Documents: ",lemmatized_lower_documents)
    #TODO: 處理 keys_target
    lemmatized_lower_documents_join = " ".join(lemmatized_lower_documents)
    all_words = lemmatized_lower_documents_join.split()

    # 組合回字符串形式
    keys_target =set( " ".join(all_words).split())
    print("Keys_Target: ", keys_target)

    #TODO: 創建 Query 向量
    query = "資安 遊戲"
    normalize_query = lemmatize_lower_sentence(query,stopwords)
    # print("Query: ",normalize_query)
    result_query = creatQuery(normalize_query,keys_target)
    print("Result_Query: ",result_query)

    # print(result_query)
    # print("lemmatized_lower_doc: ",lemmatized_lower_documents)
    #TODO:  TF Weighting (Raw TF in course PPT) + Cosine Similarity
    tf_vectors = []
    for doc in tqdm(lemmatized_lower_documents, desc="Processing TF-Vectors"): #如何加速
        tf_vector = [tfidf.tf(word, doc) for word in keys_target]
        tf_vectors.append(tf_vector)
    # for vector in tf_vectors:
    #     print("tf_vector: ",vector)

    tf_similarities = []
    for tf_vector in tqdm(tf_vectors, desc="Processing TF-Similarities"): #如何加速
        tf_similarity = cosine_similarity(result_query, tf_vector)
        tf_similarities.append(tf_similarity)

    print("TF Cosine")
    print("NewsID   Score")
    print_top_10(documents,tf_similarities)

    #TODO:  TF-IDF Weighting + Cosine Similarity
    # (fixed)IDF  math.log 其實是 ln() => 所以數字應該沒問題
    idf_vectors = []
    idf_vectors = [tfidf.idf(word,lemmatized_lower_documents) for word in keys_target]

    # print("IDF_Vector: ",idf_vectors)
    
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


if __name__ == "__main__":
    main()