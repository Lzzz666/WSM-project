import os 
import VectorSpace
import tfidf
import math
import string
import numpy as np
import argparse
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

def print_top_10_and_return_first(documents,smilarity_results,reversed=True):
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
    first_filename = top_10_results[0][0]
    return documents[first_filename]

def main():
    #TODO: 將 EnglishNews 中的文件讀取出來
    documents = {}
    for filename in os.listdir('EnglishNews/EnglishNews'):
        if filename.endswith('.txt'):
            with open('EnglishNews/EnglishNews/'+filename, 'r', encoding="utf-8") as f:
                documents[filename] = f.read()

    stopwords = open('english.stop', 'r').read().split()

    #TODO: (fixed) 處理文件 stemming and lower (文件應該也要處理 stop words?)
    lemmatized_lower_documents = [lemmatize_lower_sentence(doc,stopwords) for doc in documents.values()]

    #TODO: 處理 keys_target
    lemmatized_lower_documents_join = " ".join(lemmatized_lower_documents)
    all_words = lemmatized_lower_documents_join.split()

    # 組合回字符串形式
    keys_target =set( " ".join(all_words).split())
    print("Keys_Target: ", keys_target)

    #TODO: 創建 Query 向量
    parser = argparse.ArgumentParser(description="Process English query")
    parser.add_argument('--Eng_query', type=str, required=True, help='English query')

    args = parser.parse_args()
    query = args.Eng_query
    print("Query: ",query)
    normalize_query = lemmatize_lower_sentence(query,stopwords)
 
    result_query = creatQuery(normalize_query,keys_target)

    #TODO:  TF Vector
    tf_vectors = []
    for doc in tqdm(lemmatized_lower_documents, desc="Processing TF-Vectors"): #如何加速
        tf_vector = [tfidf.tf(word, doc) for word in keys_target]
        tf_vectors.append(tf_vector)

    #TODO:  TF-IDF Weighting + Cosine Similarity
    # (fixed)IDF  math.log 其實是 ln() => 所以數字應該沒問題
    idf_vectors = []
    idf_vectors = [tfidf.idf(word,lemmatized_lower_documents) for word in keys_target]
    
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
    
    first_docs = print_top_10_and_return_first(documents,tfidf_similarities)

    feedback_query = lemmatize_lower_sentence(first_docs,stopwords)

    feedback_query = creatQuery(feedback_query,keys_target)
    
    new_query = 1 * np.array(result_query) + 0.5 * np.array(feedback_query)

    re_ranked_tfidf_similarities = []
    for tfidf_vector in tqdm(tfidf_vectors, desc="Processing Re-Ranked TFIDF-Similarities"): #如何加速
        tfidf_similarity = cosine_similarity(new_query, tfidf_vector)
        re_ranked_tfidf_similarities.append(tfidf_similarity)
    print("TF-IDF Cosine re-ranked")
    print("NewsID   Score")
    print_top_10_and_return_first(documents,re_ranked_tfidf_similarities)


if __name__ == "__main__":
    main()