
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

    # print(result_query)
    # print("lemmatized_lower_doc: ",stem_docs)
    #TODO:  TF Weighting (Raw TF in course PPT) + Cosine Similarity
    tf_vectors = []
    for doc in tqdm(stem_docs, desc="Processing TF-Vectors"): #如何加速
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
    idf_vectors = [tfidf.idf(word,stem_docs) for word in keys_target]

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