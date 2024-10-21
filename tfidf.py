from __future__ import division, unicode_literals
import math

from textblob import TextBlob as tb

import nltk

nltk.download('punkt')
nltk.download('stopwords')

def tf(word, blob):
    words = blob.split()
    return words.count(word)

def n_containing(word, bloblist):
    return sum(1 for blob in bloblist if word in blob)

def idf(word, bloblist):
    if(n_containing(word, bloblist) == 0):
        print("word not in bloblist:",word)
        return 0
    return round(math.log(len(bloblist) / (n_containing(word, bloblist))),5)

def tfidf(word, blob, bloblist):
    return tf(word, blob) * idf(word, bloblist)

