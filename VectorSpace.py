from pprint import pprint
from Parser import Parser
import util
import os


class VectorSpace:
    """ A algebraic model for representing text documents as vectors of identifiers. 
    A document is represented as a vector. Each dimension of the vector corresponds to a 
    separate term. If a term occurs in the document, then the value in the vector is non-zero.
    """

    #Collection of document term vectors
    documentVectors = []

    #Mapping of vector index to keyword
    vectorKeywordIndex=[]



    #Tidies terms
    parser=None


    def __init__(self, documents=[]):
        self.documentVectors=[]
        self.parser = Parser()
        if(len(documents)>0):
            self.build(documents)

    def build(self,documents):
        """ Create the vector space for the passed document strings """
        self.vectorKeywordIndex = self.getVectorKeywordIndex(documents)
        self.documentVectors = [self.makeVector(document) for document in documents]

        #將關鍵字與位置關聯起來
        # print(self.vectorKeywordIndex)
        #將文件轉換成向量
        # print(self.documentVectors)


    def getVectorKeywordIndex(self, documentList):
        """ create the keyword associated to the position of the elements within the document vectors """
        # print("documentList:",documentList)
        #Mapped documents into a single word string	
        #將所有的文件合併成一個字符串
        vocabularyString = " ".join(documentList)
        # print("vocabularyString:",vocabularyString)
        #將字符串分割成單詞
        vocabularyList = self.parser.tokenise(vocabularyString)
        # print("tokenise:",vocabularyList)

        #Remove common words which have no search value
        #移除停用詞 (stopwords)
        vocabularyList = self.parser.removeStopWords(vocabularyList)
        # print("removeStopWords:",vocabularyList)

        #移除重複的單詞
        uniqueVocabularyList = util.removeDuplicates(vocabularyList)
        # print("removeStopWords:",vocabularyList)

        vectorIndex={}
        offset=0
        #Associate a position with the keywords which maps to the dimension on the vector used to represent this word
        #將單詞與位置關聯起來
        for word in uniqueVocabularyList:
            vectorIndex[word]=offset
            offset+=1
        return vectorIndex  #(keyword:position)

    #將文件轉換成向量
    def makeVector(self, wordString):
        """ @pre: unique(vectorIndex) """

        #Initialise vector with 0's
        vector = [0] * len(self.vectorKeywordIndex)
        #將字符串分割成單詞
        wordList = self.parser.tokenise(wordString)
        #移除停用詞 (stopwords)
        wordList = self.parser.removeStopWords(wordList)
        for word in wordList:
            vector[self.vectorKeywordIndex[word]] += 1; #Use simple Term Count Model
        return vector


    def buildQueryVector(self, termList):
        """ convert query string into a term vector """
        query = self.makeVector(" ".join(termList))
        return query


    def related(self,documentId):
        """ find documents that are related to the document indexed by passed Id within the document Vectors"""
        ratings = [util.cosine(self.documentVectors[documentId], documentVector) for documentVector in self.djiocumentVectors]
        #ratings.sort(reverse=True)
        return ratings


    def search(self,searchList):
        """ search for documents that match based on a list of terms """
        queryVector = self.buildQueryVector(searchList)

        ratings = [util.cosine(queryVector, documentVector) for documentVector in self.documentVectors]
        #ratings.sort(reverse=True)
        return ratings



