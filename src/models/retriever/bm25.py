import sys
sys.path.append(f'.')
from gensim.models.bm25model import OkapiBM25Model
import jieba, json
from gensim.corpora import Dictionary
from data_process.util import remove_stopwords, get_document_by_id
import functools

def before_method_decorator(func):
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        if self.documents_buffer != []:
            self.dictionary.add_documents([self.cut(document) for document in self.documents_buffer])
            with open(self.filepath, 'a', encoding='utf-8') as file:
                for document in self.documents_buffer:
                    file.write(document+"\n")
            self.documents_buffer = []
        return func(self, *args, **kwargs)
    return wrapper

class BM25:
    def __init__(self, file_path):
        self.filepath=file_path
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
        self.dictionary=Dictionary([])
        for line in lines:
            self.dictionary.add_documents([self.cut(line)])
        self.model = OkapiBM25Model(dictionary=self.dictionary)
        self.documents_buffer=[]

    def cut(self, text:str):
        return remove_stopwords(jieba.lcut(text))

    @before_method_decorator
    def tokenize(self, text:str):
        splited_text = remove_stopwords(jieba.lcut(text))
        token = self.dictionary.doc2bow(splited_text)
        return token

    @before_method_decorator
    def get_score(self, query:str, document:str):
        query_token = self.tokenize(query)
        document_token = self.tokenize(document)
        query_words_ids = {x[0] for x in query_token}
        # 从document_token中筛选出第一个元素存在于query_words_ids中的元组
        term_frequencies = [x for x in document_token if x[0] in query_words_ids]
        score = 0
        for item in self.model[term_frequencies]:
            score += item[1]
        return score

    @before_method_decorator
    def get_topK(self, query:str, k=5):
        score_list = []
        with open(self.filepath, 'r', encoding='utf-8') as file:
            lines = file.readlines()
        for index, document in enumerate(lines):
            score = self.get_score(query, document)
            score_list.append((index, score))
        
        sorted_score = sorted(score_list, key=lambda x: x[1], reverse=True)
        top_k=[sorted_score[i][0] for i in range(min(k, len(sorted_score)))]
        result_documents=get_document_by_id(self.filepath,top_k)

        return result_documents
    
    def add_documents(self, documents:list):
        self.documents_buffer.extend(documents)


if __name__ == "__main__":
    bm25 = BM25('documents.txt')
    print(bm25.get_topK("环境法中关于水污染的规定"))
