import sys
sys.path.append(f'.')
import faiss
from FlagEmbedding import BGEM3FlagModel
import numpy as np
from data_process.util import get_document_by_id

class HNSW:
    def __init__(self, save_path="", file_path="", M=32, efSearch=32, efConstruction=32):
        self.embedding_model=BGEM3FlagModel('/mnt/data/renyuming/BAAI/bge-large-zh-v1.5', use_fp16=True)
        self.filepath=file_path
        if save_path!="":
            self.vecdb=faiss.read_index(save_path)
        else:
            d=self.embedding_model.encode("test")['dense_vecs'].shape[0]
            self.vecdb = faiss.IndexHNSWFlat(d, M)
            self.vecdb.hnsw.efConstruction = efConstruction
            self.vecdb.hnsw.efSearch = efSearch
            with open(file_path, 'r', encoding='utf-8') as file:
                lines = file.readlines()
            self.build_vecdb(lines)

    def tokenize(self, text:str):
        token = self.embedding_model.encode(text)['dense_vecs']
        return token 

    def get_topK(self, query:str, k=5):
        query_token=self.tokenize(query).reshape(1,1024)
        score_list,index_list=self.vecdb.search(query_token,k)
        score_list=score_list.reshape(-1)
        index_list=index_list.reshape(-1)
        ans_list=get_document_by_id(self.filepath,index_list)

        return ans_list
    
    def add_documents(self, documents:list):
        tokens=np.array([])
        for document in documents:
            if tokens.size==0:
                tokens=np.append(tokens,self.tokenize(document))
            else:
                tokens=np.vstack([tokens,self.tokenize(document)])
        tokens=tokens.reshape(-1,1024)
        self.vecdb.add(tokens) 
    
    def build_vecdb(self, facts, batch_size=1000):
        facts_len=len(facts)
        n=facts_len//batch_size+1
        print(n)
        for i in range(n):
            print(i)
            if (i + 1) * batch_size < facts_len:
                facts_batch = facts[i * batch_size:(i + 1) * batch_size]
            else:
                facts_batch = facts[i * batch_size:]
            self.add_documents(facts_batch)
        faiss.write_index(self.vecdb, 'cail_facts.index')
            
            


    
if __name__ == "__main__":
    hnsw = HNSW(file_path='cail_facts.txt')
