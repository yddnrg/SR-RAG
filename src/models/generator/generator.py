import sys
sys.path.append(f'.')
import json, re, os
from openai import OpenAI
from generator.LLM.utils import run_vllm_server
from generator.prompt.prompt_for_generate import generate_prompt
from retriever.retriver import Retriever

class Generator():
    def __init__(self):

        self.prompt_zh = generate_prompt
        self.client = run_vllm_server("/mnt/data/renyuming/model-qwen2-7b-instruct-awq")
        self.modelId="llm"
        self.retriever=Retriever("documents.txt")

    def answer(self, question, use_retrieval=False):
        
        documents=[]
        
        if use_retrieval:
            documents=self.retriever.search(question)

        full_prompt = self.prompt_zh.format(
            DOCUMENTS=documents,
            QUESTION=question,
        )
        print(full_prompt)
        llm_res = self.client.chat.completions.create(
            model=self.modelId,
            messages=[{"role": "user", "content": full_prompt}],
            temperature=0.3,
        )
        llm_output = llm_res.choices[0].message.content
        return llm_output

