import sys
sys.path.append(f'.')

from generate.LLM.utils import run_vllm_server

client=run_vllm_server("/mnt/data/renyuming/model-qwen2-7b-instruct-awq",skip=True)
llm_res = client.chat.completions.create(
    model="llm",
    messages=[{"role": "user", "content": "你好"}],
    temperature=0.3,
)
llm_output = llm_res.choices[0].message.content
print(llm_output)