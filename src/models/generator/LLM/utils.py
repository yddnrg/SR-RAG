from openai import OpenAI
from openai import APIConnectionError
import time
import subprocess
from subprocess import DEVNULL
import logging


def run_vllm_server(model_id,model_name_ext="llm",probe_gap=5, skip=False):
    with open("./generate/LLM/start_vllm.sh", "r+") as file:
        lines = file.readlines()
        if len(lines) >= 3:
            # 找到等号的位置，并保留等号及之前的内容
            eq_index = lines[2].find('=')
            if eq_index != -1:
                lines[2] = lines[2][:eq_index + 1].rstrip() + model_id + "\n"
    
        if len(lines) >= 4:
            # 同样处理第四行
            eq_index = lines[3].find('=')
            if eq_index != -1:
                lines[3] = lines[3][:eq_index + 1].rstrip() + model_name_ext + "\n"
    
        file.seek(0)  # 将文件指针移到文件开头
        file.writelines(lines)  # 写回修改后的内容
        file.truncate()  # 如果文件内容变短，删除多余的部分

    if not skip:
        p = subprocess.Popen(
            ["sh", "./generate/LLM/start_vllm.sh"],
            stdout=DEVNULL,
            stderr=DEVNULL
        )

    client = OpenAI(
        base_url="http://localhost:4128/v1",
        api_key="123456",
    )

    # probe readiness
    i = 0
    probe_query = "Are you ready?!"
    while True:
        try:
            client.chat.completions.create(
                model="llm",
                messages=[
                    {"role": "user", "content": probe_query}
                ]
            )
            return client
        except APIConnectionError:
            i += 1
            print(f"Probe readiness: try {i}")
            time.sleep(probe_gap)



