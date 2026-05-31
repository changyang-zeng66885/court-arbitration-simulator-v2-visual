import os
from openai import OpenAI

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("Please set OPENAI_API_KEY before running this API demo.")

client = OpenAI(
    api_key=api_key,
    base_url=os.getenv("OPENAI_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
)

completion = client.chat.completions.create(
    # 模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models
    # qwen3.7-max, qwen3.6-plus, qwen3.6-flash
    model=os.getenv("OPENAI_MODEL", "qwen3.6-flash"),
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "你是谁？"},
    ],
    temperature=0.1
)
# print(completion.model_dump_json())
print(completion.choices[0].message.content)
