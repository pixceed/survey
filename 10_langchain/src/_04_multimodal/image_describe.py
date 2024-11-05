'''
画像をLLMに説明させる
'''

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_community.callbacks.manager import get_openai_callback

import base64

# .envファイルから環境変数を読み込み
load_dotenv()

# OpenAIのLLMインスタンス作成
chat_model = ChatOpenAI(model="gpt-4o", temperature=0)

# 画像をbase64形式のデータに変換
image_path = "src/_04_multimodal/input/tokuchou1.jpg"
with open(image_path, "rb") as image_file:
    # base64エンコード
    encoded_string = base64.b64encode(image_file.read())
    # バイト列を文字列にデコードして返す
    image_data = encoded_string.decode('utf-8')


message = HumanMessage(
    content=[
        {"type": "text", "text": "この画像の状況を説明してください。"},
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{image_data}"},
        },
    ],
)

with get_openai_callback() as cb:
    response = chat_model.invoke([message])
    print(response.content)
    print(f"Total Tokens: {cb.total_tokens}")
    print(f"Prompt Tokens: {cb.prompt_tokens}")
    print(f"Completion Tokens: {cb.completion_tokens}")
    print(f"Total Cost (USD): ${cb.total_cost}")

