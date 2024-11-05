'''
画像をマークダウンに変換
'''

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_community.callbacks.manager import get_openai_callback

import base64

# .envファイルから環境変数を読み込み
load_dotenv()

image_path = "src/_04_multimodal/input/2021r03h_nw_pm1_qs-2.png"
output_path = "src/_04_multimodal/output/2021r03h_nw_pm1_qs-2.md"

# OpenAIのLLMインスタンス作成
chat_model = ChatOpenAI(model="gpt-4o", temperature=0)

# 画像をbase64形式のデータに変換

with open(image_path, "rb") as image_file:
    # base64エンコード
    encoded_string = base64.b64encode(image_file.read())
    # バイト列を文字列にデコードして返す
    image_data = encoded_string.decode('utf-8')


system_prompt = SystemMessage(
    content=\
"""
あなたは天才的な文書作成者です。
画像から文章を読み取り、マークダウン形式にまとめることができます。
画像中における図や表の部分は、`![Local Image](image.png)`としてください。
なお、図や表の番号およびキャプションは、文章内に記載してください。
出力は、必ずマークダウン文章のみで、余計な文章は含めないでください。
"""
)

image_message = HumanMessage(
    content=[
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{image_data}"},
        },
    ],
)

messages = [system_prompt, image_message]

with get_openai_callback() as cb:
    result = chat_model.invoke(messages)
    print(result)

    md_text = result.content
    with open(output_path, mode="w", encoding="utf-8") as f:
        f.write(md_text)
    print(f"Total Tokens: {cb.total_tokens}")
    print(f"Prompt Tokens: {cb.prompt_tokens}")
    print(f"Completion Tokens: {cb.completion_tokens}")
    print(f"Total Cost (USD): ${cb.total_cost}")

