import os
import base64

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.callbacks.manager import get_openai_callback


def figure2text(figure_file_path, chat_model):

    # ==============================
    # 図・表の画像のテキスト化
    # ==============================

    # 画像をbase64形式のデータに変換
    with open(figure_file_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
        image_data = encoded_string.decode('utf-8')

    # システムプロンプト
    system_prompt0 = SystemMessage(
        content=\
"""
あなたは天才的な画像解析者です。
とある企業に関するドキュメントの画像が与えられます。

[指示]
・与えられた画像を、箇条書きでシンプルにまとめてください。
・余計な文言は出力しないでください。
"""
    )
    # 画像プロンプト
    image_message = HumanMessage(
        content=[
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image_data}"},
            },
        ],
    )

    messages = [system_prompt0, image_message]
    result = chat_model.invoke(messages)

    result_text = result.content
    figure_text = result_text.replace("```plaintext", "").replace("```", "")

    return figure_text