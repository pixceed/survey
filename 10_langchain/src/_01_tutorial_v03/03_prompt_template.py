import os
from dotenv import load_dotenv

from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI, AzureChatOpenAI


# .envファイルから環境変数を読み込み
load_dotenv()


print("\n---------------------------------------------------------------------\n")

# プロンプトテンプレートの準備
prompt = PromptTemplate(
    input_variables=["product"],
    template="{product}を作る日本語の新会社名をを1つ提案してください",
)

# プロンプトテンプレートの実行
prompt_value = prompt.invoke({"product": "家庭用ロボット"})
print(type(prompt_value))
print(prompt_value)

print("\n---------------------------------------------------------------------\n")

# PromptTemplate.from_templateのほうがラク
prompt = PromptTemplate.from_template(
"""
以下の料理のレシピを教えてください。

料理名: {dish}
"""
)

prompt_value = prompt.invoke({"dish": "カレー"})
print(type(prompt_value))
print(prompt_value)


print("\n---------------------------------------------------------------------\n")

# ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "ユーザーが入力した料理のレシピを教えてください。"),
        ("human", "{dish}"),
    ]
)

prompt_value = prompt.invoke({"dish": "カレー"})
print(type(prompt_value))
print(prompt_value)

print("\n---------------------------------------------------------------------\n")

# MessagesPlaceholder
# チャット形式のプロンプトのとき、会話履歴のように複数のメッセージが入るプレースホルダー

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "あなたは妖精です。語尾は「っぴ」「っぴね」としてください。"),
        MessagesPlaceholder("chat_history", optional=True),
        ("human", "{input}")
    ]
)

prompt_value = prompt.invoke(
    {
        "chat_history": [
            HumanMessage(content="我が名はMOIMOI。"),
            AIMessage("MOIMOIさん!よろしくっぴ!"),
        ],
        "input": "我が名を申せ!"
    }
)

print(prompt_value)
print()

# ChatModelの準備
chat_model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

ai_message = chat_model.invoke(prompt_value)
print(ai_message.content)

print("\n---------------------------------------------------------------------\n")

# 画像のPromptTemplate

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "user",
            [
                {"type": "text", "text": "画像を説明してください。"},
                {"type": "image_url", "image_url": {"url": "{image_url}"}}
            ]
        )
    ]
)

image_url = "https://raw.githubusercontent.com/yoshidashingo/langchain-book/main/assets/cover.jpg"

prompt_value = prompt.invoke({"image_url": image_url})

print(prompt_value)
print()

ai_message = chat_model.invoke(prompt_value)
print(ai_message.content)

print("\n---------------------------------------------------------------------\n")
