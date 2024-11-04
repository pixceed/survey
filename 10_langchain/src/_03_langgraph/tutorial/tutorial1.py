'''
パート 1: 基本的なチャットボットを構築する
'''

import os
from dotenv import load_dotenv  # .envファイルからAPIキーなどを読み込むためのライブラリ
from langchain_openai import ChatOpenAI  # OpenAIの言語モデルを使用するためのモジュール
from typing import Annotated  # 型ヒント用のモジュール
from typing_extensions import TypedDict  # 型ヒント用の拡張モジュール
from langgraph.graph import StateGraph, START, END  # グラフ構造を作成するためのモジュール
from langgraph.graph.message import add_messages  # メッセージ処理のためのモジュール

# .envファイルから環境変数（APIキーなど）を読み込み
load_dotenv()

# 状態の型定義。messagesにチャットのメッセージ履歴を保持する
class State(TypedDict):
    messages: Annotated[list, add_messages]

# TypedDictとは：Pythonの型ヒントの一種で辞書（dict）型に対して
# 「特定のキーとそのキーに対応する値の型」を定義するために使っている
# → ここでは、"messages"というキーにバリューとしてリストが対応する、という意味

# Annotatedは、型ヒントに追加情報（メタデータ）を付け加えるために使用される
# 各ノードが、stateの各要素をどのように更新するかの処理を定義（=reducers）する
# "add_messages"は、デフォルトで用意されているreducersで、単純にリストに追加する

# グラフビルダーを作成し、チャットボットのフローを定義
graph_builder = StateGraph(State)

# OpenAIのLLMインスタンスを作成。
chat_model = ChatOpenAI(
    model="gpt-4o",
    temperature=0  
)

# チャットボット関数。状態に応じてLLMが応答を生成
def chatbot(state: State):
    print(f"State: {state}")
    return {"messages": [chat_model.invoke(state["messages"])]}

# グラフにノードを追加し、チャットボット関数を実行するよう設定
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_edge(START, "chatbot")  # 開始ノードからチャットボットノードへエッジを追加
graph_builder.add_edge("chatbot", END)    # チャットボットから終了ノードへエッジを追加

# グラフをコンパイル（フローを確定）
graph = graph_builder.compile()

def stream_graph_updates(user_input: str):
    for event in graph.stream({"messages": [("user", user_input)]}):
        for value in event.values():
            print("Assistant:", value["messages"][-1].content)

# 無限ループを使用してユーザー入力を連続的に処理
while True:
    try:
        # ユーザーからの入力を取得
        user_input = input("User: ")
        
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!") 
            break 

        # ユーザーの入力を基にチャットボットが応答を生成し、リアルタイムで出力
        stream_graph_updates(user_input)

    except Exception as e:
        print(f"Error: {e}")
        break 
