'''
langgraphを用いたシンプルチャットボットの作成
'''

import os
from dotenv import load_dotenv  # .envファイルからAPIキーなどを読み込むためのライブラリ
from langchain_openai import ChatOpenAI  # OpenAIの言語モデルを使用するためのモジュール
from typing import Annotated  # 型ヒント用のモジュール
from typing_extensions import TypedDict  # 型ヒント用の拡張モジュール
from langgraph.graph import StateGraph, START, END  # グラフ構造を作成するためのモジュール
from langgraph.graph.message import add_messages  # メッセージ処理のためのモジュール

from IPython.display import Image, display  # 画像表示のためのモジュール

# .envファイルから環境変数（APIキーなど）を読み込み
load_dotenv()

# 状態の型定義。messagesにチャットのメッセージ履歴を保持する
class State(TypedDict):
    messages: Annotated[list, add_messages]

# グラフビルダーを作成し、チャットボットのフローを定義
graph_builder = StateGraph(State)

# OpenAIのLLMインスタンスを作成。
chat_model = ChatOpenAI(
    model="gpt-4o",
    temperature=0  
)

# チャットボット関数。状態に応じてLLMが応答を生成
def chatbot(state: State):
    return {"messages": [chat_model.invoke(state["messages"])]}

# グラフにノードを追加し、チャットボット関数を実行するよう設定
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_edge(START, "chatbot")  # 開始ノードからチャットボットノードへエッジを追加
graph_builder.add_edge("chatbot", END)    # チャットボットから終了ノードへエッジを追加

# グラフをコンパイル（フローを確定）
graph = graph_builder.compile()


# # 画像ファイルに保存するための関数を追加
# def save_image(graph, filename="graph.png"):
#     try:
#         img_data = graph.get_graph().draw_mermaid_png()  # グラフを画像データに変換
#         with open(filename, "wb") as f:
#             f.write(img_data)  # 画像データをファイルに保存
#         print(f"画像が {filename} として保存されました。")
#     except Exception as e:
#         print(f"エラーが発生しました: {e}")

# # グラフを表示し、画像を保存
# try:
#     image_data = graph.get_graph().draw_mermaid_png()  # グラフを画像データに変換
#     save_image(graph)  # グラフ画像をファイルに保存
# except Exception as e:
#     print(f"画像の表示または保存に失敗しました: {e}")



def stream_graph_updates(user_input: str):
    for event in graph.stream({"messages": [("user", user_input)]}):
        for value in event.values():
            print("Assistant:", value["messages"][-1].content)


# # 会話記憶を保持するため、初期状態でメッセージリストを保持
# state = {"messages": []}  # メッセージ履歴を保持するリスト

# # ユーザーの入力に基づいてチャットボットが応答を生成し、その過程をリアルタイムでストリームする関数
# def stream_graph_updates(user_input: str):
#     # ユーザーの入力をメッセージに追加
#     state["messages"].append(("user", user_input))  
    
#     # グラフのstreamメソッドを使用して、メッセージに応じたイベントを処理
#     for event in graph.stream(state):
#         for value in event.values():
#             # チャットボットの応答をメッセージに追加
#             response = value["messages"][-1].content
#             state["messages"].append(("assistant", response))  # 応答もメッセージリストに追加
#             print("Assistant:", response)
            

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

    except:
        # input()が使用できない環境でのフォールバック処理
        # input()のエラー発生時に自動的に既定の質問を使用して処理を実行
        user_input = "What do you know about LangGraph?"
        print("User: " + user_input)  
        stream_graph_updates(user_input)
        break 
