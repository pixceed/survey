'''
langgraphを用いたPython実行機能付きエージェントの作成
'''

import os
from dotenv import load_dotenv  # .envファイルからAPIキーなどを読み込むためのライブラリ
from langchain_openai import ChatOpenAI  # OpenAIの言語モデルを使用するためのモジュール
from typing import Annotated  # 型ヒント用のモジュール
from typing_extensions import TypedDict  # 型ヒント用の拡張モジュール
from langgraph.graph import StateGraph, START, END  # グラフ構造を作成するためのモジュール
from langgraph.graph.message import add_messages  # メッセージ処理のためのモジュール
from langgraph.prebuilt import chat_agent_executor, ToolNode, tools_condition  # ツールを使うためのノード定義

from langchain.agents import Tool  # Langchainでツールを扱うためのモジュール
from langchain_experimental.utilities import PythonREPL  # Pythonのコードを実行するためのモジュール

# .envファイルから環境変数（APIキーなど）を読み込み
load_dotenv()

# 状態の型定義。messagesにチャットのメッセージ履歴を保持する
class State(TypedDict):
    messages: Annotated[list, add_messages]  # メッセージのリストを保持するための型定義

# グラフビルダーを作成し、チャットボットのフローを定義
graph_builder = StateGraph(State)

# OpenAIのLLMインスタンスを作成
chat_model = ChatOpenAI(
    model="gpt-4o",  # GPT-4モデルを使用
    temperature=0   # 一貫した応答のために温度を0に設定 
)

# Pythonコードを実行するREPL環境を作成
python_repl = PythonREPL()

# Pythonコードを実行して結果を返す関数
def execute_python_code(code):
    try:
        # Pythonコードに必ずprint文を追加する
        code = f"print({code})" if "print" not in code else code
        
        # Pythonコードを実行
        result = python_repl.run(code)
        # 実行結果が空の場合
        if not result:
            result = "結果が返されませんでした。"
        # 実行されたコードと結果を表示
        print("pythonが実行されました。")
        print(f"コード:\n{code}\n\n結果:\n{result}\n\n")
    except Exception as e:
        # エラーハンドリング
        result = f"エラーが発生しました: {str(e)}"
    return result

# Pythonコードを実行できるツールを作成
python_tool = Tool(
    name="Python_REPL",
    func=execute_python_code,  # 実行する関数
    description="Pythonコードを実行して結果を表示するツール"
)

# 作成したツールをリストに追加
tools = [python_tool]


# チャットエージェントの作成（システムプロンプトを追加）
agent_executor = chat_agent_executor.create_tool_calling_executor(
    chat_model,
    tools=tools
)


# LLMとツールを結びつける
llm_with_tools = chat_model.bind_tools(tools)

# グラフにチャットエージェントのノードを追加
graph_builder.add_node("agent", agent_executor)

# Pythonツールを扱うノードを追加
tool_node = ToolNode(tools=tools)
graph_builder.add_node("tools", tool_node)

# 条件に応じてPythonツールを使用するよう設定
graph_builder.add_conditional_edges(
    "agent",  # チャットエージェントノードから
    tools_condition  # ツールを呼び出す条件
)

# ツールの実行が完了したら再びチャットエージェントへ
graph_builder.add_edge("tools", "agent")

# チャットエージェントノードをエントリーポイントとして設定
graph_builder.set_entry_point("agent")

# グラフをコンパイル（フローを確定）
graph = graph_builder.compile()

# 会話記憶を保持するため、初期状態でメッセージリストを保持
state = {"messages": []}  # メッセージ履歴を保持するリスト

# システムプロンプトとして最初に追加されるメッセージ
system_message = {
    "role": "system",
    "content": "Pythonプログラムを実行する場合、結果をprintで出力してください。"
}
state["messages"].append(system_message)

# ユーザーの入力に基づいてチャットボットが応答を生成し、その過程をリアルタイムでストリームする関数
def stream_graph_updates(user_input: str):
    # ユーザーの入力をメッセージに追加
    state["messages"].append(("user", user_input))  
    
    # グラフのstreamメソッドを使用して、メッセージに応じたイベントを処理
    for event in graph.stream(state):
        for value in event.values():
            # チャットボットの応答をメッセージに追加
            response = value["messages"][-1].content
            state["messages"].append(("assistant", response))  # 応答もメッセージリストに追加
            print("Assistant:", response)

# 無限ループを使用してユーザー入力を連続的に処理
while True:
    try:
        # ユーザーからの入力を取得
        user_input = input("User: ")
        
        # "quit", "exit", "q"の入力でループを終了
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")  # 終了メッセージを表示
            break  # ループを抜ける

        # ユーザーの入力を基にチャットボットが応答を生成し、リアルタイムで出力
        stream_graph_updates(user_input)

    except Exception as e:

        print(f"エラーが出ました\n{e}")  # 既定のユーザー入力を表示
        stream_graph_updates(user_input)  # その入力に対してチャットボットが応答を生成
        break  # エラーハンドリング後にループを終了
