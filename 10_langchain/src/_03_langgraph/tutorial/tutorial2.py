'''
パート2:ツールによるチャットボットの強化
'''
import json
from dotenv import load_dotenv  # .envファイルからAPIキーなどを読み込むためのライブラリ
from langchain_openai import ChatOpenAI  # OpenAIの言語モデルを使用するためのモジュール
from typing import Annotated, Literal  # 型ヒント用のモジュール
from typing_extensions import TypedDict  # 型ヒント用の拡張モジュール
from langgraph.graph import StateGraph, START, END  # グラフ構造を作成するためのモジュール
from langgraph.graph.message import add_messages  # メッセージ処理のためのモジュール

from langchain.agents import Tool #
from langchain_experimental.utilities import PythonREPL  # Pythonのコードを実行するためのモジュール
from langchain_core.messages import ToolMessage

# .envファイルから環境変数（APIキーなど）を読み込み
load_dotenv()

# ＜ステートの構築＞
# 状態の型定義。messagesにチャットのメッセージ履歴を保持する
class State(TypedDict):
    messages: Annotated[list, add_messages]

# グラフビルダーを作成し、チャットボットのフローを定義
graph_builder = StateGraph(State)


# ＜ノードの構築＞



# Pythonコードを実行するREPL環境を作成
python_repl = PythonREPL()

# 実行テスト
# code = "x = 2\nx += 2\nprint(x)"
# result = python_repl.run(code)
# print(result)

# Pythonコードを実行して結果を返す関数
def execute_python_code(code):
    try:
        # 最終評価を出力するようにする
        last_line = code.split('\n')[-1]
        code += f"\nprint({last_line})"

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


# ツールを実行するノード
class BasicToolNode:

    def __init__(self, tools: list) -> None:
        # ツールを登録している
        # 辞書型 {ツール名: ツール}
        self.tools_by_name = {tool.name: tool for tool in tools}

    def __call__(self, inputs: dict):
        # 最後のメッセージを取得する
        if messages := inputs.get("messages", []):
            message = messages[-1]
        else:
            raise ValueError("No message found in input")
        
        # メッセージからツールが呼び出されているところを参照し
        # ツールを実行している
        outputs = []
        for tool_call in message.tool_calls:
            tool_result = self.tools_by_name[tool_call["name"]].invoke(
                tool_call["args"]
            )
            outputs.append(
                ToolMessage(
                    content=json.dumps(tool_result),
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"],
                )
            )
        return {"messages": outputs}

# グラフにノードを追加
tool_node = BasicToolNode(tools=tools)
graph_builder.add_node("tools", tool_node)

# OpenAIのLLMインスタンスを作成。
llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0  
)
llm_with_tools = llm.bind_tools(tools)

# チャットボット関数。状態に応じてLLMが応答を生成
def chatbot(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

# グラフにノードを追加し、チャットボット関数を実行するよう設定
graph_builder.add_node("chatbot", chatbot)

# ＜エッジの構築＞

# chatbotの応答から、ツールを使うか終了するかを振り分ける関数
def route_tools(state: State,):

    # chatbotの最後のメッセージを取得する
    if isinstance(state, list):
        ai_message = state[-1]
    elif messages := state.get("messages", []):
        ai_message = messages[-1]
    else:
        raise ValueError(f"No messages found in input state to tool_edge: {state}")
    
    # chatbotの最後のメッセージから、ツールの実行を行うかを判断する
    # hasattr(ai_message, "tool_calls") これはai_messageに"tool_calls"の属性があるかを判定
    if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
        return "tools"
    return END

# エッジを追加
graph_builder.add_conditional_edges(
    "chatbot",
    route_tools,
    {"tools": "tools", END: END},
)

# ほかのエッジを追加
graph_builder.add_edge("tools", "chatbot")
graph_builder.set_entry_point("chatbot")

# グラフをコンパイル
graph = graph_builder.compile()

# # グラフの画像を保存
# with open("output_graph.png", mode="wb") as f:
#     f.write(graph.get_graph().draw_mermaid_png())

def stream_graph_updates(user_input: str):
    for event in graph.stream({"messages": [("user", user_input)]}):
        for value in event.values():
            print("Assistant:", value["messages"][-1].content)


while True:
    try:
        user_input = input("User: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break

        stream_graph_updates(user_input)

    except:
        # fallback if input() is not available
        user_input = "What do you know about LangGraph?"
        print("User: " + user_input)
        stream_graph_updates(user_input)
        break