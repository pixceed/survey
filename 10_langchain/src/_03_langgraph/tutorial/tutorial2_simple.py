'''
パート2:ツールによるチャットボットの強化
※事前に構築済みのモジュールを活用した場合
'''
from dotenv import load_dotenv  
from langchain_openai import ChatOpenAI  
from typing import Annotated
from typing_extensions import TypedDict 
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages

from langgraph.prebuilt import ToolNode, tools_condition
from src._03_langgraph.tools.python_exec import PythonExec


# .envファイルから環境変数（APIキーなど）を読み込み
load_dotenv()


# ＜ステートの構築＞
class State(TypedDict):
    messages: Annotated[list, add_messages]

# グラフビルダーを作成し、チャットボットのフローを定義
graph_builder = StateGraph(State)


# ＜ノードの構築＞
# ツールを準備
tool = PythonExec()
tools = [tool]

# LLMインスタンスを準備
llm = ChatOpenAI(model="gpt-4o", temperature=0)
llm_with_tools = llm.bind_tools(tools)

# チャットボット関数。状態に応じてLLMが応答を生成
def chatbot(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

# グラフにノードを追加し、チャットボット関数を実行するよう設定
graph_builder.add_node("chatbot", chatbot)

tool_node = ToolNode(tools=tools)
graph_builder.add_node("tools", tool_node)



# ＜エッジの構築＞
graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition,
)

# ほかのエッジを追加
graph_builder.add_edge("tools", "chatbot")
graph_builder.set_entry_point("chatbot")

# グラフをコンパイル
graph = graph_builder.compile()


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
        
    except Exception as e:
        print(f"Error: {e}")
        break 