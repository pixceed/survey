'''
パート 6: 状態のカスタマイズ
・人間をツールとして考える
'''
from dotenv import load_dotenv  
from langchain_openai import ChatOpenAI  
from typing import Annotated
from typing_extensions import TypedDict 
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

from langgraph.prebuilt import ToolNode, tools_condition
from src._03_langgraph.tools.python_exec import PythonExec

from langgraph.checkpoint.memory import MemorySaver

from langchain_core.messages import AIMessage, ToolMessage

from pydantic import BaseModel



# .envファイルから環境変数（APIキーなど）を読み込み
load_dotenv()

# 状態の保存を行う（インメモリ）
memory = MemorySaver()

# ＜ステートの構築＞
class State(TypedDict):
    messages: Annotated[list, add_messages]
    ask_human: bool

# グラフビルダーを作成し、チャットボットのフローを定義
graph_builder = StateGraph(State)


# ＜ノードの構築＞

# 見せ算の計算をアシストする
class MisezanAssistance(BaseModel):
    request: str


# ツールを準備
tool = PythonExec()
tools = [tool]

# LLMインスタンスを準備
llm = ChatOpenAI(model="gpt-4o", temperature=0)

llm_with_tools = llm.bind_tools(tools + [MisezanAssistance])

# チャットボット関数。状態に応じてLLMが応答を生成
def chatbot(state: State):
    response = llm_with_tools.invoke(state["messages"])
    ask_human = False
    if (
        response.tool_calls
        and response.tool_calls[0]["name"] == MisezanAssistance.__name__
    ):
        ask_human = True
    return {"messages": [response], "ask_human": ask_human}

# グラフにノードを追加し、チャットボット関数を実行するよう設定
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("tools", ToolNode(tools=tools))

# エージェントからの応答を作成する関数
def create_response(response: str, ai_message: AIMessage):
    return ToolMessage(
        content=response,
        tool_call_id=ai_message.tool_calls[0]["id"],
    )

# 人間ノードを作成・追加
def human_node(state: State):
    new_messages = []
    if not isinstance(state["messages"][-1], ToolMessage):

        new_messages.append(
            create_response("No response from human.", state["messages"][-1])
        )
    return {
        "messages": new_messages,
        "ask_human": False,
    }
graph_builder.add_node("human", human_node)


# ＜エッジの構築＞
def select_next_node(state: State):
    if state["ask_human"]:
        return "human"
    return tools_condition(state)


graph_builder.add_conditional_edges(
    "chatbot",
    select_next_node,
    {"human": "human", "tools": "tools", END: END},
)

# ほかのエッジを追加
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge("human", "chatbot")
graph_builder.add_edge(START, "chatbot")

# グラフをコンパイル
# メモリーを指定
graph = graph_builder.compile(
    checkpointer=memory,
    interrupt_before=["human"],
    )

# 会話のスレッドキーを設定
config = {"configurable": {"thread_id": "1"}}


def stream_graph_updates(user_input: str):

    # ユーザーの質問をエージェントに投げかける
    events = list(graph.stream(
        {"messages": [("user", user_input)]}, 
        config, 
        stream_mode="values"
    ))

    # グラフの状態を調べる
    # snapshot.nextで、次にツールを実行するかどうかわかる
    snapshot = graph.get_state(config)

    if "human" in snapshot.next:
    
        # 人のチェック
        misezan_answer = input("見せ算の回答:")

        ai_message = snapshot.values["messages"][-1]
        tool_message = create_response(misezan_answer, ai_message)

        # 状態を更新
        graph.update_state(config, {"messages": [tool_message]})

        # 何も入れずにストリーム実行
        events = list(graph.stream(None, config, stream_mode="values"))
        print("Assistant:", events[-1]["messages"][-1].content)
    
    else:
        print("Assistant:", events[-1]["messages"][-1].content)


# 無限ループを使用してユーザー入力を連続的に処理
while True:
    try:

        snapshot = graph.get_state(config)

        # ユーザーからの入力を取得
        user_input = input("User: ")

        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break

        stream_graph_updates(user_input)

    except Exception as e:
        print(f"Error: {e}")
        break 