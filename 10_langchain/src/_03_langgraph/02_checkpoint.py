import os
from dotenv import load_dotenv
from pprint import pprint

import operator
from typing import Annotated, Any

from pydantic import BaseModel, Field

from langchain_openai import ChatOpenAI, AzureChatOpenAI, OpenAIEmbeddings
from langchain_core.runnables import RunnableConfig
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, BaseMessage

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver

# .envファイルから環境変数を読み込み
load_dotenv()

def main():

    # ＜LLMと埋め込みモデルの準備＞
    # LLMの準備
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # ＜ステートの定義＞
    class State(BaseModel):
        query: str
        messages: Annotated[list[BaseMessage], operator.add] = Field(default=[])

    # ＜ノードの定義＞
    # メッセージを追加するノード関数
    def add_message(state: State) -> dict[str, Any]:
        addtional_messages = []
        if not state.messages:
            addtional_messages.append(
                SystemMessage(content="あなたは最小限の応答をする対話エージェントです。")
            )
        addtional_messages.append(HumanMessage(content=state.query))

        return {"messages": addtional_messages}

    # LLMからの応答を追加するノード関数
    def llm_response(state: State) -> dict[str, Any]:
        ai_message = llm.invoke(state.messages)
        return {"messages": [ai_message]}
    
    # チェックポイントの内容を表示するための関数を定義
    def print_checkpoint_dump(checkpointer: BaseCheckpointSaver, config: RunnableConfig):
        checkpoint_tuple = checkpointer.get_tuple(config)

        print("チェックポイントデータ:")
        pprint(checkpoint_tuple.checkpoint)
        print("\nメタデータ:")
        pprint(checkpoint_tuple.metadata)
    
    # ＜グラフの定義とコンパイル＞
    graph = StateGraph(State)
    
    graph.add_node("add_message", add_message)
    graph.add_node("llm_response", llm_response)

    graph.set_entry_point("add_message")
    graph.add_edge("add_message", "llm_response")
    graph.add_edge("llm_response", END)

    # チェックポインターを設定
    checkpointer = MemorySaver()

    # グラフをコンパイル
    compiled_graph = graph.compile(checkpointer=checkpointer)

    # ＜動作確認＞

    config = {"configurable": {"thread_id": "example-1"}}
    user_query = State(query="私の好きなものはずんだもんです。覚えておいてね。")
    first_response = compiled_graph.invoke(user_query, config)

    print("#"*80)
    print("#"*80)
    print("#"*80)
    print(first_response["messages"][-1].content)
    print()
    
    for checkpoint in checkpointer.list(config):
        print("==========================================")
        pprint(checkpoint)
        print()
        
    # print_checkpoint_dump(checkpointer, config)
    # print()
    dummy_messages = [
        SystemMessage(content="あなたは、語尾に「もい」が付きます。"),
        HumanMessage(content="やっぱり好きななのは、ドラえもん！"),
        AIMessage(content="分かったもい！")
    ]
    user_query = State(query="私の好きなものを覚えてる？",
                       messages=dummy_messages)
    # user_query = State(messages=dummy_messages)
    second_response = compiled_graph.invoke(user_query, config)

    print("#"*80)
    print("#"*80)
    print("#"*80)
    print(second_response["messages"][-1].content)
    print()

    for checkpoint in checkpointer.list(config):
        print("==========================================")
        pprint(checkpoint)
        print()

    # print_checkpoint_dump(checkpointer, config)
    # print()



if __name__=="__main__":
    print("\n---------------------------------------------------------------------\n")
    main()
    print("\n---------------------------------------------------------------------\n")

