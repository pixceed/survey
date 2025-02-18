import os
from dotenv import load_dotenv

import operator
from datetime import datetime
from typing import Annotated, Any

from langchain_openai import ChatOpenAI, AzureChatOpenAI
from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, AzureChatOpenAI


from langgraph.graph import END, StateGraph
from langgraph.prebuilt import tools_condition, ToolNode, create_react_agent

from pydantic import BaseModel, Field

from tutorial.react_tutorial.tools.calc_tools import add, multiply, divide


class ReActAgentState(BaseModel):
    query: str = Field(..., description="ユーザーが入力したクエリ")

    # ★"messages"という名前が必須
    messages: Annotated[list[AnyMessage], operator.add]  = Field(default_factory=list, description="メッセージリスト")

class ReActAgent:

    def __init__(self, llm: ChatOpenAI):

        self.llm = llm

        self.tools = [add, multiply, divide]
        self.llm_with_tools = llm.bind_tools(self.tools)

        self.graph = self._create_graph()


    def _create_graph(self) -> StateGraph:

        graph_builder = StateGraph(ReActAgentState)

        # ノード設定
        graph_builder.add_node("assistant", self._assistant)
        graph_builder.add_node("tools", ToolNode(self.tools))

        # フロー設定
        graph_builder.set_entry_point("assistant")
        graph_builder.add_conditional_edges(
            "assistant",
            self._router,
            {"continue": "assistant", "call_tool": "tools", "end":  END}
        )
        graph_builder.add_edge("tools", "assistant")

        return graph_builder.compile()

    def _router(self, state: ReActAgentState) -> str:
        """
        最終メッセージから、条件分岐に関係する情報を取り出し、条件付きエッジに渡す
        """
        messages = state.messages
        last_message = messages[-1]

        # ツール呼び出しが行われている場合
        if "tool_calls" in last_message.additional_kwargs:
            return "call_tool"

        # 最終回答が生成されている場合
        if "FINAL ANSWER" in last_message.content:
            return "end"

        return "continue"

    def _assistant(self, state: ReActAgentState) -> dict[str, Any]:

        # メッセージ履歴取得
        before_messages = state.messages
        if len(before_messages) > 0:
            print("★"*80)

            for message in before_messages:
                message_print = message.pretty_print()
                if message_print is not None:
                    print(message_print)
            print("★"*80)

        user_prompt = {
                "role": "user",
                "content": state.query
            }
        before_messages.append(user_prompt)

        # システムプロンプト
        system_prompt = {
            "role": "system",
            "content":
"""
あなたは他のアシスタントと協力して、役に立つ AI アシスタントです。
提供されたツールを使用して、質問の回答に進みます。
あなたまたは他のアシスタントが最終的な回答または成果物を持っている場合は、チームが停止することがわかるように、回答の前に「FINAL ANSWER」を付けます。
次のツールにアクセスできます:
[使用可能ツール]
・Multiply
・Adds
・Divide
"""
        }

        # ツール付LLMでメッセージを取得する
        output_message = self.llm_with_tools.invoke([system_prompt] + before_messages)

        return {"messages": [output_message]}


    def run(self, query: str) -> str:

        intial_state = ReActAgentState(query=query)

        final_state = self.graph.invoke(intial_state)

        return final_state.get("messages")



def main():

    # query = "3と4を足す。その出力に2を掛ける。 さらにその出力を5で割る。次にその出力を小数点第一位で四捨五入する。最後にその出力を5で割る。"
    query = "3と4を足す。その出力を5で割る。次に小数点第一位で四捨五入する。その出力に1を引く。その出力に2を掛ける。 さらに最後にその数を100倍する。"

    llm = AzureChatOpenAI(
        openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        azure_deployment=os.getenv("AZURE_CHAT_DEPLOYMENT"),
        temperature=0
    )

    agent = ReActAgent(llm=llm)
    result_messages = agent.run(query=query)

    for message in result_messages:
        message_print = message.pretty_print()
        if message_print is not None:
            print()
            print(message_print)

    print("#"*80)

    print(f"\n{result_messages[-1].content}")

if __name__ == "__main__":


    main()