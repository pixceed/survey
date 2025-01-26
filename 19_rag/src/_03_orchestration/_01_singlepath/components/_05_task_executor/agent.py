"""
タスクを実行する
"""

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

from pydantic import BaseModel, Field

from langgraph.prebuilt import create_react_agent

class TaskExecutor:
    def __init__(self, llm: ChatOpenAI, tools):
        self.llm = llm
        self.tools = tools

    def run(self, task: str, task_exec_history: str) -> str:
        agent = create_react_agent(self.llm, self.tools)
        result = agent.invoke(
            {
                "messages": [
                    (
                        "human",
                        (
f"""
次の<タスク></タスク>を実行し、詳細な回答を提供してください。

<タスク>
{task}
</タスク>

<これまでのタスク実行結果>
{task_exec_history}
</これまでのタスク実行結果>

[指示]
1. 必要に応じて提供されたツールを使用してください。
2. 実行は徹底的かつ包括的に行ってください。
3. 可能な限り具体的な事実やデータを提供してください。
4. 発見した内容を明確に要約してください。
"""
                        ),
                    )
                ]
            }
        )
        return result["messages"][-1].content