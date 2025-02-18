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

    def run(self, task: str) -> str:
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

[指示]
・タスク内容を十分に理解し、必要であれば提供されたツールを使用してください。
・必ず、具体的な事実やデータに基づいて、回答を生成してください。嘘は絶対に付かないでください。
・タスク内容の前提が間違っていることを考慮してください。具体的な事実やデータに基づいて、冷徹に分析し、真実を回答してください。
"""
                        ),
                    )
                ]
            }
        )
        return result["messages"][-1].content

class TaskExecutor2:
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
・タスク内容を十分に理解し、必要であれば提供されたツールを使用してください。
・必ず、具体的な事実やデータに基づいて、回答を生成してください。嘘は絶対に付かないでください。
・タスク内容の前提が間違っていることを考慮してください。具体的な事実やデータに基づいて、冷徹に分析し、真実を回答してください。
"""
                        ),
                    )
                ]
            }
        )
        return result["messages"][-1].content