"""
目標をタスク分解する
"""
from datetime import datetime

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field


class DynamicTask(BaseModel):
    task_content: str = Field(default="", description="タスク内容")
    task_result: str = Field(default="", description="タスク実行結果")

class DynamicTaskController:
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm


    def run(self, query: str, task_history: list[DynamicTask]) -> str:

        task_history_text = ""
        if len(task_history) == 0:
            task_history_text = "タスク実行結果無し"

        else:

            for i, task in enumerate(task_history):
                task_content = task.task_content
                task_result = task.task_result
                task_history_text += f"[タスク{i+1}]\n\nタスク内容:\n{task_content}\n\nタスク実行結果:\n{task_result}\n\n"

        prompt = ChatPromptTemplate.from_template(

f"""
あなたは、天才的なタスク管理者です。
与えられた<質問>と<これまでのタスクとタスク実行結果>を元に、タスク制御を行います。

<質問>
{query}
</質問>

<これまでのタスクとタスク実行結果>
{task_history_text}
</これまでのタスクとタスク実行結果>

[指示]
・<これまでのタスクとタスク実行結果>が、"タスク実行結果無し"の場合は、最初に行うべきタスクを生成してください。
・<質問>と<これまでのタスクとタスク実行結果>から、これ以上タスクを実行する必要が無いと判断できる場合は、"FINISH"と出力してください。
・作成するタスクは、必ず1ステップとしてください。
・余計な文言は絶対に出力しないでください。
"""
        )
        chain = prompt | self.llm | StrOutputParser()
        return chain.invoke({"query": query, "task_history":task_history_text})