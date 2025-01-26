"""
目標をタスク分解する
"""
from datetime import datetime

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field


class DecomposedTasks(BaseModel):
    values: list[str] = Field(
        default_factory=list,
        min_items=3,
        max_items=20,
        description="3~20個に分解されたタスク",
    )

class TaskDecomposer:
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
        self.current_date = datetime.now().strftime("%Y-%m-%d")

    def run(self, query: str) -> DecomposedTasks:
        prompt = ChatPromptTemplate.from_template(

f"""
CURRENT_DATE: {self.current_date}
-----

タスク: 与えられた目標に対して具体的で実行可能なタスクに分解してください。

要件:
1. 以下の行動を活用し目標を達成すること。※各ツールは既にアクセス済みの状態
   - 決算RAGツールを利用して、決算に関する情報の検索を行う。
2. 各タスクは具体的かつ詳細に記載されており、単独で実行ならびに検証可能な情報を含めること。一切抽象的な表現を含まないこと。
3. タスクは実行可能な順序でリスト化すること。また、前のタスクの実行結果に基づいたタスクは生成しないでください。
4. タスクは日本語で出力すること。

目標: {query}
"""
        )
        chain = prompt | self.llm.with_structured_output(DecomposedTasks)
        return chain.invoke({"query": query})