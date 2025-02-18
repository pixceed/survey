"""
質問から、目標を設定する
"""

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field


class Goal(BaseModel):
    description: str = Field(..., description="目標の説明")

class GoalCreator:
    def __init__(
        self,
        llm: ChatOpenAI,
    ):
        self.llm = llm

    def run(self, query: str) -> Goal:
        prompt = ChatPromptTemplate.from_template(
"""
ユーザーの入力を分析し、明確で実行可能な目標を設定してください。

要件
1. 目標は具体的かつ明確であり、実行可能なレベルで詳細化されている必要があります。

ユーザーの入力: {query}
"""
        )
        chain = prompt | self.llm.with_structured_output(Goal)
        return chain.invoke({"query": query})
