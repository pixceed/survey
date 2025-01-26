"""
質問の詳細化
"""

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field


class OptimizedGoal(BaseModel):
    description: str = Field(..., description="最適化した目標の説明")

class OptimizedGoalCreator:
    def __init__(
        self,
        llm: ChatOpenAI,
    ):
        self.llm = llm

    def run(self, query: str) -> OptimizedGoal:
        prompt = ChatPromptTemplate.from_template(

"""
あなたは、目標設定の専門家です。
以下の目標を具体的に最適化してください。

元の目標: {query}

指示:
1. 元の目標を分析し、不足している要素や改善点を特定してください。
2. あなたが実行可能な行動は以下の行動だけです。
  - 決算RAGツールを利用して、決算に関する情報の検索を行う。
  - ユーザーのためのレポートを生成する。
3. 目標を具体的かつ詳細に記載してください。
  - 一切抽象的な表現を含んではいけません。
  - 必ず全ての単語が実行可能かつ具体的であることを確認してください。
4. 目標の達成度を測定する方法を具体的かつ詳細に記載してください。
5. 「レポートを配布する」などメタ的な表現は避けてください。
6. 元の質問のすべての情報を維持してください。
"""
        )
        chain = prompt | self.llm.with_structured_output(OptimizedGoal)
        return chain.invoke({"query": query})
