"""
＜プロンプト最適化＞
生成された目標やユーザー要求を、より効果的なプロンプトに変換し、
LLMからより質の高い回答を得るためのパターン
"""

from dotenv import load_dotenv

# .envファイルから環境変数を読み込み
load_dotenv()

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from src._05_design_pattern._01_passive_goal_creator import Goal, PassiveGoalCreator


class OptimizedGoal(BaseModel):
    description: str = Field(..., description="目標の説明")
    metrics: str = Field(..., description="目標の達成度を測定する方法")

    @property
    def text(self) -> str:
        return f"{self.description}(測定基準: {self.metrics})"


class PromptOptimizer:
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm

    def run(self, query: str) -> OptimizedGoal:
        prompt = ChatPromptTemplate.from_template(
            "あなたは目標設定の専門家です。以下の目標をSMART原則（Specific: 具体的、Measurable: 測定可能、Achievable: 達成可能、Relevant: 関連性が高い、Time-bound: 期限がある）に基づいて最適化してください。\n\n"
            "元の目標:\n"
            "{query}\n\n"
            "指示:\n"
            "1. 元の目標を分析し、不足している要素や改善点を特定してください。\n"
            "2. あなたが実行可能な行動は以下の行動だけです。\n"
            "   - インターネットを利用して、目標を達成するための調査を行う。\n"
            "   - ユーザーのためのレポートを生成する。\n"
            "3. SMART原則の各要素を考慮しながら、目標を具体的かつ詳細に記載してください。\n"
            "   - 一切抽象的な表現を含んではいけません。\n"
            "   - 必ず全ての単語が実行可能かつ具体的であることを確認してください。\n"
            "4. 目標の達成度を測定する方法を具体的かつ詳細に記載してください。\n"
            "5. 元の目標で期限が指定されていない場合は、期限を考慮する必要はありません。\n"
            "6. REMEMBER: 決して2.以外の行動を取ってはいけません。"
        )
        chain = prompt | self.llm.with_structured_output(OptimizedGoal)
        return chain.invoke({"query": query})


def main():

    task = "カレーライスの作り方"
    
    llm = ChatOpenAI(model="gpt-4o", temperature=0)

    passive_goal_creator = PassiveGoalCreator(llm=llm)
    goal: Goal = passive_goal_creator.run(query=task)

    prompt_optimizer = PromptOptimizer(llm=llm)
    optimised_goal: OptimizedGoal = prompt_optimizer.run(query=goal.text)

    print(f"{optimised_goal.text}")


if __name__ == "__main__":
    main()