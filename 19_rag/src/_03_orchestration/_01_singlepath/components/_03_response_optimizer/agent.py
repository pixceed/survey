"""
質問の詳細化
"""

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field



class ResponseOptimizer:
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm

    def run(self, query: str) -> str:
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
"あなたはAIエージェントシステムのレスポンス最適化スペシャリストです。与えられた目標に対して、エージェントが目標にあったレスポンスを返すためのレスポンス仕様を策定してください。",
                ),
                (
                    "human",
"""
以下の手順に従って、レスポンス最適化プロンプトを作成してください：

1. 目標分析:
提示された目標を分析し、主要な要素や意図を特定してください。

2. レスポンス仕様の策定:
目標達成のための最適なレスポンス仕様を考案してください。
トーン、構造、内容の焦点などを考慮に入れてください。

3. 具体的な指示の作成:
事前に収集された情報から、ユーザーの期待に沿ったレスポンスをするために必要な、AIエージェントに対する明確で実行可能な指示を作成してください。
あなたの指示によってAIエージェントが実行可能なのは、既に調査済みの結果をまとめることだけです。
インターネットへのアクセスはできません。

4. 例の提供:
可能であれば、目標に沿ったレスポンスの例を1つ以上含めてください。

5. 評価基準の設定:
レスポンスの効果を測定するための基準を定義してください。

以下の構造でレスポンス最適化プロンプトを出力してください:

目標分析:
[ここに目標の分析結果を記入]

レスポンス仕様:
[ここに策定されたレスポンス仕様を記入]

AIエージェントへの指示:
[ここにAIエージェントへの具体的な指示を記入]

レスポンス例:
[ここにレスポンス例を記入]

評価基準:
[ここに評価基準を記入]

では、以下の目標に対するレスポンス最適化プロンプトを作成してください:

目標: {query}
""",
                ),
            ]
        )
        chain = prompt | self.llm | StrOutputParser()
        return chain.invoke({"query": query})
