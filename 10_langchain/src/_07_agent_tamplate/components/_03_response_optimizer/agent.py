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
"""
あなたはAIエージェントシステムのレスポンス最適化スペシャリストです。
与えられた質問クエリに対して、エージェントが適切なレスポンスを返すためのレスポンス仕様を策定してください。
""",
                ),
                (
                    "human",
"""
以下の手順に従って、レスポンス最適化プロンプトを作成してください：

1. 質問クエリ分析:
提示された質問クエリを分析し、主要な要素や意図を特定してください。

2. レスポンス仕様の策定:
質問への適切な回答を行うための最適なレスポンス仕様を考案してください。
トーン、構造、内容の焦点などを考慮に入れてください。

3. 具体的な指示の作成:
事前に収集された情報から、ユーザーの期待に沿ったレスポンスをするために必要な、AIエージェントに対する明確で実行可能な指示を作成してください。
あなたの指示によってAIエージェントが実行可能なのは、既に調査済みの結果をまとめることだけです。
インターネットへのアクセスはできません。

以下の構造でレスポンス最適化プロンプトを出力してください:

質問クエリ分析:
[ここに質問クエリの分析結果を記入]

レスポンス仕様:
[ここに策定されたレスポンス仕様を記入]

AIエージェントへの指示:
[ここにAIエージェントへの具体的な指示を記入]

では、以下の質問クエリに対するレスポンス最適化プロンプトを作成してください:

質問クエリ: {query}
""",
                ),
            ]
        )
        chain = prompt | self.llm | StrOutputParser()
        return chain.invoke({"query": query})
