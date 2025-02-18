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
        max_items=8,
        description="3~8個に分解されたタスク",
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

タスク:
与えられた質問クエリの<問い合わせ></問い合わせ>に対して、適切な返答案を作成するために必要なタスクを作成してください。

要件:
- 以下の行動を活用し、質問クエリの<問い合わせ></問い合わせ>に対する適切な返答をすることを目標とする。※各ツールは既にアクセス済みの状態
   - 決算RAGツール
   - 会社情報RAGツール
- 各タスクは具体的かつ詳細に記載されており、単独で実行ならびに検証可能な情報を含めること。一切抽象的な表現を含まないこと。
- タスクは実行可能な順序でリスト化すること。前のタスクの実行結果に依存するタスクは生成しないでください。
- タスクは日本語で出力すること。

質問クエリ: {query}
"""
        )
        chain = prompt | self.llm.with_structured_output(DecomposedTasks)
        return chain.invoke({"query": query})