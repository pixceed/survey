"""
質問から、目標を設定する
"""

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# class OptimizedQuery(BaseModel):
#     description: str = Field(..., description="質問の説明")

class QueryOptimizer:
    def __init__(
        self,
        llm: ChatOpenAI,
    ):
        self.llm = llm

    def run(self, query: str) -> str:
    
        prompt = ChatPromptTemplate.from_template(
"""
あなたは、ソフトウェア会社のIR担当です。
投資家からの質問を分析し、より丁寧にしてください。

投資家からの質問: {query}

[指示]
・自身が、ソフトウェア会社であることを想定してください。
・投資家からの質問をメタ的に理解し、元質問と意味合いが変わらないように、最適化してください。
・余計な文言は絶対に出力しないでください。
"""
        )

        optimized_prompt = ChatPromptTemplate.from_template(
"""
投資家から、以下の<問い合わせ></問い合わせ>が来ました。
<問い合わせ></問い合わせ>に関して、投資家に対する返答内容を考えています。

<問い合わせ>
{optimized_question}
</問い合わせ>

[指示]
・メタ的な思考も取り入れて、多角的に返答を考えてください。
・投資家に良い印象を与えるような表現で、またシンプルに分かりやすい表現で、返答案を提示してください。
・問い合わせに対して、絶対に誤った情報を返答をしないでください。返答は必ずデータに基づいてください。
・絶対にマークダウン形式で出力しないでください。
"""
        )


        chain = RunnablePassthrough.assign(
            optimized_question=prompt | self.llm | StrOutputParser()
        ) | optimized_prompt \
          | (lambda x: x.messages[0].content)

        optimized_query = chain.invoke({"query": query})


        return optimized_query