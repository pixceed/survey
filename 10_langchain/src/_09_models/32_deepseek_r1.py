import os
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser

from langchain_deepseek import ChatDeepSeek

# .envファイルから環境変数を読み込み
load_dotenv()

def main():

    ##########################################
    # モデルの準備
    ##########################################

    llm_deepseek_r1 = ChatDeepSeek(model="deepseek-reasoner")    

    ##########################################
    # プロンプト
    ##########################################

    prompt = ChatPromptTemplate.from_messages(
        [
            ("human", 
"""
# 質問
{question}

"""
            ),
        ]
    )

    ##########################################
    # 実行
    ##########################################

    chain = prompt | llm_deepseek_r1 | StrOutputParser()

    question = "円周率が3.14の理由を教えて"
    ai_message = chain.invoke({"question": question})
    print(ai_message)


if __name__=="__main__":
    print("\n---------------------------------------------------------------------\n")
    main()
    print("\n---------------------------------------------------------------------\n")
        