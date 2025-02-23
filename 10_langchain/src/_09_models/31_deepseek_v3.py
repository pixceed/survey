import os
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_openai import ChatOpenAI, AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser

from langchain_deepseek import ChatDeepSeek

# .envファイルから環境変数を読み込み
load_dotenv()

def main():

    ##########################################
    # モデルの準備
    ##########################################

    llm_deepseek_v3 = ChatDeepSeek(
        model="deepseek-chat",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
    )


    ##########################################
    # プロンプト
    ##########################################

    prompt = ChatPromptTemplate.from_template(
"""
以下の質問に回答してください。

# 質問
{question}
"""
    )

    ##########################################
    # 実行
    ##########################################

    chain1 = prompt | llm_deepseek_v3 | StrOutputParser()

    question = "円周率が3.14の理由を教えて"
    answer = chain1.invoke({"question": question})
    print(answer)


if __name__=="__main__":
    print("\n---------------------------------------------------------------------\n")
    main()
    print("\n---------------------------------------------------------------------\n")
        