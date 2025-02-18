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

    # ChatModelの準備
    llm_deepseek_v3 = ChatDeepSeek(
        model="deepseek-chat",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
    )
    llm_deepseek_r1 = ChatDeepSeek(model="deepseek-reasoner")
    
    prompt1 = ChatPromptTemplate.from_template(
"""
以下の質問に回答してください。

# 質問
{question}
"""
    )

    chain1 = prompt1 | llm_deepseek_v3 | StrOutputParser()

    question = "円周率が3.14の理由を教えて"
    answer = chain1.invoke({"question": question})
    print(answer)

    print("\n---------------------------------------------------------------------\n")


    prompt2 = ChatPromptTemplate.from_messages(
        [
            ("human", 
"""

# 質問
{question}

"""
            ),
        ]
    )

    chain2 = prompt2 | llm_deepseek_r1 | StrOutputParser()

    question = "円周率が3.14の理由を教えて"
    ai_message = chain2.invoke({"question": question})
    print(ai_message)


if __name__=="__main__":
    print("\n---------------------------------------------------------------------\n")
    main()
    print("\n---------------------------------------------------------------------\n")
