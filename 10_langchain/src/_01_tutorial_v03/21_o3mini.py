import os
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_openai import ChatOpenAI, AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser

# .envファイルから環境変数を読み込み
load_dotenv()

def main():

    # ChatModelの準備
    llm_o3_mini = ChatOpenAI(model="o3-mini")
    # question = "円周率が3.14の理由を教えて"
    # output = llm_o3_mini.invoke(question)
    
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

    chain = prompt | llm_o3_mini | StrOutputParser()

    question = "円周率が3.14の理由を教えて"
    ai_message = chain.invoke({"question": question})
    print(ai_message)


if __name__=="__main__":
    print("\n---------------------------------------------------------------------\n")
    main()
    print("\n---------------------------------------------------------------------\n")
