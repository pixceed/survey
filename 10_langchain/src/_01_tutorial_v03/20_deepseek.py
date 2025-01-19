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
    chat_model = ChatOpenAI(
        model="deepseek-chat",
        base_url="https://api.deepseek.com/v1",
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        temperature=0
    )
    


    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "ユーザーが入力した料理のレシピを教えてください。"),
            ("human", "{dish}"),
        ]
    )

    chain = prompt | chat_model

    ai_message = chain.invoke({"dish": "カレー"})
    print(ai_message.content)


if __name__=="__main__":
    print("\n---------------------------------------------------------------------\n")
    main()
    print("\n---------------------------------------------------------------------\n")
