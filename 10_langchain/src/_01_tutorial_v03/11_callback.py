import os
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_openai import ChatOpenAI, AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser


from langchain_community.callbacks.manager import get_openai_callback

# .envファイルから環境変数を読み込み
load_dotenv()

def main():

    # ChatModelの準備
    chat_model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    # chat_model = AzureChatOpenAI(
    #     openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    #     azure_deployment=os.getenv("AZURE_CHAT_DEPLOYMENT"),
    #     temperature=0
    # )

    # ChainとChain

    cot_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "ユーザーの質問にステップバイステップで回答してください。"),
            ("human", "{question}"),
        ]
    )

    cot_chain = cot_prompt | chat_model | StrOutputParser()

    summarize_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "ステップバイステップで考えた回答から結論だけ抽出してください。"),
            ("human", "{text}"),
        ]
    )

    summarize_chain = summarize_prompt | chat_model | StrOutputParser()

    cot_summarize_chain = cot_chain | summarize_chain


    with get_openai_callback() as cb:
        answer = cot_summarize_chain.invoke({"question": "10 + 2 * 3"})

        print()
        print(f"Total Tokens: {cb.total_tokens}")
        print(f"Prompt Tokens: {cb.prompt_tokens}")
        print(f"Completion Tokens: {cb.completion_tokens}")
        print(f"Total Cost (USD): ${cb.total_cost}")



    print(answer)

if __name__=="__main__":
    print("\n---------------------------------------------------------------------\n")
    main()
    print("\n---------------------------------------------------------------------\n")

