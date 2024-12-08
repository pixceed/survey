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
    chat_model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    # chat_model = AzureChatOpenAI(
    #     openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    #     azure_deployment=os.getenv("AZURE_CHAT_DEPLOYMENT"),
    #     temperature=0
    # )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "ユーザーが入力した料理のレシピを教えてください。"),
            ("human", "{dish}"),
        ]
    )

    chain = prompt | chat_model

    ai_message = chain.invoke({"dish": "カレー"})
    print(ai_message.content)

    print("\n---------------------------------------------------------------------\n")

    class Recipe(BaseModel):
        ingredients: list[str] = Field(description="ingredients of the dish")
        steps: list[str] = Field(description="steps to make the dish")

    output_parser = PydanticOutputParser(pydantic_object=Recipe)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "ユーザーが入力した料理のレシピを教えてください。\n\n"
                "{format_instructions}",
            ),
            ("human", "{dish}"),
        ]
    )
    prompt_with_format_instructions = prompt.partial(
        format_instructions=output_parser.get_format_instructions()
    )

    # ChatModelの準備
    chat_model = ChatOpenAI(model="gpt-4o-mini", temperature=0).bind(
        response_format={"type": "json_object"} # JSONモードを有効化
    )

    chain = prompt_with_format_instructions | chat_model | output_parser

    recipe = chain.invoke({"dish": "カレー"})
    print()
    print(type(recipe))
    print(recipe)

    print("\n---------------------------------------------------------------------\n")

    # with_structured_output を使う場合
    # 構造化データの出力

    class Recipe(BaseModel):
        ingredients: list[str] = Field(description="ingredients of the dish")
        steps: list[str] = Field(description="steps to make the dish")


    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "ユーザーが入力した料理のレシピを教えてください。"),
            ("human", "{dish}"),
        ]
    )

    # ChatModelの準備
    chat_model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    chain = prompt | chat_model.with_structured_output(Recipe)

    recipe = chain.invoke({"dish": "カレー"})

    print(type(recipe))
    print(recipe)

    print("\n---------------------------------------------------------------------\n")

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

    answer = cot_summarize_chain.invoke({"question": "10 + 2 * 3"})

    print(answer)

if __name__=="__main__":
    print("\n---------------------------------------------------------------------\n")
    main()
    print("\n---------------------------------------------------------------------\n")

