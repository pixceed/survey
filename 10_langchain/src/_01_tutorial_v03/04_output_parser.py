import os
from dotenv import load_dotenv

from pydantic import BaseModel, Field

from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# .envファイルから環境変数を読み込み
load_dotenv()

def main():

    # Output Parserの準備
    output_parser = StrOutputParser()

    # Output Parserの実行
    message = AIMessage(content="AIからのメッセージです")
    output = output_parser.invoke(message)
    print(type(output))
    print(output)

    print("\n---------------------------------------------------------------------\n")

    class Recipe(BaseModel):
        ingredients: list[str] = Field(description="ingredients of the dish")
        steps: list[str] = Field(description="steps to make the dish")

    output_parser = PydanticOutputParser(pydantic_object=Recipe)

    format_instructions = output_parser.get_format_instructions()
    print(f"整形するためのプロンプト\n####\n{format_instructions}\n####")

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

    # `partial`：一部だけセットする
    prompt_with_format_instructions = prompt.partial(
        format_instructions=format_instructions
    )
    prompt_value = prompt_with_format_instructions.invoke({"dish": "カレー"})
    
    print("=== role: system ===")
    print(prompt_value.messages[0].content)

    print("=== role: user ===")
    print(prompt_value.messages[1].content)


    chat_model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    ai_message = chat_model.invoke(prompt_value)
    print()
    print(ai_message.content)

    recipe = output_parser.invoke(ai_message)
    print()
    print(type(recipe))
    print(recipe)


if __name__=="__main__":
    print("\n---------------------------------------------------------------------\n")
    main()
    print("\n---------------------------------------------------------------------\n")
