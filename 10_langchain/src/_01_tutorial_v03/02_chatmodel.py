import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, AzureChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

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

    # ChatModelの実行
    messages = [
        HumanMessage(content="コンピュータゲームを作る日本語の新会社名をを1つ提案してください。")
    ]
    output = chat_model.invoke(messages)
    print(type(output))
    print(output)

    print("\n---------------------------------------------------------------------\n")

    messages = [
        SystemMessage("You are a helpful assistant."),
        HumanMessage("こんにちは!私はMOIといいます!"),
        AIMessage(content="こんにちは、MOIさん!どのようにお手伝いできますか？"),
        HumanMessage(content="私の名前が分かりますか？")
    ]

    ai_message = chat_model.invoke(messages)
    print(ai_message.content)


    print("\n---------------------------------------------------------------------\n")
    # ストリーミング出力
    messages = [
        SystemMessage("You are a helpful assistant."),
        HumanMessage("こんにちは!ポケモンに関する都市伝説を話して!")
    ]

    for chunk in chat_model.stream(messages):
        print(chunk.content, end="", flush=True)
    print()

    
if __name__=="__main__":
    main()
