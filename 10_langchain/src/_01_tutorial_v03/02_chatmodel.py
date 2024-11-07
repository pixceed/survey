import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage

# .envファイルから環境変数を読み込み
load_dotenv()

def main():

    # ChatModelの準備
    chat_model = AzureChatOpenAI(
        openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        azure_deployment=os.getenv("AZURE_CHAT_DEPLOYMENT"),
        temperature=0
    )

    # ChatModelの実行
    messages = [
        HumanMessage(content="コンピュータゲームを作る日本語の新会社名をを1つ提案してください。")
    ]
    output = chat_model.invoke(messages)
    print(type(output))
    print(output)
    
if __name__=="__main__":
    main()
