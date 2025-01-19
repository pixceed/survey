import os
from dotenv import load_dotenv
from markitdown import MarkItDown
from openai import AzureOpenAI


# .envファイルから環境変数を読み込み
load_dotenv()


def main():

    file_path = "input/test_data.xlsx"
    # file_path = "input/20241226_ゲームシーンAI_作業報告1_v4.pptx"


    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    key = os.getenv("AZURE_OPENAI_API_KEY")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION")
    model_name = os.getenv("AZURE_CHAT_MODEL_NAME")
    aoai_client = AzureOpenAI(azure_endpoint=endpoint, api_key=key, api_version=api_version)


    markitdown = MarkItDown(mlm_client=aoai_client, mlm_model=model_name)

    result = markitdown.convert(file_path)
    print(result.text_content)

if __name__=="__main__":
    main()