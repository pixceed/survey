import os
from dotenv import load_dotenv
import tiktoken
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.callbacks import get_openai_callback

# .envファイルから環境変数を読み込み
load_dotenv()

# トークンの使用量を計算する関数
def count_tokens(text, model_name="gpt-4o"):
    encoding = tiktoken.encoding_for_model(model_name)
    return len(encoding.encode(text))

# 料金を計算する関数（ここでは仮に1トークンあたり0.001ドルとして計算）
def calculate_cost(input_tokens, output_tokens):
    input_cost = input_tokens * 0.0000025
    output_cost = output_tokens * 0.0000075
    return input_cost + output_cost

def main():

    # プロンプトテンプレートの準備
    prompt_template = PromptTemplate(
        input_variables=["product"],
        template="{product}を作る日本語の新会社名を1つ提案してください",
    )

    # ChatModelの準備
    chat_model = AzureChatOpenAI(
        openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        azure_deployment=os.getenv("AZURE_CHAT_DEPLOYMENT"),
        temperature=0
    )

    # OutputParserの準備
    output_parser = StrOutputParser()

    # チェーンをつなげて実行
    with get_openai_callback() as cb:
        product_name = "家庭用ロボット"
        chain = prompt_template | chat_model | output_parser
        output = chain.invoke({"product": product_name})
        print(output)

        # 結果を表示
        print()
        print(f"提案された会社名: {output}")
        print(f"使用したトークン数: {cb.total_tokens}")
        print(f"入力トークン数: {cb.prompt_tokens}")
        print(f"出力トークン数: {cb.completion_tokens}")
        print(f"API利用料金: ${cb.total_cost:.6f}")

    # # トークンの使用量を計算（入力プロンプトと出力の両方で計算）
    # prompt_text = prompt_template.template.format(product=product_name)
    # prompt_tokens = count_tokens(prompt_text)  # 入力トークン
    # response_tokens = count_tokens(output)     # 出力トークン

    # cost = calculate_cost(prompt_tokens, response_tokens)
    # print()
    # print("入力プロンプトのトークン数:", prompt_tokens)
    # print("出力のトークン数:", response_tokens)
    # print("コスト:", cost, "ドル")

if __name__=="__main__":
    main()
