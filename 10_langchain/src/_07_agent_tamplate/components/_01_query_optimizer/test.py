import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI

from orchestration.components._01_query_optimizer.agent import QueryOptimizer

# .envファイルから環境変数を読み込み
load_dotenv()


def main():

    # query = "今後成長するためにどのような計画を考えていますか？"

#     query = \
# """
# 貴社事業は、回ごとに収入を得るフロー型ビジネスでしょうか？
# 長期契約やリピート率が高いなどストック型ビジネスでしょうか？
# """
    query = "どこから、どのようにして収益を得ているのでしょうか。"

    # LLM
    llm = AzureChatOpenAI(
        openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        azure_deployment=os.getenv("AZURE_CHAT_DEPLOYMENT"),
        temperature=0
    )

    qo = QueryOptimizer(llm=llm)

    optimized_query = qo.run(query=query)

    print(f"[最適化プロンプト]\n{optimized_query}\n\n")


if __name__ == "__main__":
    main()