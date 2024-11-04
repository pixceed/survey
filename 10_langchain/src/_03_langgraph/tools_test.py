from dotenv import load_dotenv
from langchain_community.tools.tavily_search import TavilySearchResults


# .envファイルから環境変数（APIキーなど）を読み込み
load_dotenv()


# Tavilyの準備
tavily_tool = TavilySearchResults()

# 動作確認
search_results = tavily_tool.invoke("2024年夏の日本で一番人気のアニメは？")

for result in search_results:
    print()
    print("-"*80)
    print(result)
    print()