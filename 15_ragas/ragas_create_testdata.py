'''
RAGASによるテストデータ生成
'''

import os
import pandas as pd
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import DirectoryLoader, TextLoader

from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.testset import TestsetGenerator


# .envファイルから環境変数（APIキーなど）を読み込み
load_dotenv()


def main():

    # ＜LLMと埋め込みモデルの準備＞
    # LLMの準備
    llm = ChatOpenAI(
        model="gpt-4o-mini", # モデル
        temperature=0, # ランダムさ
    )
    # 埋め込みモデルの準備
    embeddings = OpenAIEmbeddings()

    # ＜ドキュメントの読み込みと分割＞
    # ドキュメントの読み込み
    # loader = DirectoryLoader("./sample_data/")
    loader = TextLoader("./sample_data/bocchi.md")
    documents = loader.load()

    # ＜テストデータを生成するジェネレータを作成＞
    generator_llm = LangchainLLMWrapper(llm)
    generator_embeddings = LangchainEmbeddingsWrapper(embeddings)
    generator = TestsetGenerator(llm=generator_llm, embedding_model=generator_embeddings)

    # # 作成する質問情報
    # distributions = {
    #     simple: 0.5,
    #     multi_context: 0.4,
    #     reasoning: 0.1
    # }

    # Pandasの表示オプションを設定
    pd.set_option('display.max_columns', None)  # すべての列を表示
    pd.set_option('display.max_rows', None)     # すべての行を表示
    pd.set_option('display.expand_frame_repr', False)  # 列を折り返さない

    # ＜Testsetの生成＞
    testset = generator.generate_with_langchain_docs(documents, testset_size=10)

    test_df = testset.to_pandas()

    print(test_df)
    test_df.to_csv("output/testset.csv")

if __name__=="__main__":
    main()