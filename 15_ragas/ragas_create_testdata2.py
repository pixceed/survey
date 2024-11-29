'''
RAGASによるテストデータ生成
'''

import os
import pandas as pd
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.testset import TestsetGenerator

from ragas.testset.graph import KnowledgeGraph
from ragas.testset.graph import Node, NodeType
from ragas.testset.transforms import default_transforms, apply_transforms
from ragas.testset.synthesizers import default_query_distribution

# .envファイルから環境変数（APIキーなど）を読み込み
load_dotenv()


def main():

    # ＜LLMと埋め込みモデルの準備＞
    # LLMの準備
    llm = ChatOpenAI(
        model="gpt-4o", # モデル
        temperature=0, # ランダムさ
    )
    # 埋め込みモデルの準備
    embeddings = OpenAIEmbeddings()

    # ＜ドキュメントの読み込みと分割＞
    # ドキュメントの読み込み
    # loader = DirectoryLoader("./sample_data/")
    # path = "./sample_data/Sample_Docs_Markdown/"
    # loader = DirectoryLoader(path, glob="**/*.md")
    loader = TextLoader("./sample_data/bocchi.md")
    documents = loader.load()

    # ＜テストデータ生成器の準備＞
    generator_llm = LangchainLLMWrapper(llm)
    generator_embeddings = LangchainEmbeddingsWrapper(embeddings)

    # ＜Knowledge Graphの作成＞
    kg = KnowledgeGraph()
    for doc in documents:
        kg.nodes.append(
            Node(
                type=NodeType.DOCUMENT,
                properties={"page_content": doc.page_content, "document_metadata": doc.metadata}
            )
        )

    transformer_llm = generator_llm
    embedding_model = generator_embeddings

    trans = default_transforms(documents, llm=transformer_llm, embedding_model=embedding_model)
    apply_transforms(kg, trans)

    # ＜Knowledge Graphの保存＞
    kg_path = "output/knowledge_graph_bocchi.json"
    kg.save(kg_path)
    loaded_kg = KnowledgeGraph.load(kg_path)

    # ＜テストデータを生成するジェネレータを作成＞
    generator = TestsetGenerator(llm=generator_llm, embedding_model=generator_embeddings ,knowledge_graph=loaded_kg)
    query_distribution = default_query_distribution(generator_llm)

    # Pandasの表示オプションを設定
    pd.set_option('display.max_columns', None)  # すべての列を表示
    pd.set_option('display.max_rows', None)     # すべての行を表示
    pd.set_option('display.expand_frame_repr', False)  # 列を折り返さない

    # ＜Testsetの生成＞
    testset = generator.generate(testset_size=10, query_distribution=query_distribution)

    test_df = testset.to_pandas()

    print(test_df)
    test_df.to_csv("output/testset_kg_bocchi.csv")

if __name__=="__main__":
    main()