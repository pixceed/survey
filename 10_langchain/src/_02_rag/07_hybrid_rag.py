import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, AzureChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import GitLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel

from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

# .envファイルから環境変数を読み込み
load_dotenv()

def main():

    # ＜LLMと埋め込みモデルの準備＞
    # ChatModelの準備

    chat_model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    # 埋め込みモデルの準備
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # ＜ドキュメントの読み込み＞
    def file_filter(file_path: str) -> bool:
        return file_path.endswith(".mdx")
    loader = GitLoader(
        clone_url="https://github.com/langchain-ai/langchain",
        repo_path="./langchain",
        branch="master",
        file_filter=file_filter
    )
    raw_docs = loader.load()

    # ＜TextSplitterによるドキュメント分割＞
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, # ドキュメントサイズ (トークン数)
        chunk_overlap=0, # 前後でオーバーラップするサイズ
        separators=["\n\n"] # セパレーター
    )
    docs = text_splitter.split_documents(raw_docs)


    # ＜VectorStoreの作成＞
    db = Chroma.from_documents(docs, embeddings)

    # 検索結果をRRFで並べるRAG-Fusion
    def reciprocal_rank_fusion(
        retriever_outputs: list[list[Document]],
        k: int = 60
    ) -> list[str]:
        
        # 各ドキュメントのコンテンツ（文字列）とそのスコアの対応を保持する辞書を準備
        content_score_mapping = {}

        # 検索クエリごとにループ
        for docs in retriever_outputs:
            # 検索結果のドキュメントごとにループ
            for rank, doc in enumerate(docs):
                content = doc.page_content

                # 初めて登場したコンテンツの場合はスコアを0で初期化
                if content not in content_score_mapping:
                    content_score_mapping[content] = 0
                
                # (1 / (順位 + k))のスコアを加算
                content_score_mapping[content] += 1 / (rank + k)

        # スコアの大きい順にソート
        ranked = sorted(content_score_mapping.items(), key=lambda x:x[1], reverse=True)

        return [content for content, _ in ranked]


    # ＜RAGのチェーンの作成＞
    retriever = db.as_retriever()

    chroma_retriever = retriever.with_config(
        {"run_name": "chroma_retriever"}
    )

    # BM25を使った検索のReriever
    bm25_retriever = BM25Retriever.from_documents(docs).with_config(
        {"run_name": "bm25_retriever"}
    )

    hybrid_retriever = (
        RunnableParallel({
            "chroma_documents": chroma_retriever,
            "bm25_documents": bm25_retriever
        })
        | (lambda x: [x["chroma_documents"], x["bm25_documents"]])
        | reciprocal_rank_fusion
    )


    prompt = ChatPromptTemplate.from_template('''
以下の文脈だけを踏まえて質問に回答してください。
                                            
文脈:"""
{context}                                       
"""
                                            
質問: {question}
''')

    hybrid_rag_chain = (
        {"context": hybrid_retriever, "question": RunnablePassthrough()}
        | prompt
        | chat_model
        | StrOutputParser()
    )

    output = hybrid_rag_chain.invoke("LangChainの概要を教えてください。")
    print(output)


if __name__=="__main__":
    print("\n---------------------------------------------------------------------\n")
    main()
    print("\n---------------------------------------------------------------------\n")


