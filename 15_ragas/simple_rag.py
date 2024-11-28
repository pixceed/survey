'''
検証用のRAG
'''

import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.callbacks.manager import get_openai_callback


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

    # ドキュメントの分割
    documents = RecursiveCharacterTextSplitter(
        chunk_size=1000, # ドキュメントサイズ (トークン数)
        chunk_overlap=0, # 前後でオーバーラップするサイズ
        separators=["\n\n"] # セパレーター
    ).split_documents(documents)
    print(f"Number of document chunks: {len(documents)}")

    # ＜VectorStoreの準備＞
    vectorstore = Chroma.from_documents(
        documents,
        embedding=embeddings,
    )

    # ＜Retrieverの準備＞
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 2},
    )

    # ＜PromptTemplateの準備＞

    prompt_template = ChatPromptTemplate.from_messages(
        [
            ("system", "次のコンテキストのみを使用して、この質問に答えてください。\n\n{context}"),
            ("human", "{input}"),
        ]
    )

    # # ＜RAGチェーンの準備＞
    # # 応答だけ返す
    # rag_chain = (
    #     {"context": retriever, "input": RunnablePassthrough()}
    #     | prompt_template
    #     | llm
    # )

    # # ＜質問してみる＞
    # with get_openai_callback() as cb:
    #     response = rag_chain.invoke("ギターヒーローの正体は？")

    #     print("------------------------------------------")
    #     print("回答: ", response.content)
    #     print("------------------------------------------")
    #     print(f"Total Tokens: {cb.total_tokens}")
    #     print(f"Prompt Tokens: {cb.prompt_tokens}")
    #     print(f"Completion Tokens: {cb.completion_tokens}")
    #     print(f"Total Cost (USD): ${cb.total_cost}")
    #     print("------------------------------------------")

    # ＜RAGチェーンの準備＞
    # ソース (応答に使用したコンテキスト) も返す

    # コンテキストのフォーマット
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    # Generationチェーンの準備
    gemeration_chain = (
        RunnablePassthrough.assign(context=(lambda x: format_docs(x["context"])))
        | prompt_template
        | llm
        | StrOutputParser()
    )

    # Retrieverチェーンの準備
    retrieve_chain = (lambda x: x["input"]) | retriever

    # ソース付きRAGチェーンの準備
    rag_chain_with_source = RunnablePassthrough.assign(context=retrieve_chain).assign(
        answer=gemeration_chain
    )

    # ＜質問してみる＞
    with get_openai_callback() as cb:
        response = rag_chain_with_source.invoke({"input": "ギターヒーローの正体は？"})

        print("------------------------------------------")
        print("回答: ", response["answer"])
        print("------------------------------------------")
        for i, doc in enumerate(response["context"]):
            print(f"======== {i+1} ========")
            print(doc.page_content)
            print(f"metadata: {doc.metadata}")
            print()

        print("------------------------------------------")
        print(f"Total Tokens: {cb.total_tokens}")
        print(f"Prompt Tokens: {cb.prompt_tokens}")
        print(f"Completion Tokens: {cb.completion_tokens}")
        print(f"Total Cost (USD): ${cb.total_cost}")
        print("------------------------------------------")






if __name__=="__main__":
    main()