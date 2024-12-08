import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, AzureChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import GitLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from typing import Any
from langchain_cohere import CohereRerank
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

    # ＜リランクする関数＞
    def rerank(inp: dict[str, Any], top_n: int = 3) -> list[Document]:
        question = inp["question"]
        documents = inp["documents"]

        cohere_reranker = CohereRerank(model="rerank-multilingual-v3.0", top_n=top_n)

        return cohere_reranker.compress_documents(documents=documents, query=question)

    # ＜RAGのチェーンの作成＞
    retriever = db.as_retriever()

    prompt = ChatPromptTemplate.from_template('''
以下の文脈だけを踏まえて質問に回答してください。
                                            
文脈:"""
{context}                                       
"""
                                            
質問: {question}
''')
    

    # rag_chain = (
    #     {"documents": retriever, "question": RunnablePassthrough()}
    #     | RunnablePassthrough.assign(context=rerank)
    #     | prompt
    #     | chat_model
    #     | StrOutputParser()
    # )

    # output = rag_chain.invoke("LangChainの概要を教えてください。")
    # print(output)

    rag_chain = (
        {"documents": retriever, "question": RunnablePassthrough()}
        | RunnablePassthrough.assign(context=rerank)
        | RunnablePassthrough.assign(answer=(prompt | chat_model | StrOutputParser()))
    )

    output = rag_chain.invoke("LangChainの概要を教えてください。")
    print(output["answer"])



if __name__=="__main__":
    print("\n---------------------------------------------------------------------\n")
    main()
    print("\n---------------------------------------------------------------------\n")

