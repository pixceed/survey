import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, AzureChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import GitLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from langchain_community.retrievers import TavilySearchAPIRetriever
from langchain_core.documents import Document
from pydantic import BaseModel
from enum import Enum
from typing import Any


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


    # ＜RAGのチェーンの作成＞
    retriever = db.as_retriever()

    langchain_document_retriever = retriever.with_config(
        {"run_name": "langchain_document_retriever"}
    )

    # Web検索のRetiever
    web_retriever = TavilySearchAPIRetriever(k=3).with_config(
        {"run_name": "web_retriever"}
    )

    # ＜ルート分岐チェーンの作成＞
    # ユーザーの入力をもとにLLMがRetrieverを選択するChainを実装する

    class Route(str, Enum):
        langchain_document = "langchain_document"
        web = "web"
    
    class RouteOutput(BaseModel):
        route: Route

    route_prompt = ChatPromptTemplate.from_template("""\
質問に回答するための適切なRetrieverを選択してください。

質問: {question}
""")

    route_chain = (
        route_prompt
        | chat_model.with_structured_output(RouteOutput)
        | (lambda x: x.route)
    )

    def routed_retriever(inp: dict[str, Any]) -> list[Document]:
        question = inp["question"]
        route = inp["route"]

        if route == Route.langchain_document:
            return langchain_document_retriever.invoke(question)
        elif route == Route.web:
            return web_retriever.invoke(question)
        
        raise ValueError(f"Unknown retriever: {retriever}")

    prompt = ChatPromptTemplate.from_template('''
以下の文脈だけを踏まえて質問に回答してください。
                                            
文脈:"""
{context}                                       
"""
                                            
質問: {question}
''')

    route_rag_chain = (
        {"route": route_chain, "question": RunnablePassthrough()}
        | RunnablePassthrough.assign(context=routed_retriever)
        | prompt
        | chat_model
        | StrOutputParser()
    )

    # output = route_rag_chain.invoke("LangChainの概要を教えてください。")
    output = route_rag_chain.invoke("東京の今日の天気は？")
    print(output)


if __name__=="__main__":
    print("\n---------------------------------------------------------------------\n")
    main()
    print("\n---------------------------------------------------------------------\n")


