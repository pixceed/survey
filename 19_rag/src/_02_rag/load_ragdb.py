import os
from uuid import uuid4
from dotenv import load_dotenv

import pypdfium2 as pdfium
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import GitLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document

# .envファイルから環境変数を読み込み
load_dotenv()

def main():

    rag_dir = "output/kessan"

    ##########################################
    # LLMと埋め込みモデルの準備
    ##########################################

    # ChatModelの準備
    chat_model = AzureChatOpenAI(
        openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        azure_deployment=os.getenv("AZURE_CHAT_DEPLOYMENT"),
        temperature=0
    )
    # 埋め込みモデルの準備
    embeddings = AzureOpenAIEmbeddings(
        openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        azure_deployment=os.getenv("AZURE_EMBEDDING_DEPLOYMENT"),
    )

    #####################
    # VectorStoreの読み込み
    #####################
    # ＜作成済みのVectorStoreの読込＞
    db = Chroma(
        collection_name="langchain_store",
        embedding_function=embeddings,
        persist_directory=rag_dir,
    )

    # ＜RAGのチェーンの作成＞
    retriever = db.as_retriever(search_kwargs={"k": 20})

    prompt = ChatPromptTemplate.from_template('''
以下の文脈だけを踏まえて質問に回答してください。

文脈:"""
{context}
"""

質問: {question}
''')

    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | chat_model
        | StrOutputParser()
    )

    question = "2022年の売り上げは？"
    answer = rag_chain.invoke(question)

    print("-"*80)
    print(f"\n質問: {question}\n回答: {answer}\n")

    question = "過去５年間の売上高の変化は？"
    answer = rag_chain.invoke(question)

    print("-"*80)
    print(f"\n質問: {question}\n回答: {answer}\n")

if __name__=="__main__":
    main()