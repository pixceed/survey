import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, AzureChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import GitLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


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

    # ＜HyDEの仮説的な回答を生成するチェーン＞
    hypothentical_prompt = ChatPromptTemplate.from_template("""
次の質問に回答する一文を書いてください。
                                                            
質問: {question}
""")
    
    hypothentical_chain = hypothentical_prompt | chat_model | StrOutputParser()

    # ＜RAGのチェーンの作成＞
    retriever = db.as_retriever()

    prompt = ChatPromptTemplate.from_template('''
以下の文脈だけを踏まえて質問に回答してください。
                                            
文脈:"""
{context}                                       
"""
                                            
質問: {question}
''')
    
    def check_hyde(hyde_query):
        print("HyDE Query:", hyde_query)
        print()
        return hyde_query

    def check_retriever(context):
        print("Context:", context)
        print(type(context))
        print()
        return context

    hyde_rag_chain = (
        {
            "context": hypothentical_chain | check_hyde | retriever | check_retriever, 
            "question": RunnablePassthrough()
        }
        | prompt
        | chat_model
        | StrOutputParser()
    )

    output = hyde_rag_chain.invoke("LangChainの概要を教えてください。")
    print(output)


if __name__=="__main__":
    print("\n---------------------------------------------------------------------\n")
    main()
    print("\n---------------------------------------------------------------------\n")


