import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, AzureChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import GitLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from pydantic import BaseModel, Field

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

    # ＜複数の検索クエリを生成するチェーン

    class QueryGenerationOutput(BaseModel):
        queries: list[str] = Field(..., description="検索クエリのリスト")
    
    query_genaration_prompt = ChatPromptTemplate.from_template("""\
質問に対してベクターデータベースから関連文書を検索するために、
3つの異なる検索クエリを生成してください。
距離ベースの類似性検索の限界を克服するために、
ユーザーの質問に対して複数の視点を提供することが目的です。

質問:{question}                                                               
""")
    
    query_genaration_chain = (
        query_genaration_prompt
        | chat_model.with_structured_output(QueryGenerationOutput)
        | (lambda x: x.queries)
    )

    # ＜RAGのチェーンの作成＞
    retriever = db.as_retriever()

    prompt = ChatPromptTemplate.from_template('''
以下の文脈だけを踏まえて質問に回答してください。
                                            
文脈:"""
{context}                                       
"""
                                            
質問: {question}
''')
    
    def check_multi_query(multi_query):
        print("Multi Query:", multi_query)
        print(type(multi_query))
        print()
        return multi_query

    multi_query_rag_chain = (
        {
            "context": query_genaration_chain | check_multi_query | retriever.map(), 
            "question": RunnablePassthrough()
        }
        | prompt
        | chat_model
        | StrOutputParser()
    )

    output = multi_query_rag_chain.invoke("LangChainの概要を教えてください。")
    print(output)


if __name__=="__main__":
    print("\n---------------------------------------------------------------------\n")
    main()
    print("\n---------------------------------------------------------------------\n")


