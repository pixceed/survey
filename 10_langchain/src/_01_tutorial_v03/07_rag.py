import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, AzureChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import GitLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


# .envファイルから環境変数を読み込み
load_dotenv()

def main():

    # ChatModelの準備
    chat_model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    # chat_model = AzureChatOpenAI(
    #     openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    #     azure_deployment=os.getenv("AZURE_CHAT_DEPLOYMENT"),
    #     temperature=0
    # )

    def file_filter(file_path: str) -> bool:
        return file_path.endswith(".mdx")
    
    loader = GitLoader(
        clone_url="https://github.com/langchain-ai/langchain",
        repo_path="./langchain",
        branch="master",
        file_filter=file_filter
    )

    raw_docs = loader.load()

    print(len(raw_docs))

    print("\n---------------------------------------------------------------------\n")

    # TextSplitter
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)

    docs = text_splitter.split_documents(raw_docs)

    print(len(docs))

    print("\n---------------------------------------------------------------------\n")

    # VectorStoreの作成

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    db = Chroma.from_documents(docs, embeddings)

    retriever = db.as_retriever()

    query = "AWSのS3からデータを読み込むためのDocument loaderはありますか?"

    context_docs = retriever.invoke(query)

    print(f"len = {len(context_docs)}")

    first_doc = context_docs[0]
    print(f"metadata = {first_doc.metadata}")
    print(first_doc.page_content)

    print("\n---------------------------------------------------------------------\n")

    prompt = ChatPromptTemplate.from_template('''
    以下の文脈だけを踏まえて質問に回答してください。
                                              
    文脈:"""
    {context}                                       
    """
                                              
    質問: {question}
''')
    
    chat_model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | chat_model
        | StrOutputParser()
    )

    output = chain.invoke(query)
    print(output)


if __name__=="__main__":
    print("\n---------------------------------------------------------------------\n")
    main()
    print("\n---------------------------------------------------------------------\n")


