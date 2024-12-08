import os
from dotenv import load_dotenv
from pprint import pprint

from langchain_openai import ChatOpenAI, AzureChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import GitLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from langchain_core.runnables import RunnableLambda
from typing import Iterator

from langchain_core.runnables import RunnableParallel

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


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

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant."),
            ("human", "{input}")
        ]
    )

    output_parser = StrOutputParser()

    """
    RunnableLambda
    任意の関数をRunnableにする
    """

    def upper(text: str) -> str:
        return text.upper()
    
    chain = prompt | chat_model | output_parser | RunnableLambda(upper)

    output = chain.invoke({"input": "Hello!"})
    print(output)

    # chainデコレーターを使う
    from langchain_core.runnables import chain
    @chain
    def upper(text: str) -> str:
        return text.upper()

    chain = prompt | chat_model | output_parser | upper

    output = chain.invoke({"input": "Hello!"})
    print(output)

    # 自動変換

    def upper(text: str) -> str:
        return text.upper()

    chain = prompt | chat_model | output_parser | upper

    output = chain.invoke({"input": "Hello!"})
    print(output)


    # streamに対応させる

    def upper(input_stream: Iterator[str]) -> Iterator[str]:
        for text in input_stream:
            yield text.upper()
    
    chain = prompt | chat_model | output_parser | upper

    for chunk in chain.stream({"input": "Hello!"}):
        print(chunk, end="", flush=True)


    print("\n---------------------------------------------------------------------\n")


    optimistic_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "あなたは楽観主義者です。ユーザーが入力に対して楽観的な意見をください。"),
            ("human", "{topic}")
        ]
    )
    optimistic_chain = optimistic_prompt | chat_model | output_parser

    pessimistic_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "あなたは悲観主義者です。ユーザーが入力に対して悲観的な意見をください。"),
            ("human", "{topic}")
        ]
    )
    pessimistic_chain = pessimistic_prompt | chat_model | output_parser

    parallel_chain = RunnableParallel(
        {
            "optimistic_opinion": optimistic_chain,
            "pessimistic_opinion": pessimistic_chain
        }
    )

    topic = "生成AIの進化について"
    output = parallel_chain.invoke({"topic": topic})
    print("\n楽観的な意見")
    pprint(output["optimistic_opinion"])
    print("\n悲観的な意見")
    pprint(output["pessimistic_opinion"])

    synthesize_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "あなたは客観的AIです。2つの意見をまとめてください。"),
            ("human", "楽観的意見: {optimistic_opinion} \n悲観的意見: {pessimistic_opinion}")
        ]
    )

    synthesize_chain = (
        RunnableParallel(
            {
                "optimistic_opinion": optimistic_chain,
                "pessimistic_opinion": pessimistic_chain
            }
        )
        | synthesize_prompt
        | chat_model
        | output_parser
    )
    output = synthesize_chain.invoke({"topic": topic})
    print("\n", output)

    # RunnableParallelが無くても自動変換される
    synthesize_chain = (
        {
            "optimistic_opinion": optimistic_chain,
            "pessimistic_opinion": pessimistic_chain
        }
        | synthesize_prompt
        | chat_model
        | output_parser
    )
    output = synthesize_chain.invoke({"topic": topic})
    print("\n", output)

    from operator import itemgetter
    synthesize_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "あなたは客観的AIです。{topic}についての2つの意見をまとめてください。"),
            ("human", "楽観的意見: {optimistic_opinion} \n悲観的意見: {pessimistic_opinion}")
        ]
    )
    synthesize_chain = (
        {
            "optimistic_opinion": optimistic_chain,
            "pessimistic_opinion": pessimistic_chain,
            "topic": itemgetter("topic")
        }
        | synthesize_prompt
        | chat_model
        | output_parser
    )
    output = synthesize_chain.invoke({"topic": topic})
    print("\n", output)

    print("\n---------------------------------------------------------------------\n")

    prompt = ChatPromptTemplate.from_template('''\
以下の文脈だけを踏まえて質問に回答してください。
                                                       
文脈: """
{context}
"""
                                                       
質問: {question}
''')
    
    loader = TextLoader("./sample_data/bocchi.md")
    documents = loader.load()

    # ドキュメントの分割
    documents = RecursiveCharacterTextSplitter(
        chunk_size=1000, # ドキュメントサイズ (トークン数)
        chunk_overlap=0, # 前後でオーバーラップするサイズ
        separators=["\n\n"] # セパレーター
    ).split_documents(documents)

    # ＜VectorStoreの準備＞
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_documents(
        documents,
        embedding=embeddings,
    )

    # ＜Retrieverの準備＞
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4},
    )

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | chat_model
        | StrOutputParser()
    )

    output = chain.invoke("結束バンドのメンバーは？")
    print(output)

    print("#####")

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | RunnablePassthrough.assign(
            answer=prompt | chat_model | StrOutputParser()
        )
    )

    output = chain.invoke("結束バンドのメンバーは？")
    print(output)

    print("#####")

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | RunnablePassthrough.assign(
            answer=prompt | chat_model | StrOutputParser()
        )
    )

    output = chain.invoke("結束バンドのメンバーは？")
    print(output)

    print("#####")

    chain = RunnableParallel(
        {
            "context": retriever, "question": RunnablePassthrough()
        }
    ).assign(answer=prompt | chat_model | StrOutputParser())

    output = chain.invoke("結束バンドのメンバーは？")
    print(output)

    print("#####")


if __name__=="__main__":
    print("\n---------------------------------------------------------------------\n")
    main()
    print("\n---------------------------------------------------------------------\n")

