'''
RAGASによるRAG評価
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

from ragas import evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.metrics import LLMContextRecall, Faithfulness, FactualCorrectness, SemanticSimilarity
from datasets import Dataset 


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


    # ＜質問・正解・コンテキスト・回答の準備＞
    # 質問
    question = [
        "後藤ひとりの得意な楽器は？",
        "後藤ひとりの妹の名前は？",
        "後藤ひとりが加入したバンド名は？",
        "ギターヒーローの正体は？",
        "喜多郁代の髪の色は？",
        "伊地知虹夏が通う学校の名前は？",
        "山田リョウの趣味は？",
        "廣井きくりが所属するバンド名は？",
        "ライブハウス「STARRY」の店長の名前は？",
        "ぼっちちゃんが文化祭で披露した演奏法は？",
    ]

    # 正解
    ground_truth = [
        "ギター",
        "後藤ふたり",
        "結束バンド",
        "後藤ひとり",
        "赤",
        "下北沢高校",
        "廃墟探索と古着屋巡り",
        "SICKHACK",
        "伊地知星歌",
        "ボトルネック奏法",
    ]

    # コンテキストと回答の生成
    contexts = []
    answer = []
    for q in question:
        # print("question:", q)
        response = rag_chain_with_source.invoke({"input": q})
        contexts.append([x.page_content for x in response["context"]])
        answer.append(response["answer"])

    # print("contexts:", contexts)
    # print("answer:", answer)

    # ＜質問・正解・コンテキスト・回答をデータセットにまとめる＞
    ds = Dataset.from_dict(
        {
            "question": question,
            "answer": answer,
            "contexts": contexts,
            "ground_truth": ground_truth,
        }
    )


    # ＜評価の実行＞

    # eval_result = evaluate(ds, [faithfulness, answer_relevancy, context_precision, context_recall])
    
    # print("====== 評価結果 ======")
    # # for k, v in eval_result.items():
    # #     print(f"{k}: {v}")

    # print(eval_result)
    # print()


    evaluator_llm = LangchainLLMWrapper(llm)
    evaluator_embeddings = LangchainEmbeddingsWrapper(embeddings)
    metrics = [
        LLMContextRecall(llm=evaluator_llm), 
        FactualCorrectness(llm=evaluator_llm), 
        Faithfulness(llm=evaluator_llm),
        SemanticSimilarity(embeddings=evaluator_embeddings)
    ]
    results = evaluate(dataset=ds, metrics=metrics)
    df = results.to_pandas()
    print(df.head())

    df.to_csv('output/result.csv')








if __name__=="__main__":
    main()