import os
import shutil
from uuid import uuid4
from typing import List
from dotenv import load_dotenv

import pypdfium2 as pdfium
from sudachipy import tokenizer
from sudachipy import dictionary

from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_core.tools import tool

from pydantic import BaseModel, Field


# .envファイルから環境変数を読み込み
load_dotenv()

def get_kessan_rag_retriever():

    current_file_path = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.join(current_file_path, "input")
    save_dir = os.path.join(current_file_path, "output")


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

    ##########################################
    # PDFの読み込み　VectorStoreの作成
    ##########################################

    # ＜VectorStoreの作成＞
    db = Chroma(
        collection_name="langchain_store",
        embedding_function=embeddings,
        persist_directory=save_dir,
    )

    # PDFファイルのパス
    files = os.listdir(input_dir)
    pdf_path_list = [os.path.join(input_dir, file) for file in files if file.endswith('.pdf')]

    # ドキュメントを読み込んで分割
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, # ドキュメントサイズ (トークン数)
        chunk_overlap=200, # 前後でオーバーラップするサイズ
        separators=["\n\n"] # セパレーター
    )

    all_docs = []
    for pdf_path in pdf_path_list:
        pdf = pdfium.PdfDocument(pdf_path)

        page_docs = []
        for page_num, page in enumerate(pdf):

            textpage = page.get_textpage()
            text = textpage.get_text_range()

            doc = Document(
                page_content=text,
                metadata={
                    "source": os.path.basename(pdf_path),
                    "page": page_num+1
                }
            )

            page_docs.append(doc)

        split_docs = text_splitter.split_documents(page_docs)

        for split_doc in split_docs:
            source_title = split_doc.metadata["source"]
            page_n = split_doc.metadata["page"]
            add_text = f"----------------- {source_title} {page_n}ページ目 -----------------\n"
            add_text += f"{split_doc.page_content}\n"

            add_doc = Document(
                page_content=add_text,
                metadata=split_doc.metadata
            )

            all_docs.append(add_doc)

    ##########################################
    # RAGシステムの構築
    ##########################################

    # ＜複数の検索クエリを生成するチェーン＞
    class QueryGenerationOutput(BaseModel):
        queries: list[str] = Field(..., description="検索クエリのリスト")

    query_genaration_prompt = ChatPromptTemplate.from_template("""\
ユーザーの質問に対して、ベクターデータベースから関連文書を検索するための
以下の5つの検索クエリを生成してください。

1. そのままの質問
2. ユーザーの質問を別の言い方で捉えたクエリ（同義語や言い換え）
3. 質問のトピックを上位概念・下位概念で捉えたクエリ
4. ユーザーの質問から外れないが、補足的なキーワードを使うクエリ
5. ユーザーの質問に対する想定される回答をし、その仮説的な回答を使うクエリ

質問: {question}
""")

    query_genaration_chain = (
        query_genaration_prompt
        | chat_model.with_structured_output(QueryGenerationOutput)
        | (lambda x: x.queries)
    )

    # ＜検索結果をRRFで並べるRAG-Fusion＞
    def reciprocal_rank_fusion(
        retriever_outputs: list[list[Document]],
        k: int = 60
    ) -> str:

        # 各ドキュメントのコンテンツ（文字列）とそのスコアの対応を保持する辞書を準備
        content_score_mapping = {}
        content_document_mapping = {}

        # 検索クエリごとにループ
        for docs in retriever_outputs:
            # 検索結果のドキュメントごとにループ
            for rank, doc in enumerate(docs):
                content = doc.page_content

                # 初めて登場したコンテンツの場合はスコアを0で初期化
                if content not in content_score_mapping:
                    content_score_mapping[content] = 0
                    content_document_mapping[content] = doc

                # (1 / (順位 + k))のスコアを加算
                content_score_mapping[content] += 1 / (rank + k)

        # スコアの大きい順にソート
        ranked = sorted(content_score_mapping.items(), key=lambda x:x[1], reverse=True)

        # 最終的な文脈テキスト
        context_text = ""
        for content, _ in ranked:
            doc = content_document_mapping[content]
            add_text = f"\n{doc.page_content}\n"

            context_text += add_text

        return context_text

    # ＜ベクトル検索のReriever＞
    retriever = db.as_retriever(search_kwargs={"k": 20})
    chroma_retriever = retriever.with_config(
        {"run_name": "chroma_retriever"}
    )

    # ＜BM25を使った検索のReriever＞
    # 単語単位のn-gramを作成
    def generate_word_ngrams(text, i, j, binary=False):
        tokenizer_obj = dictionary.Dictionary(dict="core").create()
        mode = tokenizer.Tokenizer.SplitMode.A
        tokens = tokenizer_obj.tokenize(text ,mode)
        words = [token.surface() for token in tokens]

        ngrams = []

        for n in range(i, j + 1):
            for k in range(len(words) - n + 1):
                ngram = tuple(words[k:k + n])
                ngrams.append(ngram)

        if binary:
            ngrams = list(set(ngrams))  # 重複を削除

        return ngrams

    def preprocess_word_func(text: str) -> List[str]:
        return generate_word_ngrams(text,1, 1, True)

    # 文字単位のn-gramを作成
    def generate_character_ngrams(text, i, j, binary=False):
        ngrams = []

        for n in range(i, j + 1):
            for k in range(len(text) - n + 1):
                ngram = text[k:k + n]
                ngrams.append(ngram)

        if binary:
            ngrams = list(set(ngrams))  # 重複を削除

        return ngrams

    def preprocess_char_func(text: str) -> List[str]:
        i, j = 1, 3
        if len(text) < i:
            return [text]
        return generate_character_ngrams(text, i, j, True)

    # BM25を使った検索のReriever
    word_retriever = BM25Retriever.from_documents(all_docs, preprocess_func=preprocess_word_func)
    char_retriever = BM25Retriever.from_documents(all_docs, preprocess_func=preprocess_char_func)
    word_retriever.k = 4
    char_retriever.k = 4
    bm25_retriever = EnsembleRetriever(
        retrievers=[word_retriever, char_retriever], weights=[0.7, 0.3]
    )
    bm25_retriever = bm25_retriever.with_config(
        {"run_name": "bm25_retriever"}
    )

    # ＜プロンプト＞
    prompt = ChatPromptTemplate.from_template('''
** Please be sure to answer in Japanese **
# Steps

1. Receive the question from the user.
2. Retrieve relevant context based on the question.
3. Generate a simple, concise answer using only the retrieved context.
4. Ensure the response does not include information beyond the provided context.
5. Answers must be faithful to the sentences in the context.
(When answering the amount, please check the context to see if it is tax-exclusive or tax-included.)

# Output Format

The output should be a single sentence or short paragraph that directly answers the question using only the given context.

# Examples

**Example 1:**
- **Input Question:** "What is the capital of France?"
- **Context:** "France is a country in Europe. The capital city is Paris."
- **Output Answer:** "The capital of France is Paris."

**Example 2:**
- **Input Question:** "Who is the author of '1984'?"
- **Context:** "'1984' is a novel written by George Orwell."
- **Output Answer:** "The author of '1984' is George Orwell."

(Real examples should include relevant complexity and contextual information retrieved from a source.)

# Notes

- Avoid providing any information not directly supported by the context.
- Ensure clarity and accuracy in the output.
- Handle ambiguous questions by stating that the context does not provide an answer.

# Context
{context}

# Question
{question}

'''
    )


    final_prompt = ChatPromptTemplate.from_template(
'''
あなたは専門家です。
以下に、文脈および質問に対する回答があります。
指示に従って、出力例に則った最終回答を作成してください。

# 文脈
{context}

# 質問
{question}

# 回答
{answer}

<出力例1(情報が見つかった場合)>
Vロートプレミアムは、第2類医薬品です。

情報元
・xxxx.pdf 1ページ目
</出力例1>

<出力例2(情報が見つかった場合)>
Vロートプレミアムは、第2類医薬品です。

情報元
・yyyy.pdf 10ページ目
・zzzz.pdf 4ページ目
</出力例2>

<出力例3(情報が見つからない場合)>
申し訳ありません。情報が見つかりませんでした。
</出力例3>

[指示]
・質問に対して回答の中から最も簡潔に重要な内容のみを抽出して、情報元を明示しつつ回答してください。
・情報元は、必ず抜けもれなくすべて記載してください。
・簡潔かつ直接的な回答を提供してください。過度な詳細や冗長な説明は避けてください。
・回答が、文脈に基づいているか確認してください。基づいていない場合は、'申し訳ありません。情報が見つかりませんでした。'と回答してください。
・最終的な回答は、日本語にしてください。
・有害で攻撃的な表現を使った回答は避けるようにしてください。
・質問文が何かわからないような回答は避けるようにしてください。
'''
    )

    hybrid_retriever = (
        query_genaration_chain
        | RunnableParallel({
            "chroma_documents": chroma_retriever.map(),
            "bm25_documents": bm25_retriever.map()
        })
        | (lambda x: [*x["chroma_documents"], *x["bm25_documents"]])
        | reciprocal_rank_fusion
    )


    # rag_fusion_chain = (
    #     {
    #         "context": hybrid_retriever,
    #         "question": RunnablePassthrough()
    #     }
    #     | prompt | chat_model | StrOutputParser()
    # )

    # rag_fusion_chain = (
    #     {
    #         "context": hybrid_retriever,
    #         "question": RunnablePassthrough()
    #     }
    #     | RunnablePassthrough.assign(
    #       answer=prompt | chat_model | StrOutputParser()
    #     )
    #     | final_prompt | chat_model | StrOutputParser()
    # )

    return hybrid_retriever


def get_kessan_rag_tool():

    kessan_rag_retriever = get_kessan_rag_retriever()

    @tool
    def kessan_rag_tool(input_text: str) -> str:
        """決算に関する情報を検索します。決算に関する質問がある場合は、このツールを使用する必要があります。"""
        print(f"★★★★★★★★★★★★★★ ツール実行 [クエリ: {input_text}] ★★★★★★★★★★★★★★★★★")
        return kessan_rag_retriever.invoke(input_text)

    return kessan_rag_tool


