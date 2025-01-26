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
from langchain_community.document_loaders import GitLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.documents import Document



# .envファイルから環境変数を読み込み
load_dotenv()

def main():

    current_file_path = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.join(current_file_path, "input")
    save_dir = os.path.join(current_file_path, "output")

    ##########################################
    # LLMと埋め込みモデルの準備
    ##########################################

    # 埋め込みモデルの準備
    embeddings = AzureOpenAIEmbeddings(
        openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        azure_deployment=os.getenv("AZURE_EMBEDDING_DEPLOYMENT"),
    )

    ##########################################
    # PDFの読み込み　VectorStoreの作成
    ##########################################

    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)

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

        add_docs = []
        for split_doc in split_docs:
            source_title = split_doc.metadata["source"]
            page_n = split_doc.metadata["page"]
            add_text = f"----------------- {source_title} {page_n}ページ目 -----------------\n"
            add_text += f"{split_doc.page_content}\n"

            add_doc = Document(
                page_content=add_text,
                metadata=split_doc.metadata
            )

            add_docs.append(add_doc)
            all_docs.append(add_doc)

        uuids = [str(uuid4()) for _ in range(len(add_docs))]
        db.add_documents(documents=add_docs, ids=uuids)


if __name__ == "__main__":
    main()