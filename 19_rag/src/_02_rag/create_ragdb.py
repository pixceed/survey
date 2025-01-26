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

    input_dir = "input/kessan"
    save_dir = "output/kessan"

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
    # PDFの読み込み
    ##########################################

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

    #####################
    # VectorStoreの作成
    #####################
    # https://python.langchain.com/docs/integrations/vectorstores/chroma/

    db = Chroma(
        collection_name="langchain_store",
        embedding_function=embeddings,
        persist_directory=save_dir,
    )
    uuids = [str(uuid4()) for _ in range(len(all_docs))]
    db.add_documents(documents=all_docs, ids=uuids)

    print(f"Chroma database is saved at: {save_dir}")







if __name__=="__main__":
    main()