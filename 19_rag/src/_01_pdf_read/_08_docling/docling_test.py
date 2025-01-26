import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    EasyOcrOptions,
    TableFormerMode,
)
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
from docling.backend.docling_parse_v2_backend import DoclingParseV2DocumentBackend

# .envファイルから環境変数を読み込み
load_dotenv()


def main():

    input_path = "input/140120241007594481.pdf"

    output_dir = "survey/01_pdf_read/08_docling"

    #####################
    # PDFの読み込み
    #####################

    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = False
    pipeline_options.do_table_structure = True

    doc_converter = (
        DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=pipeline_options,
                    # PDFバックエンドの設定
                    backend=DoclingParseV2DocumentBackend,
                ),
            }
        )
    )

    result = doc_converter.convert(input_path)
    pdf_text = result.document.export_to_markdown()

    print(pdf_text)

    pdf2text_file = os.path.join(output_dir, "pdf2text.md")
    with open(pdf2text_file, "w", encoding='utf-8') as f:
        f.write(pdf_text)

    #####################
    # LLMセットアップ
    #####################
    chat_model = AzureChatOpenAI(
        openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        azure_deployment=os.getenv("AZURE_CHAT_DEPLOYMENT"),
        temperature=0
    )

    prompt = ChatPromptTemplate.from_template('''
    以下の文脈だけを踏まえて質問に回答してください。
    余計な文言は含めず、シンプルに回答してください。

    文脈:"""
    {context}
    """

    質問: {question}
    ''')

    chain = prompt | chat_model | StrOutputParser()

    llm_result_text = ""

    question = "累計経営成績の2024年3月中間期の売上高の増減率は？"
    output = chain.invoke({"question": question, "context": pdf_text})

    llm_result_text += f"\n質問: {question}\n"
    llm_result_text += f"回答: {output}\n"
    llm_result_text += f"(正解: 16.2%)\n\n"

    question = "今年度の売上高の見込みは？"
    output = chain.invoke({"question": question, "context": pdf_text})

    llm_result_text += f"\n質問: {question}\n"
    llm_result_text += f"回答: {output}\n"
    llm_result_text += f"(正解: 9,550百万円)\n\n"


    question = "前年度の純資産は？"
    output = chain.invoke({"question": question, "context": pdf_text})

    llm_result_text += f"\n質問: {question}\n"
    llm_result_text += f"回答: {output}\n"
    llm_result_text += f"(正解: 8,398百万円)\n\n"

    question = "ビジネスフィールド別売上高の表における、15.2が記載されているセルの下の数字は？"
    output = chain.invoke({"question": question, "context": pdf_text})

    llm_result_text += f"\n質問: {question}\n"
    llm_result_text += f"回答: {output}\n"
    llm_result_text += f"(正解: 14.1)\n\n"


    question = "発行済株式数の2025年の合計（期末発行済株式数と期末自己株式数と期中平均株式数の合計）は？"
    output = chain.invoke({"question": question, "context": pdf_text})

    llm_result_text += f"\n質問: {question}\n"
    llm_result_text += f"回答: {output}\n"
    llm_result_text += f"(正解: 10,238,164株)\n\n"


    llm_result_file = os.path.join(output_dir, "llm_result.txt")
    with open(llm_result_file, "w") as f:
        f.write(llm_result_text)

    print("#"*80)
    print(llm_result_text)

if __name__=="__main__":
    main()