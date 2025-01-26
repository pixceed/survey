import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

import pymupdf4llm

# .envファイルから環境変数を読み込み
load_dotenv()


def main():

    input_path = "input/140120241007594481.pdf"

    output_dir = "survey/01_pdf_read/07_pymupdf4llm"

    #####################
    # PDFの読み込み
    #####################

    pdf_text = pymupdf4llm.to_markdown(input_path)

    print(pdf_text)

    pdf2text_file = os.path.join(output_dir, "pdf2text.txt")
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