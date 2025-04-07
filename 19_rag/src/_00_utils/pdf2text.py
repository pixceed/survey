import base64
import os
import re
from datetime import datetime

from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_community.callbacks.manager import get_openai_callback
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from src._00_utils._01_pdf2image import pdf2image, pdf2image_crop
from src._00_utils._02_ocr import image2text_ymtk
from src._00_utils._03_figure2text import figure2text

# .envファイルから環境変数を読み込み
load_dotenv()


def pdf2text(pdf_path, output_dir, crop_info=None):

    #####################
    # PDF2TEXT
    #####################

    # ＜PDFの各ページを高画質画像で保存する＞
    if crop_info is not None:
        
        image_path_list = pdf2image_crop(
            pdf_path=pdf_path, 
            output_dir=output_dir,
            crop_info=crop_info)
    else:
        image_path_list = pdf2image(
            pdf_path=pdf_path, 
            output_dir=output_dir)
    
    # ＜画像をテキスト化する＞
    all_text = ""
    all_figure_path_list = []

    for image_path in image_path_list:
        image_text, figure_path_list = image2text_ymtk(image_path=image_path, output_dir=output_dir)

        all_text += image_text + "\n"
        all_figure_path_list.extend(figure_path_list)

    output_path = os.path.join(output_dir, f"pdf2text.md")
    with open(output_path, mode='w') as f:
        f.write(all_text)
    
    return all_text, all_figure_path_list
    

def pdf2text_figure2text(pdf_path, output_dir, chat_model, crop_info=None):

    #####################
    # PDF2TEXT
    #####################

    # ＜PDFの各ページを高画質画像で保存する＞
    if crop_info is not None:
        
        image_path_list = pdf2image_crop(
            pdf_path=pdf_path, 
            output_dir=output_dir,
            crop_info=crop_info)
    else:
        image_path_list = pdf2image(
            pdf_path=pdf_path, 
            output_dir=output_dir)
    
    # ＜画像をテキスト化する＞
    all_text = ""
    all_figure_path_list = []

    for image_path in image_path_list:
        image_text, figure_path_list = image2text_ymtk(image_path=image_path, output_dir=output_dir)

        all_text += image_text + "\n"
        all_figure_path_list.extend(figure_path_list)

    output_path = os.path.join(output_dir, f"pdf2text.md")

    with open(output_path, mode='w') as f:
        f.write(all_text)
    
    # ＜図をテキスト化して埋め込み＞
    for figure_file_path in all_figure_path_list:

        # 図のテキスト説明を取得
        figure_text = figure2text(figure_file_path=figure_file_path, chat_model=chat_model)
        figure_text = f"\n<画像の説明>\n{figure_text}\n</画像の説明>\n"

        # 置き換え
        figure_name = os.path.basename(figure_file_path)
        escaped_figure_name = re.escape(figure_name)
        pattern = rf'<img\s+src="figures/{escaped_figure_name}"[^>]*>'
        all_text = re.sub(pattern, figure_text, all_text)

    output_path = os.path.join(output_dir, f"pdf2text_full.md")
    with open(output_path, mode='w') as f:
        f.write(all_text)

    return all_text


def pdf2text_llm(pdf_path, output_dir, chat_model, crop_info=None):

    #####################
    # PDF2TEXT
    #####################

    # ＜PDFの各ページを高画質画像で保存する＞
    if crop_info is not None:
        
        image_path_list = pdf2image_crop(
            pdf_path=pdf_path, 
            output_dir=output_dir,
            crop_info=crop_info)
    else:
        image_path_list = pdf2image(
            pdf_path=pdf_path, 
            output_dir=output_dir)
    

    # ＜画像をテキスト化する＞
    all_text = ""
    all_figure_path_list = []

    for image_path in image_path_list:
        image_text, figure_path_list = image2text_ymtk(image_path=image_path, output_dir=output_dir)

        system_prompt0 = SystemMessage(
            content=\
f"""
あなたは天才的な文書作成者です。

<ベーステキスト>を元に、画像から文章を読み取り、テキスト形式にまとめてください。

<ベーステキスト>
{image_text}
</ベーステキスト>

[指示]
・出力は、必ずテキスト文章のみで、余計な文章は含めないでください。
"""
    )
        # 画像をbase64形式のデータに変換
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read())
            image_data = encoded_string.decode('utf-8')

        image_message = HumanMessage(
            content=[
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_data}"},
                },
            ],
        )
        messages = [system_prompt0, image_message]
        result = chat_model.invoke(messages)

        result_text = result.content
        result_text = result_text.replace("```plaintext", "").replace("```", "")
        
        all_text += result_text + "\n"
        # all_figure_path_list.extend(figure_path_list)

    output_path = os.path.join(output_dir, f"pdf2text.md")

    with open(output_path, mode='w') as f:
        f.write(all_text)






if __name__=="__main__":

    # pdf_path = "input/V_Rohto_Premium_Product_Information.pdf"
    pdf_path = "input/pages_7_to_11.pdf"
    # pdf_path = "input/2024r06h_ap_pm_ans.pdf"
    output_dir = "output"
    crop_info = [0, 140, 0, 0]

    #####################
    # LLMセットアップ
    #####################
    chat_model = ChatOpenAI(model="gpt-4o", temperature=0)


    # ＜入力ファイル名からタイムスタンプ付きディレクトリの作成＞
    file_base_name = os.path.splitext(os.path.basename(pdf_path))[0]
    file_name = f"{file_base_name}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
    output_dir = os.path.join(output_dir, file_name)
    os.makedirs(output_dir, exist_ok=True)

    # pdf2text(pdf_path=pdf_path, output_dir=output_dir, crop_info=crop_info)
    pdf2text_llm(
        pdf_path=pdf_path, 
        output_dir=output_dir, 
        chat_model=chat_model,
        crop_info=crop_info)