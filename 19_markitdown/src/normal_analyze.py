import os
from markitdown import MarkItDown


def main():

    # file_path = "input/test_data.xlsx"
    file_path = "input/20241226_ゲームシーンAI_作業報告1_v4.pptx"

    markitdown = MarkItDown()

    result = markitdown.convert(file_path)
    print(result.text_content)

if __name__=="__main__":
    main()