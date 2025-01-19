import os
import pymupdf4llm


def main():

    file_path = "input/2408.14837.pdf"

    md_text = pymupdf4llm.to_markdown(file_path)

    print(md_text) 

if __name__=="__main__":
    main()