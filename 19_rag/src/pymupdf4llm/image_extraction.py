import os
import pymupdf4llm


def main():

    file_path = "input/2408.14837.pdf"

    md_text_images = pymupdf4llm.to_markdown(
        doc=file_path,
        pages=[1, 11],
        page_chunks=True,
        write_images=True,
        image_path="output/images",
        image_format="png",
        dpi=300
    )

    print(md_text_images[0]['images']) 

if __name__=="__main__":
    main()