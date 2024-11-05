from typing import Iterator

from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document as LCDocument

from docling.document_converter import DocumentConverter

from langchain_text_splitters import RecursiveCharacterTextSplitter

# Doclingのカスタムローダー
class DoclingPDFLoader(BaseLoader):

    def __init__(self, file_path: str | list[str]) -> None:
        self._file_paths = file_path if isinstance(file_path, list) else [file_path]
        self._converter = DocumentConverter()

    def lazy_load(self) -> Iterator[LCDocument]:
        for source in self._file_paths:
            dl_doc = self._converter.convert(source).document
            text = dl_doc.export_to_markdown()
            yield LCDocument(page_content=text)

pdf_path = "input/1706.03762v7.pdf"

# DoclingでPDFをロード
loader = DoclingPDFLoader(file_path=pdf_path)
docs = loader.load()

for doc in docs:
    print(doc)
    print("-"*80)

# text_splitter = RecursiveCharacterTextSplitter(
#     chunk_size=1000,
#     chunk_overlap=200,
# )
# splits = text_splitter.split_documents(docs)