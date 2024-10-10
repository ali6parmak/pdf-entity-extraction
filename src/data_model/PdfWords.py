import subprocess
import tempfile
from os.path import join
from pathlib import Path
from lxml import etree
from lxml.etree import ElementBase
from pdf_features.PdfFeatures import PdfFeatures
from pdf_features.PdfPage import PdfPage
from data_model.WordBox import WordBox


class PdfWords:
    def __init__(self, pdf_features: PdfFeatures, pdf_words: list[list[WordBox]]):
        self.pdf_features: PdfFeatures = pdf_features
        self.pdf_words: list[list[WordBox]] = pdf_words

    @staticmethod
    def get_pdf_words(pdf_path, pdf_pages: list[PdfPage]):
        xml_path = join(tempfile.gettempdir(), "pdf_etree.xml")
        subprocess.run(["pdftotext", "-bbox-layout", pdf_path, xml_path])
        file_content: str = open(xml_path, errors="ignore").read()
        file_bytes: bytes = file_content.encode("utf-8")
        parser = etree.XMLParser(recover=True, encoding="utf-8")
        root: ElementBase = etree.fromstring(file_bytes, parser=parser)
        page_number = 0
        pdf_words: list[list[WordBox]] = []
        page_words: list[WordBox] = []
        for element in root.iter():
            if "page" in element.tag:
                page_number += 1
                if page_words:
                    pdf_words.append(page_words)
                page_words = []
            elif "word" in element.tag and element.text and element.text.strip():
                page_width = pdf_pages[page_number - 1].page_width
                page_height = pdf_pages[page_number - 1].page_height
                page_words.append(WordBox.from_etree_element(element, page_number, page_width, page_height))
        if page_words:
            pdf_words.append(page_words)
        return pdf_words

    @staticmethod
    def from_pdf_path(pdf_path: str | Path, pdf_name: str = ""):

        pdf_features: PdfFeatures = PdfFeatures.from_pdf_path(pdf_path)
        pdf_words: list[list[WordBox]] = PdfWords.get_pdf_words(pdf_path, pdf_features.pages)

        if pdf_name:
            pdf_features.file_name = pdf_name
        else:
            pdf_name = Path(pdf_path).parent.name if Path(pdf_path).name == "document.pdf" else Path(pdf_path).stem
            pdf_features.file_name = pdf_name
        return PdfWords(pdf_features, pdf_words)
