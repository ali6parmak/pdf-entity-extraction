import json
from pathlib import Path

from pdf_features.Rectangle import Rectangle

from configuration import PDFS_PATH, CACHED_DATA_PATH
from data_model.PdfWords import PdfWords
from data_model.WordBox import WordBox


def fix_word_boxes(file_name: str):
    pdf_words: PdfWords = PdfWords.from_pdf_path(Path(PDFS_PATH, f"{file_name}.pdf"))
    segment_boxes: list[dict] = json.loads(Path(CACHED_DATA_PATH, f"{file_name}.json").read_text())
    segment_box = segment_boxes[3]
    segment_bounding_box = Rectangle.from_width_height(
        segment_box["left"], segment_box["top"], segment_box["width"], segment_box["height"]
    )
    word_boxes_for_page = pdf_words.pdf_words[segment_box["page_number"] - 1]
    word_boxes_in_segment: list[WordBox] = WordBox.find_word_boxes_in_rectangle(segment_bounding_box, word_boxes_for_page)

    print("WORD BOXES IN SEGMENT: \n", [wb.text for wb in word_boxes_in_segment])
    word_boxes_full_text = " ".join([wb.text for wb in word_boxes_in_segment])
    print("FULL TEXT WITH WORD BOXES: \n", word_boxes_full_text)
    segment_box_text = " ".join(segment_box["text"].split())
    print("SEGMENT BOX TEXT: \n", segment_box_text)
    print("WORDBOX TEXT EQUALS TO SEGMENT BOX TEXT: ", word_boxes_full_text == segment_box_text)


if __name__ == "__main__":
    fix_word_boxes("cejil_staging33")
