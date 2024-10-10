import json
from pathlib import Path
import spacy
from pdf_features.Rectangle import Rectangle
from spacy import displacy
from configuration import CACHED_DATA_PATH, PDFS_PATH
from data_model.EntityBox import EntityBox
from data_model.PdfWords import PdfWords
from data_model.WordBox import WordBox
from save_visualization_to_pdf import save_output_to_pdf


def check_en_core_web_sm():
    # python -m spacy download en_core_web_sm
    nlp = spacy.load("en_core_web_sm")
    text = (
        "When Sebastian Thrun started working on self-driving cars at Google in 2007, few people outside of the company took him seriously.\n\n"
        "When Sebastian Thrun started working on self-driving cars at Google in 2007, few people outside of the company took him seriously.\n\n"
        "When Sebastian Thrun started working on self-driving cars at Google in 2007, few people outside of the company took him seriously.\n\n"
    )
    doc = nlp(text)
    html = displacy.render(doc, style="ent", jupyter=False)
    Path("en_core_web_sm.html").write_text(html)
    for entity in doc.ents:
        print(entity.text, entity.label_, entity.label)


def find_entities(file_name: str):
    pdf_words: PdfWords = PdfWords.from_pdf_path(Path(PDFS_PATH, f"{file_name}.pdf"))
    segment_boxes: list[dict] = json.loads(Path(CACHED_DATA_PATH, f"{file_name}.json").read_text())
    nlp = spacy.load("en_core_web_sm")
    entity_boxes: list[EntityBox] = []
    total_entity_count = 0
    for segment_box in segment_boxes:
        segment_bounding_box = Rectangle.from_width_height(
            segment_box["left"], segment_box["top"], segment_box["width"], segment_box["height"]
        )
        word_boxes_for_page = pdf_words.pdf_words[segment_box["page_number"] - 1]
        word_boxes_in_segment: list[WordBox] = WordBox.find_word_boxes_in_rectangle(
            segment_bounding_box, word_boxes_for_page
        )
        doc = nlp(' '.join(segment_box["text"].split()))
        total_entity_count += len([ent for ent in doc.ents if ent.label_ in {"DATE", "GPE", "LAW", "ORG", "PERSON"}])
        for entity in doc.ents:
            if entity.label_ not in {"DATE", "GPE", "LAW", "ORG", "PERSON"}:
                continue
            word_boxes_for_entity: list[WordBox] = WordBox.find_word_boxes_from_text(word_boxes_in_segment, entity.start_char, entity.end_char)
            if not word_boxes_for_entity:
                print("NO WORD BOXES FOUND FOR ENTITY: ", entity.text, entity.label_, segment_box["page_number"], entity.start_char, entity.end_char)
                print("SEGMENT WORDS:", len(segment_box["text"].split()))
                print(segment_box["text"])
                print("WORD BOXES:", len(word_boxes_in_segment))
                print("\n")
                continue

    print("COUNTS")
    print("TOTAL ENTITY COUNT: ", total_entity_count)
    print("ENTITY BOX COUNT: ", len(entity_boxes))


    save_output_to_pdf(Path(PDFS_PATH, f"{file_name}.pdf"), segment_boxes, entity_boxes)


def print_all_entities():
    nlp = spacy.load("en_core_web_sm")
    entities = nlp.get_pipe("ner").labels

    print("Available entities:")
    for entity in entities:
        print(entity)


def check_sample():
    nlp = spacy.load("en_core_web_sm")
    text = "irreversible. 6  As  regards  the  divers  who  have  died,  AMHBLI  maintains  a  registry  with  400  victims.  In those cases of death in which compensation has been obtained, it has not been greater than US$ 2,000; in many cases the compensation payment is for US$ 500 or US$ 100.  They add that a large part of the accide nts occur “due to lack of supervision of the diving equipment, especially its quality and maintenance; there have been accidents caused by the use of scuba cylinders and due to obstruction of the system, which forces them to  come quickly to the surface.” P etition of November 5, 2004, pp. 4 and 5."
    text = " ".join(text.split())

    doc = nlp(text)
    for entity in doc.ents:
        # print(entity.sent)
        print(entity.text, entity.label_, entity.start_char, entity.end_char)


if __name__ == "__main__":
    find_entities("cejil_staging33")
