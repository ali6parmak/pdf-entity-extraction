import spacy
from pathlib import Path
from pdf_features.Rectangle import Rectangle

from data_model.EntityBox import EntityBox
from data_model.WordBox import WordBox
from methods.NERTransformerModel import NERTransformerModel
from configuration import ROOT_PATH

# pip install spacy-transformers


def print_with_line_breaks(text, line_length=150):
    for i in range(0, len(text), line_length):
        print(text[i : i + line_length])


class OpennyaiLegalNERTRF(NERTransformerModel):
    def __init__(self, model_name: str = "opennyai_en_legal_ner_trf", show_logs: bool = False):
        super().__init__(model_name, show_logs, initialize_auto_model=False)
        self.model_name = model_name
        self.classifier = spacy.load(Path(ROOT_PATH, "opennyai_model", "en_legal_ner_trf", "en_legal_ner_trf-3.2.0"))
        self.show_logs = show_logs

    def process_segment(self, pdf_words, segment_box, total_entity_count) -> list[EntityBox]:
        segment_bounding_box = Rectangle.from_width_height(
            segment_box["left"], segment_box["top"], segment_box["width"], segment_box["height"]
        )
        word_boxes_for_page = pdf_words.pdf_words[segment_box["page_number"] - 1]
        word_boxes_in_segment: list[WordBox] = WordBox.find_word_boxes_in_rectangle(
            segment_bounding_box, word_boxes_for_page
        )

        text = " ".join([wb.text for wb in word_boxes_in_segment])
        print(f"PAGE: {segment_box['page_number']}")
        print_with_line_breaks(text)
        print("-" * 30)

        doc = self.classifier(text)
        aggregated_entities = []
        for entity in doc.ents:
            aggregated_entities.append(
                {
                    "text": entity.text,
                    "entity_label": entity.label_,
                    "start_index": entity.start_char,
                    "end_index": entity.end_char,
                }
            )
        total_entity_count += len(aggregated_entities)
        return self.create_entity_boxes(aggregated_entities, segment_box, word_boxes_in_segment)


if __name__ == "__main__":
    model = OpennyaiLegalNERTRF()
    model.get_entities("cejil_staging33", save_output=True)
