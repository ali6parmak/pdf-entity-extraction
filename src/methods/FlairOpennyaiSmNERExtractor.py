from time import time

import spacy
from flair.data import Sentence
from flair.nn import Classifier
from pdf_features.Rectangle import Rectangle

from data_model.EntityBox import EntityBox
from data_model.WordBox import WordBox
from methods.NERTransformerModel import NERTransformerModel


def print_with_line_breaks(text, line_length=150):
    for i in range(0, len(text), line_length):
        print(text[i : i + line_length])


class FlairOpennyaiReferenceExtractor(NERTransformerModel):
    def __init__(self, model_name: str = "flair_opennyai_extractor", show_logs: bool = False):
        super().__init__(model_name, show_logs, initialize_auto_model=False)
        self.model_name = model_name
        self.opennyai_model = spacy.load(
            "/home/ali/projects/pdf-entity-extraction/opennyai_model/en_legal_ner_sm/en_legal_ner_sm-3.2.0"
        )
        self.flair_model = Classifier.load("ner-ontonotes-large")
        self.show_logs = show_logs

    @staticmethod
    def remove_overlapping_entities(entities):
        sorted_entities = sorted(entities, key=lambda x: (x["start_index"], -len(x["text"])))

        result = []
        last_end = -1

        for entity in sorted_entities:
            if entity["start_index"] >= last_end:
                result.append(entity)
                last_end = entity["end_index"]

        return result

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

        opennyai_result = self.opennyai_model(text)
        aggregated_entities = []
        for entity in opennyai_result.ents:
            if entity.label_ not in {"PROVISION", "STATUTE", "CASE_NUMBER", "COURT"}:
                continue

            aggregated_entities.append(
                {
                    "text": entity.text,
                    "entity_label": entity.label_,
                    "start_index": entity.start_char,
                    "end_index": entity.end_char,
                }
            )

        sentence = Sentence(" ".join([wb.text for wb in word_boxes_in_segment]))
        self.flair_model.predict(sentence)
        flair_result = sentence.get_spans("ner")

        for entity in flair_result:
            if entity.tag not in {"LAW", "CARDINAL"}:
                continue
            aggregated_entities.append(
                {
                    "text": entity.text,
                    "entity_label": entity.tag,
                    "start_index": entity.start_position,
                    "end_index": entity.end_position,
                }
            )

        aggregated_entities = self.remove_overlapping_entities(aggregated_entities)

        total_entity_count += len(aggregated_entities)
        return self.create_entity_boxes(aggregated_entities, segment_box, word_boxes_in_segment)


if __name__ == "__main__":
    extractor = FlairOpennyaiReferenceExtractor()
    start = time()
    extractor.get_entities("cejil_staging33", save_output=True)
    print("Process finished in", round(time() - start, 2), "seconds")
