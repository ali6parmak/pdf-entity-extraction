from time import time

from flair.nn import Classifier
from pdf_features.Rectangle import Rectangle

from data_model.EntityBox import EntityBox
from data_model.WordBox import WordBox
from methods.NERTransformerModel import NERTransformerModel
from flair.data import Sentence
from flair.models import SequenceTagger


def print_with_line_breaks(text, line_length=150):
    for i in range(0, len(text), line_length):
        print(text[i : i + line_length])


class FlairNERDateExtractor(NERTransformerModel):
    def __init__(self, model_name: str = "flair_ner_english", show_logs: bool = False, initialize_auto_model=True):
        super().__init__(model_name + "_dates", show_logs, False)
        classifier_name_by_model_name = {
            "flair_ner_english": "ner",
            "flair_ner_english_fast": "ner-fast",
            "flair_ner_multilingual_large": "ner-large",
            "flair_ner_spanish_large": "es-ner-large",
            "flair_ner_ontonotes_multilingual_large": "ner-ontonotes-large",
        }
        self.classifier = Classifier.load(classifier_name_by_model_name[model_name])

    def process_segment(self, pdf_words, segment_box, total_entity_count) -> list[EntityBox]:
        segment_bounding_box = Rectangle.from_width_height(
            segment_box["left"], segment_box["top"], segment_box["width"], segment_box["height"]
        )
        word_boxes_for_page = pdf_words.pdf_words[segment_box["page_number"] - 1]
        word_boxes_in_segment: list[WordBox] = WordBox.find_word_boxes_in_rectangle(
            segment_bounding_box, word_boxes_for_page
        )

        sentence = Sentence(" ".join([wb.text for wb in word_boxes_in_segment]))
        self.classifier.predict(sentence)
        entities = sentence.get_spans("ner")
        aggregated_entities = []

        for entity in entities:
            if entity.tag != "DATE":
                continue
            aggregated_entities.append(
                {
                    "text": entity.text,
                    "entity_label": entity.tag,
                    "start_index": entity.start_position,
                    "end_index": entity.end_position,
                }
            )

        # print("PAGE: ", segment_box["page_number"])
        # print_with_line_breaks(" ".join([wb.text for wb in word_boxes_in_segment]))
        # print("-" * 30)
        # print("\n".join([str(e) for e in aggregated_entities]))
        # print("*" * 30)

        total_entity_count += len(aggregated_entities)
        return self.create_entity_boxes(aggregated_entities, segment_box, word_boxes_in_segment)


if __name__ == "__main__":
    model = FlairNERDateExtractor("flair_ner_ontonotes_multilingual_large")
    start = time()
    model.get_entities("cejil_staging41", save_output=True)
    print("results finished in", round(time() - start, 2), "seconds")
