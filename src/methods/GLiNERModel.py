import json

from gliner import GLiNER
from pdf_features.Rectangle import Rectangle
from data_model.EntityBox import EntityBox
from data_model.WordBox import WordBox
from methods.NERTransformerModel import NERTransformerModel


def print_with_line_breaks(text, line_length=150):
    for i in range(0, len(text), line_length):
        print(text[i : i + line_length])


class GLiNERModel(NERTransformerModel):
    def __init__(self, model_name: str, show_logs: bool = False):
        super().__init__(model_name, show_logs, initialize_auto_model=False)
        self.model_name = model_name
        self.classifier = GLiNER.from_pretrained(model_name)
        self.show_logs = show_logs

    @staticmethod
    def find_unique_dicts(dict_list: list[dict]) -> list[dict]:
        def dict_hash(d: dict) -> str:
            return json.dumps(d, sort_keys=True)

        unique_dicts = {dict_hash(d): d for d in dict_list}
        return list(unique_dicts.values())

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

        labels = ["date"]
        entities = []
        window_size = 20
        slide_size = 10
        last_slide_end_index = 0

        for i in range(0, len(word_boxes_in_segment), slide_size):
            window_word_boxes = word_boxes_in_segment[i : i + window_size]
            window_text = " ".join([wb.text for wb in window_word_boxes])
            window_entities = self.classifier.predict_entities(window_text, labels)

            for entity in window_entities:
                entity["start"] += last_slide_end_index
                entity["end"] += last_slide_end_index

            slide_window_boxes = word_boxes_in_segment[i : i + slide_size]
            slide_text = " ".join([wb.text for wb in slide_window_boxes])
            last_slide_end_index += len(slide_text) + 1

            entities.extend(window_entities)

        aggregated_entities = [
            {
                "text": entity["text"],
                "entity_label": entity["label"][: 3 if len(entity["label"]) > 4 else len(entity["label"])].upper(),
                "start_index": entity["start"],
                "end_index": entity["end"],
            }
            for entity in entities
        ]

        aggregated_entities = self.find_unique_dicts(aggregated_entities)
        aggregated_entities = self.remove_overlapping_entities(aggregated_entities)

        print("PAGE: ", segment_box["page_number"])
        print_with_line_breaks(" ".join([wb.text for wb in word_boxes_in_segment]))
        print("-" * 30)
        print("\n".join([str(r) for r in aggregated_entities]))
        print("*" * 30)

        total_entity_count += len(aggregated_entities)
        return self.create_entity_boxes(aggregated_entities, segment_box, word_boxes_in_segment)
