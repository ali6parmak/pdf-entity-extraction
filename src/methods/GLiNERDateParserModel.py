import json
from time import time

from gliner import GLiNER
from pdf_features.Rectangle import Rectangle
from data_model.EntityBox import EntityBox
from data_model.WordBox import WordBox
from methods.NERTransformerModel import NERTransformerModel
from dateparser.search import search_dates


def print_with_line_breaks(text, line_length=150):
    for i in range(0, len(text), line_length):
        print(text[i : i + line_length])


class GLiNERDateParserModel(NERTransformerModel):
    def __init__(self, model_name: str, show_logs: bool = False):
        super().__init__(model_name, show_logs, initialize_auto_model=False)
        self.model_name = model_name + "_date_parser_temp"
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

    @staticmethod
    def get_parseable_entities(entities: list) -> list:
        parseable_entities = []
        for entity in entities:
            if search_dates(entity["text"]):
                parseable_entities.append(entity)
        return parseable_entities

    @staticmethod
    def remove_found_dates_from_text(text: str, entities: list) -> str:
        for entity in entities:
            replacement = "X" * (entity["end_index"] - entity["start_index"])
            text = text[: entity["start_index"]] + replacement + text[entity["end_index"] :]
        return text

    def get_date_parser_predictions(self, text: str):
        date_parser_result = search_dates(text)
        if not date_parser_result:
            return []
        cleaned_results = []

        for date_text, date_time in date_parser_result:
            # if len(date_text) < 4:
            #     continue
            if not self.classifier.predict_entities(date_text, ["date"]):
                continue
            cleaned_results.append((date_text, date_time))

        occurrences = {}
        entities = []
        end_index = 0
        for date_text, date_time in cleaned_results:
            if date_text not in occurrences:
                occurrences[date_text] = 0
            occurrences[date_text] += 1
            start_index = text.find(date_text, end_index)
            end_index = start_index + len(date_text)

            entities.append({"text": date_text, "entity_label": "DATE", "start_index": start_index, "end_index": end_index})
        return entities

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
        aggregated_entities = self.get_parseable_entities(aggregated_entities)
        aggregated_entities = self.remove_overlapping_entities(aggregated_entities)

        # full_text = " ".join([wb.text for wb in word_boxes_in_segment])
        # cleaned_text = self.remove_found_dates_from_text(full_text, aggregated_entities)
        # date_parser_entities = self.get_date_parser_predictions(cleaned_text)
        # aggregated_entities.extend(date_parser_entities)

        # print("PAGE: ", segment_box["page_number"])
        # print_with_line_breaks(" ".join([wb.text for wb in word_boxes_in_segment]))
        # print("-" * 30)
        # print("\n".join([str(r) for r in aggregated_entities]))
        # print("*" * 30)

        total_entity_count += len(aggregated_entities)
        return self.create_entity_boxes(aggregated_entities, segment_box, word_boxes_in_segment)


if __name__ == "__main__":
    model = GLiNERDateParserModel("urchade/gliner_multi-v2.1")
    start = time()
    model.get_entities("cejil_staging41", save_output=True)
    print("results finished in", round(time() - start, 2), "seconds")
