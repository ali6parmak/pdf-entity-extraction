from time import time
from dateparser.search import search_dates
from pdf_features.Rectangle import Rectangle
from data_model.EntityBox import EntityBox
from data_model.WordBox import WordBox
from methods.NERTransformerModel import NERTransformerModel


def print_with_line_breaks(text, line_length=150):
    for i in range(0, len(text), line_length):
        print(text[i : i + line_length])


class DateParser(NERTransformerModel):
    def __init__(self, model_name: str = "date_parser", show_logs: bool = False):
        super().__init__(model_name, show_logs, initialize_auto_model=False)

    @staticmethod
    def find_all_occurrences(main_text: str, date_parser_result: list):
        occurrences = {}
        entities = []
        end_index = 0
        for date_text, date_time in date_parser_result:
            if date_text not in occurrences:
                occurrences[date_text] = 0
            occurrences[date_text] += 1
            start_index = main_text.find(date_text, end_index)
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

        entities = []

        text = " ".join([wb.text for wb in word_boxes_in_segment])
        result = search_dates(text, languages=["es"])
        print(f"PAGE: {segment_box['page_number']}")
        print_with_line_breaks(text)
        print("-" * 30)
        if result:
            entities = self.find_all_occurrences(text, result)
            print("\n".join([str(date) for date in result]))
            # print("\n".join([date[0] for date in result]))
        else:
            print("NO DATE FOUND")
        print("*" * 30)

        total_entity_count += len(entities)
        return self.create_entity_boxes(entities, segment_box, word_boxes_in_segment)


if __name__ == "__main__":
    start = time()
    date_parser = DateParser("date_parser")
    date_parser.get_entities("cejil_staging41", save_output=True)
    print("date parse finished in", round(time() - start, 2), "seconds")
