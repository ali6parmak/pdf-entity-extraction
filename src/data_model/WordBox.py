from lxml.etree import ElementBase
from pdf_features.Rectangle import Rectangle


class WordBox:
    def __init__(self, text: str, bounding_box: Rectangle, page_number: int, page_width: int, page_height: int):
        self.text = text
        self.bounding_box = bounding_box
        self.page_number = page_number
        self.page_width = page_width
        self.page_height = page_height

    def __str__(self):
        return f'WordBox(text="{self.text}", Rectangle({self.bounding_box.to_dict()}), page_number={self.page_number})'

    def __repr__(self):
        return self.__str__()

    @staticmethod
    def from_etree_element(element: ElementBase, page_number: int, page_width: int, page_height: int):
        left = int(float(element.attrib["xMin"]))
        top = int(float(element.attrib["yMin"]))
        right = int(float(element.attrib["xMax"]))
        bottom = int(float(element.attrib["yMax"]))
        bounding_box: Rectangle = Rectangle(left, top, right, bottom)
        text = element.text
        return WordBox(text, bounding_box, page_number, page_width, page_height)

    @staticmethod
    def find_word_boxes_in_rectangle(rectangle: Rectangle, word_boxes: list["WordBox"]):
        word_boxes_in_rectangle: list[WordBox] = []
        for word_box in word_boxes:
            if word_box.bounding_box.top > rectangle.bottom:
                continue
            if word_box.bounding_box.bottom < rectangle.top:
                continue
            if word_box.bounding_box.left > rectangle.right:
                continue
            if word_box.bounding_box.right < rectangle.left:
                continue
            if word_box.bounding_box.get_intersection_percentage(rectangle) > 50:
                word_boxes_in_rectangle.append(word_box)
        return word_boxes_in_rectangle

    @staticmethod
    def find_word_boxes_from_indices(word_boxes: list["WordBox"], start_index: int, end_index: int) -> list["WordBox"]:
        current_index = 0
        found_word_boxes: list["WordBox"] = []

        for word_box in word_boxes:
            word_start = current_index
            word_end = current_index + len(word_box.text)

            if word_end <= start_index:
                current_index = word_end + 1  # +1 for space before next word
                continue

            if word_start >= end_index:
                break

            if start_index <= word_end and word_start < end_index:
                size_by_letter = word_box.bounding_box.width / len(word_box.text)

                selection_start = max(0, start_index - word_start)
                selection_end = min(len(word_box.text), end_index - word_start)

                new_text = word_box.text[selection_start:selection_end]
                new_left = word_box.bounding_box.left + size_by_letter * selection_start
                new_width = size_by_letter * len(new_text)

                new_bounding_box = Rectangle.from_width_height(
                    left=new_left,
                    top=word_box.bounding_box.top,
                    width=new_width,
                    height=word_box.bounding_box.height
                )

                new_word_box = WordBox(new_text, new_bounding_box, word_box.page_number, word_box.page_width,
                                       word_box.page_height)
                found_word_boxes.append(new_word_box)

                if word_end >= end_index:
                    break

            current_index = word_end + 1

        return found_word_boxes
