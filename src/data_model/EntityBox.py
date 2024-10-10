from random import randint
from pdf_features.Rectangle import Rectangle
from data_model.WordBox import WordBox


class EntityBox:
    def __init__(self, word_boxes: list[WordBox], entity_label: str, label_rectangles: list[Rectangle]):
        self.word_boxes = word_boxes
        self.entity_label = entity_label
        self.label_rectangles = label_rectangles
        self.page_number = word_boxes[0].page_number
        self.page_width = word_boxes[0].page_width
        self.page_height = word_boxes[0].page_height

    def __str__(self):
        label_rectangle_str = ", ".join([str(label_rectangle.to_dict()) for label_rectangle in self.label_rectangles])
        return f'EntityBox(entity_label="{self.entity_label}", label_rectangles=[{label_rectangle_str}], WordBoxes({self.word_boxes}))'

    def __repr__(self):
        return self.__str__()

    @staticmethod
    def from_word_boxes(word_boxes: list[WordBox], entity_label: str):
        label_rectangles: list[Rectangle] = []
        bounding_boxes: list[Rectangle] = sorted([word_box.bounding_box for word_box in word_boxes], key=lambda x: x.top)
        same_line_rectangles: list[Rectangle] = [bounding_boxes[0]]
        for bounding_box in bounding_boxes[1:]:
            if bounding_box.top > same_line_rectangles[-1].bottom - same_line_rectangles[-1].height / 2:
                label_rectangles.append(Rectangle.merge_rectangles(same_line_rectangles))
                same_line_rectangles = [bounding_box]
            else:
                same_line_rectangles.append(bounding_box)
        if same_line_rectangles:
            label_rectangles.append(Rectangle.merge_rectangles(same_line_rectangles))

        return EntityBox(word_boxes, entity_label, label_rectangles)

    @staticmethod
    def get_rgb_by_entity_label(entity_labels: list[str], alpha=1):
        rgb_by_label: dict[str, tuple[float, float, float, float]] = {}
        for entity_label in entity_labels:
            r, g, b = randint(0, 255) / 255, randint(0, 255) / 255, randint(0, 255) / 255
            rgb_by_label[entity_label] = (r, g, b, alpha)
        return rgb_by_label
