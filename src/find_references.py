import json
from pathlib import Path
from ollama import Client
from configuration import CACHED_DATA_PATH


def print_with_line_breaks(text, line_length=150):
    for i in range(0, len(text), line_length):
        print(text[i : i + line_length])


def get_prompt(text: str):
    return f"""You are going to receive a piece of text from a document. Your task is to find the references in this text.
While finding these references, you should be following these instructions:

1. Extract the text as it is, do not add anything additional.
2. Only output what you find, do not add any comments, any characters, basically any other text.
3. If you do not find any references, output 'False'.  

Here is the text you are going to find the references in:

{text}
    """


def get_prompt_2(text: str):
    return f"""You are going to receive a piece of text from a document. Your task is to find the 'Articles' in this text.
While finding these Articles, you should be following these instructions:

1. Extract the text as it is, do not add anything additional.
2. Only output what you find, do not add any comments, any characters, basically any other text.
3. The text does not have to include Articles. There might not be any Articles. In that case, output 'False'. 

Here is the text you are going to find the references in:

{text}
    """


def check_model():

    segment_boxes: list[dict] = json.loads(Path(CACHED_DATA_PATH, "cejil_staging33.json").read_text())
    model_name = "llama3.2"
    client = Client(host=f"http://localhost:11434")

    for segment_box in segment_boxes:
        if segment_box["page_number"] != 2:
            continue

        print_with_line_breaks(segment_box["text"])
        print("-" * 30)
        # response = client.chat(model=model_name, messages=[{"role": "user", "content": get_prompt(segment_box["text"])}])
        response = client.chat(model=model_name, messages=[{"role": "user", "content": get_prompt_2(segment_box["text"])}])
        print(response["message"]["content"])
        print("*" * 30)


if __name__ == "__main__":
    check_model()
