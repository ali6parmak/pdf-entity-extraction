# Using spacy.load().
from pathlib import Path
import spacy
from huggingface_hub import snapshot_download

from configuration import ROOT_PATH


def download_model():
    snapshot_download(
        repo_id="opennyaiorg/en_legal_ner_sm",
        local_dir=Path(ROOT_PATH, "opennyai_model"),
        local_dir_use_symlinks=False,
    )


def run_model():
    nlp = spacy.load(Path(ROOT_PATH, "opennyai_model", "en_legal_ner_sm", "en_legal_ner_sm-3.2.0"))
    text = (
        "Articles 46 and 47 of the American Convention, as well as Articles 30 and 37 of its Rules of Procedure, "
        "the IACHR concludes that the petition is admissible in relation to the alleged violation of the rights established at "
        "Articles 4, 5, 8(1), 17(1), 19, 24, 25, and 26 of the American Convention, in relation to the general "
        "obligations enshrined in Articles 1(1) and 2 of the same international instrument."
    )
    doc = nlp(text)
    # Print indentified entites
    for ent in doc.ents:
        print(ent, ent.label_)


if __name__ == "__main__":
    # download_model()
    # run_model()
    print(ROOT_PATH)
