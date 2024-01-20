from google.colab import drive
import json
import typer
import spacy
import re
from pathlib import Path

from spacy.tokens import Span, DocBin, Doc
from spacy.vocab import Vocab
from wasabi import Printer
from spacy.tokenizer import Tokenizer
from spacy.lang.en import English
from spacy.util import compile_infix_regex

# Intialize blank spacy pipeline
nlp = spacy.blank("en")

# Labels used for annotation
SYMM_LABELS = ["Binds"]
MAP_LABELS = {
    "INCLUDE_SERVICE": "INCLUDE_SERVICE",
    "HAS_FUNCTION": "HAS_FUNCTION",
    "ACCESS": "ACCESS",
    "USES": "USES"
}

msg = Printer()


def annotator(json_data: Path, output_file: Path):
    """Creating the corpus from annotations."""
    Doc.set_extension("rel", default={}, force=True)
    vocab = Vocab()

    docs = {"train": [], "dev": [], "test": [], "total": []}
    ids = {"train": set(), "dev": set(), "test": set(), "total": set()}
    count_all = {"train": 0, "dev": 0, "test": 0, "total": 0}
    count_pos = {"train": 0, "dev": 0, "test": 0, "total": 0}

    with open(json_data, encoding="utf8") as jsonfile:
        file = json.load(jsonfile)
        for document in file:
            span_starts = set()
            neg = 0
            pos = 0
            # Parse the tokens
            tokens = nlp(document["document"])

            spaces = []
            spaces = [True if tok.whitespace_ else False for tok in tokens]
            words = [t.text for t in tokens]
            doc = Doc(nlp.vocab, words=words, spaces=spaces)

            # Parse the GGP entities
            spans = document["tokens"]
            entities = []
            span_end_to_start = {}
            for span in spans:
                entity = doc.char_span(
                    span["start"], span["end"], label=span["entityLabel"]
                )

                span_end_to_start[span["token_start"]] = span["token_start"]
                entities.append(entity)
                span_starts.add(span["token_start"])

            doc.ents = entities

            # Parse the relations
            rels = {}
            for x1 in span_starts:
                for x2 in span_starts:
                    rels[(x1, x2)] = {}
                    # print(rels)
            relations = document["relations"]

            for relation in relations:
                # 'head' and 'child' annotations refer to the end token in the span
                start = span_end_to_start[relation["head"]]
                end = span_end_to_start[relation["child"]]
                label = relation["relationLabel"]

                print(rels[(start, end)])
                print(label)

                if label not in rels[(start, end)]:
                    rels[(start, end)][label] = 1.0
                    pos += 1

            # Zero-imputation (Reduce Bias)
            for x1 in span_starts:
                for x2 in span_starts:
                    for label in MAP_LABELS.values():
                        if label not in rels[(x1, x2)]:
                            neg += 1
                            rels[(x1, x2)][label] = 0.0

            doc._.rel = rels

            # only keeping documents with at least 1 positive case
            if pos > 0:
                docs["total"].append(doc)
                count_pos["total"] += pos
                count_all["total"] += pos + neg
    print(len(docs["total"]))
    output_path = output_file

    docbin = DocBin(docs=docs["total"], store_user_data=True)
    docbin.to_disk(output_path)

    msg.info(
        f"{len(docs['total'])} training sentences"
    )


# Mount Google Drive
drive.mount('/content/drive')

# Load json data from the file
data_file = "/content/drive/MyDrive/Final Year Prototype/Datasets/ER-Data-03-WR.json"

with open(data_file, 'r', encoding='utf-8') as file:
    data = json.load(file)

# File Paths for data loading
annotated_data = "/content/drive/MyDrive/Final Year Prototype/Datasets/ER-Data-03-WR.json"
output_file = '/content/drive/MyDrive/Final Year Prototype/ER-SPACY/relations_training.spacy'

annotator(annotated_data, output_file)
