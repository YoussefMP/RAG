from Source.paths import *
import os


def extract_metadata(year, file_name):

    metadata = {}

    meta_file_name = file_name.replace(".csv", ".txt")
    meta_file_path = os.path.join(jotr_documents_path, "Loi", year, meta_file_name)

    with open(meta_file_path, "r", encoding="utf-8") as meta_file:
        for line in meta_file.readlines():
            line_data = line.split(":")
            metadata[line_data[0]] = line_data[1]
    meta_file.close()

    return metadata


def process_documents(model, data: dict):

    results = {}

    for key, sentences in data.items():
        embeddings = {}
        metadata = extract_metadata(key[0], key[1])



