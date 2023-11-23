from transformers import AutoTokenizer, AutoModel
from arabert.preprocess import ArabertPreprocessor
from Source.paths import *
import json

__DEBUG__ = True


def load_data():
    folders = list(os.listdir(jotr_documents_path))
    folders.remove("Archive")

    if __DEBUG__:
        folders = ["قانون"]

    for folder in folders:
        folder_path = os.path.join(jotr_documents_path, folder)
        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)

            json_file = open(file_path, "r", encoding="utf-8")
            data = json.load(json_file)


def main():

    model_name = "aubmindlab/bert-base-arabertv2"
    arabert_prep = ArabertPreprocessor(model_name=model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)


if __name__ == "__main__":
    main()
