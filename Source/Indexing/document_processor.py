import torch.cuda
from transformers import AutoTokenizer, AutoModel
from arabert.preprocess import ArabertPreprocessor
from Source.Logging.loggers import get_logger
from Source.Database_API.db_operations import DbManager
from embedding import generate_embeddings, FOLDERS_IDS
from Source.paths import *
import json

__DEBUG__ = False

logger = get_logger("indexer", "indexing.log")


MODEL_NAME = "aubmindlab/bert-base-arabertv2"
DB_NAME = "Tunisian_Law_Database"
DB_HOST = "localhost"
DB_PORT = 5432


def load_data():
    folders = list(os.listdir(jotr_documents_path))
    folders.remove("Archive")

    if __DEBUG__:
        folders = ["قانون"]

    data = {}

    for folder in folders:
        folder_path = os.path.join(jotr_documents_path, folder)

        for file_name in os.listdir(folder_path):
            logger.info(f"\tLoading the data from {folder} for the year {file_name.replace('.json', '')}")
            file_path = os.path.join(folder_path, file_name)

            json_file = open(file_path, "r", encoding="utf-8")

            td = json.load(json_file)
            k = next(iter(td))
            try:
                data[f"{FOLDERS_IDS[folder]}-{file_name.replace('.json', '')}"] = td[k]
            except TypeError:
                print("wait")

            yield data


def load_database():

    credentials = open(os.path.join(config_folder_path, "db_credentials.json"), "r", encoding="utf-8")
    credentials = json.load(credentials)
    db = DbManager(DB_NAME, user=credentials["user"], password=["password"], host=DB_HOST, port=DB_PORT)
    return db


def load_models():

    logger.info("Loading preprocessing model...")
    arabert_prep = ArabertPreprocessor(model_name=MODEL_NAME)

    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.bos_token = "[CLS]"
    tokenizer.eos_token = "[SEP]"
    tokenizer.pad_token = "[PAD]"

    logger.info("Loading embedding model...")
    model = AutoModel.from_pretrained(MODEL_NAME)
    if torch.cuda.is_available():
        logger.info("\t moving the model to CUDA")
        model.to("cuda")

    return arabert_prep, tokenizer, model


def main():

    logger.info("Creating the Generator for the csv data files ...")
    data = load_data()

    arabert_prep, tokenizer, model = load_models()

    logger.info("Initializing database...")
    db = load_database()

    logger.info("Start embedding ...")
    metadata_map, vec_batch = generate_embeddings(arabert_prep, tokenizer, data, model, db)


if __name__ == "__main__":
    main()







# if __name__ == "__main__":
#
#     logger.info("Initiating Database...")
#     db_manager = DBManager(dbname="vector_db",
#                            user="postgres",
#                            password="test",
#                            host="localhost",
#                            port="5432"
#                            )
#

