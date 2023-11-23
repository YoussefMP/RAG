from transformers import AutoTokenizer, AutoModel
from arabert.preprocess import ArabertPreprocessor
from Source.Logging.loggers import get_logger
from embedding import process_batch
from Source.paths import *
import json

__DEBUG__ = True

logger = get_logger("indexer", "indexing")


def load_data():
    folders = list(os.listdir(jotr_documents_path))
    folders.remove("Archive")

    if __DEBUG__:
        folders = ["قانون"]

    for folder in folders:
        folder_path = os.path.join(jotr_documents_path, folder)

        for file_name in os.listdir(folder_path):
            logger.info(f"\tLoading the data from {folder} for the year {file_name.replace('.json', '')}")
            file_path = os.path.join(folder_path, file_name)

            json_file = open(file_path, "r", encoding="utf-8")
            data = json.load(json_file)

            return data


def main():

    logger.info("Loading the data from the csv files ...")
    data = load_data()

    logger.info("Loading preprocessing model...")
    model_name = "aubmindlab/bert-large-arabertv2"
    arabert_prep = ArabertPreprocessor(model_name=model_name)

    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.bos_token = "[CLS]"
    tokenizer.eos_token = "[SEP]"

    logger.info("Loading embedding model...")
    model = AutoModel.from_pretrained(model_name)

    logger.info("Start embedding ...")
    metadata_map, vec_batch = process_batch(arabert_prep, tokenizer, model, data)


if __name__ == "__main__":
    main()





# from Source.Indexing.Database_Manager import DBManager
# from sentence_transformers import SentenceTransformer
# from Source.Logging.loggers import get_logger
# from Source.Indexing.docs_processor import *
# from Source.paths import *
# import tqdm
# import csv
#
# __DEBUG__ = True
#
#
# CONNECTION_STRING = "postgresql+psycopg2://postgres:test@localhost:5432/vector_db"
# COLLECTION_NAME = "state_of_union_vectors"
#
# logger = get_logger("Embeddings", "embeddings_generator.log")
#
#
# def generate_documents(base_path):
#
#     folders_list = os.listdir(base_path) if __DEBUG__ else ["2023"]
#
#     for year_folder in folders_list:
#
#         data = {}
#         year_files_path = os.path.join(base_path, year_folder)
#
#         for file_name in os.listdir(year_files_path):
#
#             csv_path = os.path.join(year_files_path, file_name)
#             with open(csv_path, newline='', encoding="utf-8") as csv_file:
#                 data[(year_folder, file_name)] = list(csv.reader(csv_file))
#             csv_file.close()
#
#         yield data
#
#
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
#     model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
#
#     path = os.path.join(jotr_documents_path, "Loi_csv")
#     logger.info(f"Setting the path of the csv files to {path}")
#     logger.info("Loading the csv documents...")
#     docs_generator = generate_documents(path)
#
#     for year_data in docs_generator:
#         embeddings, meta_data = process_documents(model, year_data)
#         db_manager.upsert_data()

