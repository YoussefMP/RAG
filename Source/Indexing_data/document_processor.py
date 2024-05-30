import torch.cuda
from transformers import AutoTokenizer, AutoModel
from arabert.preprocess import ArabertPreprocessor
from Source.Logging.loggers import get_logger
from Source.Database_Client.db_operations import DbLocalManager, DBPineconeManager
from Source.Indexing_data.embedding import generate_batch_embeddings, FOLDERS_IDS
from Source.paths import *
from tqdm import tqdm
import json

__DEBUG__ = False

logger = get_logger("indexer", "indexing.log")

# Model Name
MODEL_NAME = "aubmindlab/bert-base-arabertv2"

# model's embeddings dimensions
DIMENSIONS = {"aubmindlab/bert-base-arabertv2": 768, }

# Database Variables
# DB_HOST = "pinecone"
# DB_HOST = "localhost"
DB_HOST = "Neo4J"

DB_NAME = "Texts_Metadata_DB"
DB_PORT = 5432


def load_data():
    folders = list(os.listdir(jotr_documents_path))
    folders.remove("Archive")

    if __DEBUG__:
        folders.remove("أمر")

    data = {}
    for folder in folders:
        folder_path = os.path.join(jotr_documents_path, folder)

        for file_name in os.listdir(folder_path):
            data = {}
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

    if "local" in DB_HOST:
        credentials = open(os.path.join(config_folder_path, "db_credentials.json"), "r", encoding="utf-8")
        credentials = json.load(credentials)
        db = DbLocalManager(DB_NAME, user=credentials["user"], password=credentials["password"], host=DB_HOST, port=DB_PORT)
        return db

    elif "pinecone" in DB_HOST:

        metadata_config = {"indexed": ["text_year", "type"]} if "arabertv2" in MODEL_NAME else None

        credentials = open(os.path.join(config_folder_path, "pinecone_api.json"), "r", encoding="utf-8")
        credentials = json.load(credentials)

        db = DBPineconeManager(DB_NAME,
                               credentials["environment"], credentials["api_key"],
                               DIMENSIONS[MODEL_NAME], metadata_config)

        db.create_index()
        return db


def load_models():
    """
        This method load the models needed for the embedding of the texts as well as all the necessary pre-processing steps
    """

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
    batch = generate_batch_embeddings(arabert_prep, tokenizer, data, model)

    for next_batch in tqdm(batch, total=2800, desc="Processing"):
        upsert_response = db.upsert_batch_with_metadata(next_batch)
        logger.debug(f"\t\tResponse: {upsert_response}")


if __name__ == "__main__":
    main()

