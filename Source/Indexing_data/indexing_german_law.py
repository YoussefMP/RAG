from Source.Logging.loggers import get_logger
from Source.Database_Client.Neo4J_Client import DBNeo4JManager
from sentence_transformers import SentenceTransformer
from Utils import paths
import os
import json

DEBUG = True

logger = get_logger("oldp_indexing_logger", "indexing_oldp.log")


def generate_batch_embeddings(model, data):
    return []


def index_data(model, db_manager, data):
    """
    This function is used to index the embeddings generated from the paths of the law texts to a Neo4J database.
    It also is used to create nodes and relationships in the database.
    :param model: the embedding model used to embed the data.
    :param db_manager: the database manager
    :param data: dict containing the law texts whenre the path of keys represents the book and the sections of the text
    :return:
    """
    logger.info("Start embedding...")
    batch = generate_batch_embeddings(model, data)

    for next_batch in batch:
        upsert_response = db_manager.upsert_batch_with_metadata(next_batch)
        logger.debug(f"\t\tResponse: {upsert_response}")


def load_model(url, tokenizer=None):
    if tokenizer:
        return None
    else:
        model = SentenceTransformer(url, trust_remote_code=True)
    return model


def load_database():
    """
    This methods will instantiate a new database if one doesn't exist.
    :return:
    """
    config_path = paths.config_folder_path + "/Neo4J_config.json"
    config_file = open(config_path, "r", encoding="utf-8")
    config = json.load(config_file)

    db_manager = DBNeo4JManager(**config)
    return db_manager


def load_data():
    files = list(os.listdir(paths.crawl_results_folder))

    if DEBUG:
        files = files[:1]

    for file in files:
        json_data = json.load(open(os.path.join(paths.crawl_results_folder, file), "r", encoding="utf-8"))

        yield json_data


def main():

    logger.info("Creating the Generator for the csv data files ...")
    data = load_data()

    # logger.info("Initializing database...")
    # db = load_database()
    #
    # model_url = "jinaai/jina-embeddings-v2-base-de"
    # model = load_model(model_url)
    #
    # index_data(model, db, data)

    # this loop iterated the data like a tree using postorder traversal


# Usage:
# for item in postorder_traversal(data):
#     print(item)


if __name__ == "__main__":
    main()

