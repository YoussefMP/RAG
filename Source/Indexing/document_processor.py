from transformers import AutoTokenizer, AutoModel
from arabert.preprocess import ArabertPreprocessor
from Source.Logging.loggers import get_logger
import json
from Source.paths import *
from tqdm.auto import tqdm
from utils import *
import openai

__DEBUG__ = False

logger = get_logger("indexer", "indexing")

CHUNK_SIZE = 1200
OVERLAP_SIZE = 0.15

EMBED_MODEL_ID = "text-embedding-ada-002"


# TODO Transfer to a separate file at some pooint maybe
FOLDERS_IDS = {"قانون": "0", "قرار": "1", "مرسوم": "2", "رأي": "3", "أمر": "4"}



def create_index():
    return None


def populate_index(data):
    batch_size = 100

    for i in tqdm(range(0, len(data), batch_size)):
        i_end = min(len(data), i + batch_size)
        batch = dict(list(data.items())[i:i_end])

        ids = batch.keys()
        texts = [data[k]["chunk"] for k in ids]

        res = openai.Embedding.create(input=texts, engine=EMBED_MODEL_ID)



        metadata = [data[k]["metadata"] for k in ids]
        to_upsert = list(zip(ids, texts, metadata))

        print("donme")


def overlapping_chunk(text, chunk_size, overlap_ratio=0.2):
    """
    Perform overlapping chunking on a given text.

    Parameters:
    - text (str): The input text to be chunked.
    - chunk_size (int): The desired size of each chunk.
    - overlap_ratio (float): The overlap ratio between consecutive chunks (0 to 1).

    Returns:
    - List of chunks (str).
    """

    # Calculate the overlap size based on the overlap ratio
    overlap_size = int(chunk_size * overlap_ratio)

    # Initialize variables
    chunks = []
    start_idx = 0
    end_idx = 0

    while end_idx < len(text):
        # Determine the end index of the current chunk
        end_idx = min(start_idx + chunk_size, len(text))

        # Append the chunk to the list
        chunks.append(text[start_idx:end_idx])

        # Move the start index forward with the overlap
        start_idx = end_idx - overlap_size

    return chunks


def load_data():
    uids = []
    data = {}

    folders = list(os.listdir(jotr_documents_path))
    folders.remove("Archive")

    if __DEBUG__:
        folders = ["قانون"]

    for folder in folders:
        folder_path = os.path.join(jotr_documents_path, folder)

        for file_name in os.listdir(folder_path):
            main_key = file_name[:4]
            logger.info(f"\tLoading the data from {folder} for the year {file_name.replace('.json', '')}")
            file_path = os.path.join(folder_path, file_name)

            json_file = open(file_path, "r", encoding="utf-8")
            data_obj = json.load(json_file)

            for entry in data_obj[main_key]:
                metadata = entry["metadata"]
                if metadata is None:
                    continue

                uid = (FOLDERS_IDS[folder] + "-" +
                       (metadata["journal_nb"] + "-" +
                       metadata["date"].replace("/", "-") +
                       "-" + metadata["text_nb"]))

                if uid in uids:
                    continue
                    # logger.debug(f"{metadata['text_title']}")
                    # logger.debug(f"{data[uid]['metadata']['text_title']}")
                    # logger.debug(f"________________________________________")
                uids.append(uid)

                text = entry["text"]
                chunks = overlapping_chunk(text, CHUNK_SIZE, OVERLAP_SIZE)

                for cid, chunk in enumerate(chunks):
                    cuid = uid + "-" + str(cid)
                    data[cuid] = {}
                    data[cuid]["metadata"] = metadata
                    data[cuid]["chunk"] = chunk

    return data


def main():

    logger.info("Loading the data from the csv files ...")
    data = load_data()

    logger.info("Loading openai API key")
    openai.api_key_path = os.path.join(source_folder_path, "0_conf", "openai_api.txt")

    logger.info("Loading Pinecone API Key")

    logger.info("Generating embedding for the chunks and inserting them into pinecone")
    create_index()
    populate_index(data)


if __name__ == "__main__":

    main()
