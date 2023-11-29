from Source.paths import *
from tqdm import tqdm
import logging
import time
import torch

logger = logging.getLogger("Embedder")

CHUNK_SIZE = 510
OVERLAP_SIZE = 64

BATCH_SIZE = 65

# TODO Transfer to a separate file at some pooint maybe
FOLDERS_IDS = {"قانون": "0", "قرار": "1", "مرسوم": "2", "رأي": "3", "أمر": "4"}


def reduce_dimensions_method_one(vector, chunks):

    if chunks == 0:
        return vector.mean(dim=1)


def generate_batches(data, preprocessor, tokenizer):
    batch = {}
    for entries_list in data.values():

        for entry in entries_list:

            metadata = entry["metadata"]
            if metadata is None:
                continue

            text = entry["text"]
            preprocessed_text = preprocessor.preprocess(text)
            tokens = tokenizer.tokenize(preprocessed_text)
            chunks = [tokens[i:i + CHUNK_SIZE] for i in range(0, len(tokens), CHUNK_SIZE - OVERLAP_SIZE)]

            uid = (FOLDERS_IDS[metadata["type"]] + "-" +
                   (metadata["journal_nb"] + "-" +
                    metadata["date"].replace("/", "-") +
                    "-" + metadata["text_nb"]))

            for cid, chunk in enumerate(chunks):
                cuid = uid + "-" + str(cid)

                batch[cuid] = {"metadata": metadata, "tokens": chunk}

                if len(batch) == BATCH_SIZE:
                    yield batch
                    batch = {}

    if batch:
        yield batch


def generate_embeddings(preprocessor, tokenizer, data_gen, model, db_manager):

    data = next(data_gen)
    # embeddings_file = open(os.path.join(resources_folder_path, "results", "iort_embeddings."), "a")

    batch_gen = generate_batches(data, preprocessor, tokenizer)

    for batch in tqdm(batch_gen, total=5000, desc="Processing"):
        chunks = []
        metadata_list = []

        for cuid, value in batch.items():
            chunk = value["tokens"]
            token_ids = tokenizer.encode(" ".join(chunk),
                                         add_special_tokens=True,
                                         padding="max_length", truncation=True,
                                         max_length=CHUNK_SIZE+2,
                                         )
            chunks.append(torch.tensor(token_ids, dtype=torch.int32))
            metadata_list.append(value["metadata"])

        tensor_chunks = torch.stack(chunks).to("cuda")

        embeddings = model(tensor_chunks)
        stored_vector = embeddings.mean(dim=1)



        del embeddings
        torch.cuda.empty_cache()


