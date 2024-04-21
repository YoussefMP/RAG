from Source.paths import *
from Source.Logging.loggers import get_logger
import logging
import torch
import gc

logger = get_logger("Embedder", "indexing.log")

CHUNK_SIZE = 510
OVERLAP_SIZE = 64

BATCH_SIZE = 65

RESTART_CONDITION = True

# TODO Transfer to a separate file at some pooint maybe
FOLDERS_IDS = {"قانون": "0", "قرار": "1", "مرسوم": "2", "رإي": "3", "أمر": "4"}


def group_ids(ids):
    """
    This returns the number of chunks for each text. The number is repeated for each chunk of text, because of laziness
    pretty sure there is a better way to do this.
    :param ids: list of chunks UIDs
    :return: Number of chunks for each text in a list.
    """
    id_groups = {}
    group_count = 1

    for cid in ids:
        prefix = '-'.join(cid.split('-')[:-1])
        if prefix in id_groups:
            id_groups[prefix].append(group_count)
        else:
            id_groups[prefix] = [group_count]
        group_count += 1

    result = [len(groups) for groups in id_groups.values() for group in groups]
    return result


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
                    metadata["text_date"].replace("/", "-") +
                    "-" + metadata["text_nb"]))

            for cid, chunk in enumerate(chunks):
                cuid = uid + "-" + str(cid)

                batch[cuid] = {"metadata": metadata, "tokens": chunk}

                if len(batch) == BATCH_SIZE:
                    yield batch
                    batch = {}

    if batch:
        yield batch


def generate_batch_embeddings(preprocessor, tokenizer, data_gen, model):
    """

    :param preprocessor:
    :param tokenizer:
    :param data_gen:
    :param model:
    :return:
    """

    global RESTART_CONDITION
    processed_batches = 0

    for data in data_gen:
        batch_gen = generate_batches(data, preprocessor, tokenizer)

        for batch in batch_gen:

            torch.cuda.empty_cache()
            embeddings = None
            vectors = None
            gc.collect()

            chunks = []
            metadata_list = []
            cuids = []

            num_chunks_per_id = group_ids(batch.keys())

            for cid, item in enumerate(batch.items()):
                cuid, value = item[0], item[1]
                chunk = value["tokens"]
                token_ids = tokenizer.encode(" ".join(chunk),
                                             add_special_tokens=True,
                                             padding="max_length", truncation=True,
                                             max_length=CHUNK_SIZE+2,
                                             )
                chunks.append(torch.tensor(token_ids, dtype=torch.int32))

                value["metadata"]["chunks"] = num_chunks_per_id[cid]
                metadata_list.append(value["metadata"])

                cuids.append(cuid)

            tensor_chunks = torch.stack(chunks).to("cuda")

            embeddings = model(tensor_chunks)
            # This takes the embedding of the [CLS] Token.
            vectors = embeddings["last_hidden_state"][:, 0, :]

            vectors = vectors.cpu().detach().tolist()

            yield cuids, vectors, metadata_list


