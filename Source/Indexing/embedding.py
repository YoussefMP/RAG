from transformers import AutoTokenizer, AutoModel
from arabert.preprocess import ArabertPreprocessor
from Source.Logging.loggers import get_logger
from torch.nn.functional import pad
import torch

logger = get_logger("indexer", "indexing")

chunk_size = 510
overlap_size = 128

type_keys = {
    "أمر": "4",
    "رأي": "3",
    "قانون": "2",
    "قرار": "1",
    "مرسوم": "0"
}



def reduce_dimensions_method_one(vector, chunks):

    if chunks == 0:
        return vector.mean(dim=1)


def process_batch(preprocessor, tokenizer, model, data: dict):

    meta_data_table = {}
    vectors_table = []

    did = 0

    for year in data.keys():

        logger.info(f"\t Year {year}")
        for entry in data[year]:

            meta_data_table[did] = entry["metadata"]

            text = entry["text"]
            preprocessed_text = preprocessor.preprocess(text)
            tokens = tokenizer.tokenize(preprocessed_text)
            chunks = [tokens[i:i + chunk_size] for i in range(0, len(tokens), chunk_size - overlap_size)]

            for i, chunk in enumerate(chunks):
                chunk_input_ids = torch.tensor(
                    [tokenizer.cls_token_id] + tokenizer.convert_tokens_to_ids(chunk) + [tokenizer.sep_token_id]
                )

                padded_chunk = pad(chunk_input_ids,
                                   pad=(0, chunk_size+2-chunk_input_ids.size()[0]),
                                   mode='constant', value=0
                                   )

                embedding = model(padded_chunk.unsqueeze(0))["last_hidden_state"]
                reduced_embeddings = reduce_dimensions_method_one(embedding, 0)
                vectors_table.append((did, reduced_embeddings))

        return meta_data_table, vectors_table
