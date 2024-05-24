from Source.Logging.loggers import get_logger
from Source import paths
import os
import json

logger = get_logger("oldp_indexing_logger", "indexing_oldp.log")


def load_data():
    files = list(os.listdir(paths.german_law_books))

    for file in files:
        json_data = json.load(open(os.path.join(paths.german_law_books, file), "r", encoding="utf-8"))

        yield json_data


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

