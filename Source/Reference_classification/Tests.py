from Source.Reference_classification.sequence_classifier import RefDissassembler
from data_processor import get_dataloaders_with_labels_and_relations
from Source.Utils.io_operations import load_jsonl_dataset
from transformers import AutoTokenizer
from Source.Utils.paths import *
from Source.Utils.labels import *

TEST_CONFIG = {
    "MODEL_NAME": "FacebookAI/xlm-roberta-large",
    "NUM_CLASSES": 9,
    "NUM_RELATIONS": 1,
    "BATCH_SIZE": 1,
}


def test_span_pairs_generation(data, labels=None):

    for batch in data:
        input_ids = batch["input_ids"]
        labels = batch['labels']
        # Removing the padding from the relations and flattening the labels
        relations = batch["relations"]

        dummy_embeddings = classifier(input_ids)
        generated_training_pairs = classifier.get_spans_pairs(dummy_embeddings, None, labels)

        print("hello")

    return None


classifier = RefDissassembler("DummyModel", 9, 1, True)

tokenizer = AutoTokenizer.from_pretrained("FacebookAI/xlm-roberta-large")

test_dataset = load_jsonl_dataset(os.path.join(annotations_folder, "test_examples.jsonl"))
dataloader = get_dataloaders_with_labels_and_relations(tokenizer,
                                                       test_dataset,
                                                       TEST_CONFIG["BATCH_SIZE"],
                                                       TAG2ID,
                                                       None
                                                       )

test_span_pairs_generation(dataloader, None)
