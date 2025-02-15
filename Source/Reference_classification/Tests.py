from Source.Reference_classification.sequence_classifier import RefDissassembler
from data_processor import get_dataloaders_with_labels_and_relations
from Source.Utils.io_operations import load_jsonl_dataset
from transformers import AutoTokenizer
from Source.Utils.paths import *
from Source.Utils.labels import *
import torch

TEST_CONFIG = {
    "MODEL_NAME": "FacebookAI/xlm-roberta-large",
    "NUM_CLASSES": 9,
    "NUM_RELATIONS": 1,
    "BATCH_SIZE": 8,
}


def test_span_pairs_generation(data, labels=None):

    # TODO: relation to twos and threes in forward direction in classifier
    for batch in data:
        input_ids = batch["input_ids"]
        labels = batch['labels']
        # Removing the padding from the relations and flattening the labels
        relations = batch["relations"]

        # generating random embeddings for testing
        dummy_embeddings = torch.randn(len(input_ids), len(input_ids[0]), 16)
        classifier_spans = classifier.pair_sequence_embeddings(dummy_embeddings, None, labels)

        for rid, relation_sample in enumerate(relations):
            classifier_pairs = [item[0] for item in classifier_spans[rid][1]]
            for pid, pair in enumerate(relation_sample):
                pair = (int(pair[0]), int(pair[1]))
                if sum(pair) == -2:
                    break
                elif pair not in classifier_pairs:
                    print(f"labels in order {classifier_spans[rid][0]}")
                    print(f"classifier pairs {classifier_pairs}")
                    print(f"missing annotated pair {pair}")
                    print(f"sentence is : {tokenizer.decode(input_ids[rid])}")
                    print("\n")
                    print("#" * 64)
                    print("\n")

    return None


print("Defining classifier ...")
classifier = RefDissassembler(TEST_CONFIG["MODEL_NAME"],
                              TEST_CONFIG["NUM_CLASSES"],
                              TEST_CONFIG["NUM_RELATIONS"]
                              )

print("Defining Tokenizer ...")
tokenizer = AutoTokenizer.from_pretrained("FacebookAI/xlm-roberta-large")

print("Loading dataset ...")
test_dataset = load_jsonl_dataset(os.path.join(annotations_folder, "Annotated_dataset_VDV5.4.jsonl"))
# This generated the first pairs that are defined in the training dataset
dataloader = get_dataloaders_with_labels_and_relations(tokenizer,
                                                       test_dataset,
                                                       TEST_CONFIG["BATCH_SIZE"],
                                                       TAG2ID,
                                                       None
                                                       )

# def build_relation_matrix(entities, relations):

# This generates the pairs that are generated live during training
print("Testing span generation method ...")
test_span_pairs_generation(dataloader, None)
