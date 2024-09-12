import torch
from datasets import Dataset
from torch.utils.data import DataLoader
from Utils.labels import TAG2ID, POSSIBLE_RELATIONS
import matplotlib.pyplot as plt


class BatchEncodingDataset():
    def __init__(self, batch_encoding):
        self.batch_encoding = batch_encoding

    def __getitem__(self, idx):
        # Create a dictionary with string keys and tensor values for the given index
        item = {key: torch.tensor(val[idx]) for key, val in self.batch_encoding.items()}
        return item

    def __len__(self):
        # Return the number of samples (length of input_ids)
        return len(self.batch_encoding['input_ids'])


def print_labeled_sequence(tokenizer, input_ids, labels):
    """extract the "1" labeled sequences from the input_ids and decode them with the tokenizer
    :param tokenizer:
    :param input_ids:
    :param labels:
    """
    labeled_sequences = []
    s = ""
    for i in range(len(input_ids)):

        if labels[i] == 1:
            original_token = tokenizer.decode([input_ids[i]])
            s += " " + original_token

        elif labels[i] == 0:
            if len(s) > 0:
                labeled_sequences.append(s)
                s = ""

    print(labeled_sequences)


def convert_to_word_level_annotations(text, entities) -> (list, list):
    words = text.split()
    word_offsets = []
    offset = 0
    for word in words:
        start = text.find(word, offset)
        end = start + len(word)
        word_offsets.append((start, end))
        offset = end

    word_labels = ["O"] * len(words)

    entities.sort(key=lambda e: e["start_offset"])

    idx_offset = 0
    for entity in entities:
        entity_start, entity_end, entity_label = entity["start_offset"], entity["end_offset"], entity["label"]
        for idx, (start, end) in enumerate(word_offsets[idx_offset:]):
            if start <= entity_end and end >= entity_start:
                word_labels[idx+idx_offset] = entity_label
            elif entity_end < start:
                idx_offset = idx
                break

    return words, word_labels


def tokenize_and_align_labels(tokenizer, examples, tag2id, max_length, remove_refs=False):
    """
    Tokenize and align the labels with the words!! not tokens. Aligning with tokens proved to be problematic, since the
    annotations is at character level, so it was difficult to map characters to tokens.
    :param tokenizer:
    :param examples:
    :param tag2id:
    :param max_length:
    :param remove_refs: if True, remove reference labels. As a preprocessing step for training the RefDisassembler
    :return:
        tokenized_inputs: BatchEncoding: an object with three attributes
                            - data = {"input_ids": , "attention_mask": , "labels": ,}
                            - encodings {list: number of examples}
    """
    # TODO: convert too long examples into multiple examples
    # tokenized_inputs = tokenizer(examples["text"], truncation=True, padding='max_length', max_length=max_length)
    tokenized_inputs = tokenizer(examples["text"])

    all_labels = []
    for i, text in enumerate(examples["text"]):

        entities = examples["entities"][i]
        if remove_refs:
            entities = [entity for entity in entities if entity["label"] != "Ref"]
        else:
            entities = [entity for entity in entities if entity["label"] == "Ref"]

        words, word_labels = convert_to_word_level_annotations(text, entities)

        # Tokenize the words
        words_tokenized_inputs = tokenizer(words,
                                           # truncation=True,
                                           # padding='max_length',
                                           # max_length=max_length,
                                           is_split_into_words=True
                                           )
        word_ids = words_tokenized_inputs.word_ids()

        labels = []
        for wi, word_id in enumerate(word_ids):
            if word_id is None:
                labels.append(0)
            else:
                labels.append(tag2id[word_labels[word_id]])

        all_labels.append(labels)

        # el = [text[e["start_offset"]:e["end_offset"]] for e in entities]
        # print(el)
        # print(len(el))
        # s = ""
        # ts = ""
        #
        # for i, l in enumerate(labels):
        #     if l != 0:
        #         s += tokenizer.convert_ids_to_tokens(words_tokenized_inputs["input_ids"][i])
        #     else:
        #         if s != "":
        #             print(s)
        #             ts += s
        #             s = ""
        # print()
        #
        #
        # for pe in el:
        #     if pe.replace(" ", "▁") not in ts:
        #         print(f"cound not find {pe.replace(' ', '▁')}")
        #         break
        # print("________________________________________________________________")

    tokenized_inputs["labels"] = all_labels
    return tokenized_inputs


def build_relation_matrix(entities, relations):
    """
    :param entities: List[ List [ dict ]], (#Examples, #Entities, EntityObject) entities as read from the dataset
    :param relations: List [ List [ dict ]], (#Examples, #Relations, RelationObject) relations as read from the dataset

    :return:
        relation_matrices: List[List[int]] representing a set of positive and negative examples for training.
                             The value at position (i, j) represents the presence of a relation from entity i to j.
                             If it is a negative example, relation_matrices[i][j] = 0 else 1.
                             The relation matrix is built based on the entities and relations provided in the input.
                             The entities are sorted based on their start index (start_offset) and
                             then based on the label ID to have a uniform order for the relation matrix during training
                              as well as during evaluation and inference.
    """
    # List of relation matrices
    relation_matrices = []

    for set_index, entity_set in enumerate(entities):

        relations_set = []
        # remove ref labeled entities
        filtered_entities = [entity for entity in entity_set if entity["label"] != "Ref"]

        if len(filtered_entities) > 1:

            # Sort entities based on their start index (start_offset) and then based on the label ID to have a uniform
            # order for the relation matrix during training as well as during evaluation and inference
            filtered_entities.sort(key=lambda entity: entity["start_offset"])
            filtered_entities.sort(key=lambda x: TAG2ID[x["label"]], reverse=True)

            for start_entity in filtered_entities:
                if TAG2ID[start_entity["label"]] == 2 or TAG2ID[start_entity["label"]] == 3:
                    continue

                for end_entity in filtered_entities[filtered_entities.index(start_entity):]:
                    if TAG2ID[end_entity["label"]] < TAG2ID[start_entity["label"]]:

                        if TAG2ID[end_entity["label"]] not in POSSIBLE_RELATIONS[TAG2ID[start_entity["label"]]]:
                            continue

                        true_relations = False
                        for rel in relations[set_index]:
                            if rel["from_id"] == start_entity["id"] and rel["to_id"] == end_entity["id"]:
                                relations_set.append(1)
                                true_relations = True
                                break

                        if not true_relations:
                            relations_set.append(0)

        relation_matrices.append(relations_set)

    # pad all lists in the relation_matrices to the length of the longest relations_set
    longest_relations_set = max(len(relation) for relation in relation_matrices)
    for relation_set in relation_matrices:
        relation_set.extend([-1] * (longest_relations_set - len(relation_set)))

    return relation_matrices


def get_dataloaders_with_labels(tokenizer, dataset, batch_size, tag2id, max_length):
    """
    This generates a dataloader for datasets with true labels.
    :param tokenizer:
    :param dataset:
    :param batch_size:
    :param tag2id:
    :param max_length:
    :return:
    """
    # Apply the function to the dataset
    encoded_dataset = tokenize_and_align_labels(tokenizer, dataset, tag2id, max_length)
    batch_encoding_dataset = BatchEncodingDataset(encoded_dataset.convert_to_tensors("pt"))

    # Initialize dataloader
    dataloader = DataLoader(batch_encoding_dataset, batch_size=batch_size, shuffle=True)

    return dataloader


def get_dataloaders_with_labels_and_relations(tokenizer, dataset, batch_size, tag2id, max_length):
    """
    This generates a dataloader for datasets with true labels and relations.
    :param tokenizer:
    :param dataset:
    :param batch_size:
    :param tag2id:
    :param max_length:
    :return:
    """
    encoded_dataset = tokenize_and_align_labels(tokenizer, dataset, tag2id, max_length, remove_refs=True)
    encoded_dataset["relations"] = build_relation_matrix(dataset["entities"], dataset["relations"])
    batch_encoding_dataset = BatchEncodingDataset(encoded_dataset.convert_to_tensors("pt"))
    # encoded_dataset["relations"] = [[0] * len(dataset["relations"])] * len(dataset["relations"])

    # Initialize dataloader
    dataloader = DataLoader(batch_encoding_dataset, batch_size=batch_size, shuffle=False)

    return dataloader


def get_dataloader(dataset, batch_size):

    # Initialize dataloader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    return dataloader

