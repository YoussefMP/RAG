import torch
from tqdm import tqdm
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from Source.Utils.labels import TAG2ID, POSSIBLE_RELATIONS


# Class modelling the Dataset
class BatchEncodingDataset(Dataset):
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


def collate_fn(batch):
    """
    Custom collate function for DataLoader to handle batching of dictionaries with varying lengths.
    This function is used to pad the relations list in each batch to the maximum length of relations in the batch.

    Parameters:
    batch (List[Dict]): A list of dictionaries, where each dictionary represents a sample.
                        Each dictionary contains the following keys:
                        - "relations": A list of integers representing the relations for the sample.

    Returns:
    Dict: A dictionary containing the collated and padded batch data.
          The dictionary has the following keys:
          - "relations": A tensor of shape (batch_size, max_rel_length) containing the padded relations.
    """
    keys = batch[0].keys()
    collated = {}

    max_rel_length = max(len(item["relations"]) for item in batch)

    for key in keys:
        if key == "relations":
            padded_relations = []
            for item in batch:
                rels = item[key]  # Tensor of shape [N, 2]
                pad_amount = max_rel_length - rels.shape[0]
                pad_tensor = torch.full((pad_amount, 2), -1, dtype=rels.dtype)  # Padding tensor
                padded_relations.append(torch.cat([rels, pad_tensor], dim=0))  # Concatenate
            collated[key] = torch.stack(padded_relations)  # Stack into batch tensor
        else:
            collated[key] = torch.stack([item[key].clone().detach() for item in batch])

    return collated


#######################################################
# Methods for getting the data in the right format ####
#######################################################

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
    truncate = True if max_length is not None else False
    pad = "max_length" if max_length is not None else True

    tokenized_inputs = tokenizer(examples["text"], truncation=truncate, padding=pad, max_length=max_length)
    # Split the examples into lists of words
    split_examples = list(map(lambda s: s.split(), examples["text"]))
    # Tokenize the words using the tokenizer for alignment with the labels
    words_tokenized_inputs = tokenizer(split_examples,
                                       truncation=truncate,
                                       padding=pad,
                                       max_length=max_length,
                                       is_split_into_words=True
                                       )

    all_labels = []
    for i, text in tqdm(enumerate(examples["text"]),
                        total=len(examples['text']),
                        desc="Tokenizing and aligning labels with their corresponding tokens: "):

        entities = examples["entities"][i]
        if not remove_refs:
            entities = [entity for entity in entities if entity["label"] == "Ref"]

        words, word_labels = convert_to_word_level_annotations(text, entities)
        word_ids = words_tokenized_inputs[i].word_ids

        labels = [0]
        for wi, word_id in enumerate(word_ids[1:]):
            if word_id is None:
                labels += [0] * (len(word_ids) - len(labels))
                break
            labels.append(tag2id[word_labels[word_id]])

        all_labels.append(labels)

    tokenized_inputs["labels"] = all_labels
    return tokenized_inputs


def build_relation_matrix_v2(entities, relations):
    # List of relation matrices
    relation_matrices = []

    for set_index, entity_set in tqdm(enumerate(entities),
                                      total=len(entities),
                                      desc="Building the relations matrix: "):

        relations_set = []

        if len(entity_set) > 1:
            # Sort entities based on their start index (start_offset)
            entity_set.sort(key=lambda element: element["start_offset"])

            for relation in relations[set_index]:
                relation_position_based = [-1, -1]
                start_entity_id, target_entity_id = relation["from_id"], relation["to_id"]
                for eid, entity in enumerate(reversed(entity_set)):
                    if entity["id"] == start_entity_id:
                        relation_position_based[0] = eid
                    elif entity["id"] == target_entity_id:
                        relation_position_based[1] = eid
                    if relation_position_based[0] != -1 and relation_position_based[1] != -1:
                        break
                relations_set.append(relation_position_based)
        relation_matrices.append(relations_set)

    return relation_matrices


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
    max_relations_length = 0

    for set_index, entity_set in tqdm(enumerate(entities),
                                      total=len(entities),
                                      desc="Building the relations matrix: "):

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
        max_relations_length = max(max_relations_length, len(relations_set))

    # pad all lists in the relation_matrices to the length of the longest relations_set
    for relation_set in relation_matrices:
        relation_set.extend([-1] * (max_relations_length - len(relation_set)))

    return relation_matrices


####################
# Main Methods #####
####################

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
    encoded_dataset["relations"] = build_relation_matrix_v2(dataset["entities"], dataset["relations"])
    # batch_encoding_dataset = BatchEncodingDataset(encoded_dataset.convert_to_tensors("pt"))

    batch_encoding_dataset = BatchEncodingDataset(encoded_dataset)
    # encoded_dataset["relations"] = [[0] * len(dataset["relations"])] * len(dataset["relations"])

    # Initialize dataloader
    dataloader = DataLoader(batch_encoding_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    return dataloader


def get_dataloader(dataset, batch_size):

    # Initialize dataloader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    return dataloader

