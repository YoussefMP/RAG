import torch
from datasets import Dataset
from torch.utils.data import DataLoader


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


def convert_to_word_level_annotations(text, entities):
    words = text.split()
    word_offsets = []
    offset = 0
    for word in words:
        start = text.find(word, offset)
        end = start + len(word)
        word_offsets.append((start, end))
        offset = end

    word_labels = ["O"] * len(words)

    sorted_entities = sorted(entities, key=lambda x: x[2] != 'Ref')

    for entity in sorted_entities:
        if entity[2] != "Ref":
            continue
        entity_start, entity_end, entity_label = entity[0], entity[1], entity[2]
        for idx, (start, end) in enumerate(word_offsets):
            if start >= entity_start and end <= entity_end:
                word_labels[idx] = entity_label

    return words, word_labels


def tokenize_and_align_labels(tokenizer, examples, tag2id, max_length):
    """
    Tokenize and align the labels with the words!! not tokens. Aligning with tokens proved to be problematic, since the
    annotations is at character level, so it was difficult to map characters to tokens.
    :param tokenizer:
    :param examples:
    :param tag2id:
    :param max_length:
    :return:
    """
    tokenized_inputs = tokenizer(examples["text"], truncation=True, padding='max_length', max_length=max_length)
    all_labels = []
    for i, text in enumerate(examples["text"]):

        entities = examples["label"][i]
        words, word_labels = convert_to_word_level_annotations(text, entities)

        # Tokenize the words
        words_tokenized_inputs = tokenizer(words,
                                           truncation=True,
                                           padding='max_length',
                                           max_length=max_length,
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

    tokenized_inputs["labels"] = all_labels
    return tokenized_inputs


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
    dataloader = DataLoader(batch_encoding_dataset, batch_size=batch_size)

    return dataloader


def get_dataloader(dataset, batch_size):

    # Initialize dataloader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return dataloader

