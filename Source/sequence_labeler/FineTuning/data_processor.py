import json
import torch
import pandas as pd
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

    sorted_entities = sorted(entities, key=lambda x: x['label'] != 'Ref')

    for entity in sorted_entities:
        entity_start, entity_end, entity_label = entity["start_offset"], entity["end_offset"], entity["label"]
        for idx, (start, end) in enumerate(word_offsets):
            if start >= entity_start and end <= entity_end:
                word_labels[idx] = entity_label

    return words, word_labels


def tokenize_and_align_labels(tokenizer, examples, tag2id, max_length):

    tokenized_inputs = tokenizer(examples["text"], truncation=True, padding='max_length', max_length=max_length)
    all_labels = []
    for i, text in enumerate(examples["text"]):

        entities = examples["entities"][i]
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


def get_dataloaders(tokenizer, dataset, batch_size, tag2id, max_length):

    # Apply the function to the dataset
    encoded_dataset = tokenize_and_align_labels(tokenizer, dataset, tag2id, max_length)
    batch_encoding_dataset = BatchEncodingDataset(encoded_dataset.convert_to_tensors("pt"))

    # Initialize dataloader
    dataloader = DataLoader(batch_encoding_dataset, batch_size=batch_size)

    return dataloader


def load_dataset(file_path):
    data = []
    with open(file_path, "r", encoding='utf8') as file:
        for line in file:
            data.append(json.loads(line))

    # convert the data list into a dataframe
    df = pd.DataFrame(data, columns=["id", "text", "entities"])

    # Convert the DataFrame to a Dataset
    dataset = Dataset.from_pandas(df)

    return dataset



