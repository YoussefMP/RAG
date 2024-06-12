from Source.Utils.io_operations import load_text_file_content_as_list
from data_processor import get_dataloader
from transformers import AutoTokenizer
from collections import defaultdict
from Utils.io_operations import dump_to_jsonl
from Utils import paths
from Utils.labels import *
from tqdm import tqdm
from torch import torch
import os
import gc


CONFIG = {
    "MODEL_LINK": "FacebookAI/xlm-roberta-large",
    "MODEL_NAME": "xlm-roberta-large",
    "VERSION": "r0.5",
    "BATCH_SIZE": 8,
    "MAX_LENGTH": 256,
    "DEVICE": torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'),
    # "DEVICE": "cpu",
    "OUT_FILE": "ref_annotations_r0.5_t0.95.jsonl",
}


def merge_intervals(intervals):
    merged = []
    for start, end in sorted(intervals):
        if merged and merged[-1][1] >= start - 1:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def format_result(texts, input_ids, offset_mapping, predictions):

    formatted_batch_result = []
    for eid in range(len(input_ids)):
        grouped_intervals = defaultdict(list)

        for label, offset in zip(predictions[eid], offset_mapping[eid]):
            if label == 0:
                continue
            grouped_intervals[label].append(offset)

        formatted_result = []
        for label, intervals in grouped_intervals.items():
            merged = merge_intervals(intervals)

            for interval in merged:
                formatted_result.append([interval[0], interval[1], ID2TAG[label]])

        result_entry = {"text": texts[eid], "label": formatted_result}
        formatted_batch_result.append(result_entry)

    return formatted_batch_result


def filter_predictions(predictions, confidences, threshold):
    filtered_predictions = []
    for preds, confs in zip(predictions, confidences):
        filtered_preds = []
        for pred, conf in zip(preds, confs):
            if conf > threshold:
                filtered_preds.append(pred)
            elif 0.9 < conf < threshold:
                filtered_preds.append(9)  # Replace with uncertainty tag
            else:
                filtered_preds.append(0)  # Replace with a neutral tag or None
        filtered_predictions.append(filtered_preds)
    return filtered_predictions


def annotate_dataset(tokenizer, classifier, dataloader, max_length, device, threshold=None):

    # Transferring model to device
    classifier.to(device)

    # Evaluation
    classifier.eval()
    predictions = []

    annotated_results = []
    with torch.no_grad():
        processed_batches = 0
        file_extension = 0
        for batch in tqdm(dataloader):
            batch = [line.rstrip() for line in batch]
            tokenized_input = tokenizer.batch_encode_plus(batch,
                                                          add_special_tokens=True,
                                                          truncation=True,
                                                          padding=True,
                                                          max_length=max_length,
                                                          return_offsets_mapping=True
                                                          )

            input_ids = torch.tensor(tokenized_input.input_ids).to(device)
            attention_mask = torch.tensor(tokenized_input.attention_mask).to(device)

            if threshold:
                output, confidence = classifier.predict_with_confidence(input_ids, attention_mask, device)
                output = filter_predictions(output, confidence, threshold=threshold)
            else:
                output = classifier(input_ids, attention_mask)

            predictions.extend(output)

            results = format_result(
                batch,
                tokenized_input.input_ids,
                tokenized_input.offset_mapping,
                predictions
            )

            annotated_results.extend(results)

            processed_batches += 1
            if processed_batches % 200 == 0:
                file_name = CONFIG['OUT_FILE'].replace(".jsonl", f"_{str(file_extension)}.jsonl")
                dump_to_jsonl(
                    os.path.join(os.path.join(paths.model_output_folder, CONFIG["VERSION"]), file_name),
                    annotated_results
                )
                annotated_results = []
                file_extension += 1

            # optimizing by emptying cache and collecting garbage
            gc.collect()
            torch.cuda.empty_cache()


        file_name = CONFIG['OUT_FILE'].replace(".jsonl", f"_{str(file_extension)}.jsonl")
        dump_to_jsonl(
            os.path.join(os.path.join(paths.model_output_folder, CONFIG["VERSION"]), file_name),
            annotated_results
        )


if __name__ == "__main__":

    TOKENIZER = AutoTokenizer.from_pretrained(CONFIG["MODEL_LINK"])
    model_name = f"{CONFIG['MODEL_NAME']}_{CONFIG['VERSION']}"

    # initialize model
    CLASSIFIER = torch.load(os.path.join(paths.pretrained_classifiers_folder, model_name))

    # files_names = list_folder_content(paths.extracted_refs_folder)
    # TEXTS = []
    #
    # for filename in files_names:
    #     TEXTS += load_text_file_content_as_list(os.path.join(paths.extracted_refs_folder, filename))

    TEXTS = load_text_file_content_as_list(os.path.join(paths.extracted_refs_folder, "refs_dataset_without_training.txt"))
    DATALOADER = get_dataloader(TEXTS, CONFIG["BATCH_SIZE"])

    annotate_dataset(TOKENIZER, CLASSIFIER, DATALOADER, CONFIG["MAX_LENGTH"], CONFIG['DEVICE'], threshold=0.96)

