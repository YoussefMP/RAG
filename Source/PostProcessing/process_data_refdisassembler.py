"""
When I finished training the sequence_classifier and moved to the training of the refDisassembler, I had to
separate the examples in each text. By separating the Ref spans.
Then I found out that there is a lot more Sections and Paragraphs examples then the others, so I balanced the dataset
by removing the examples that only had paragraphs and sections.
Before that however IO had to clean the dataset of duplicates that are present because of how I traversed the books
and created the examples.

29.09.2024 -
"""

from Source.Utils.io_operations import load_jsonl_dataset, dump_to_jsonl, load_jsonl_dataset_as_list
from Source.Utils import paths
import os
import json

CONFIG = {
    "TRAINING_DATASET": "Annotated_dataset",
    "DATASET_VERSION": "VR5.3"
}


def divide_ref_spans(data):
    result = []
    newid = 0

    for eid, entry_text in enumerate(data["text"]):
        entities = data["entities"][eid]
        ref_entities = [entity for entity in entities if entity["label"] == "Ref"]
        non_ref_entities = [entity for entity in entities if entity["label"] != "Ref"]

        for ref_entity in ref_entities:

            new_entry = {
                "id": newid,
                "text": entry_text[int(ref_entity["start_offset"]):int(ref_entity["end_offset"])],
                "entities": [],
                "relations": [],
                "comments": []
            }
            newid += 1

            ids_record = []
            for entity in non_ref_entities:

                if entity["start_offset"] >= ref_entity["start_offset"] and entity["end_offset"] <= ref_entity["end_offset"]:

                    new_entry["entities"].append({
                        "id": entity["id"],
                        "start_offset": entity["start_offset"] - ref_entity["start_offset"],
                        "end_offset": entity["end_offset"] - ref_entity["start_offset"],
                        "label": entity["label"]
                    })
                    ids_record.append(entity["id"])

            for relation in data["relations"][eid]:
                if len(ids_record) < 2:
                    break
                if relation["from_id"] in ids_record and relation["to_id"] in ids_record:
                    new_entry["relations"].append(relation)

            result.append(new_entry)

    return result


def balance_examples(data):
    ids_to_remove = []
    sections_removed_count = 0
    paragraph_removed_count = 0

    for i, annotation_example in enumerate(data):

        if i % 100 == 0:
            print(f"Processed {i}/{len(data)} examples")

        entry = annotation_example["entities"]
        non_ref_entities = [entity for entity in entry if entity["label"] != "Ref"]

        if len(non_ref_entities) == 1:
            if non_ref_entities[0]["label"] == "Section" and sections_removed_count < 3951:
                sections_removed_count += 1
                ids_to_remove.append(non_ref_entities[0]["id"])

            if non_ref_entities[0]["label"] == "Paragraph" and paragraph_removed_count < 3260:
                paragraph_removed_count += 1
                ids_to_remove.append(non_ref_entities[0]["id"])

        else:
            labels_list = [x["label"] for x in non_ref_entities]
            if len(set(labels_list)) == 1 and labels_list[0] == "Paragraph" and paragraph_removed_count < 3260:
                print("Removed_multiple")
                paragraph_removed_count += len(non_ref_entities)
                ids_to_remove += [e["id"] for e in non_ref_entities]
            elif len(set(labels_list)) == 1 and labels_list[0] == "Section" and paragraph_removed_count < 3951:
                print("Removed_multiple")
                sections_removed_count += len(non_ref_entities)
                ids_to_remove += [e["id"] for e in non_ref_entities]

        if paragraph_removed_count >= 3260 and sections_removed_count >= 3951:
            break

    filtered_dataset = data.filter(lambda x: (len(x["entities"]) > 0 and x["entities"][0]["id"] not in ids_to_remove))

    print(f"removed a total of {paragraph_removed_count} annotated paragraphs and "
          f"{sections_removed_count} annotated Sections")

    return filtered_dataset


def clean_duplicates(file_path, cleaned_file):
    dataset_list = load_jsonl_dataset_as_list(file_path)
    dataset_object = load_jsonl_dataset(file_path)
    ids_to_remove = []
    mapping = {}

    for eid, entry in enumerate(dataset_list):

        if dataset_object["text"].count(dataset_object["text"][eid]) > 1:

            if dataset_object["text"][eid] not in mapping.keys():
                mapping[dataset_object["text"][eid]] = dataset_object["text"].count(dataset_object["text"][eid])

            if mapping[dataset_object["text"][eid]] > 1:
                ids_to_remove.append(eid)
                mapping[dataset_object["text"][eid]] -= 1

    offset = 0
    for i in ids_to_remove:
        if dataset_list[i-offset]["text"] in mapping:
            dataset_list.pop(i - offset)
            offset += 1
        else:
            print("onerroe")

    with open(cleaned_file, "w", encoding="utf8") as f:

        for entry in dataset_list:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    f.close()


if __name__ == "__main__":
    # dataset = load_jsonl_dataset(os.path.join(paths.annotations_folder,
    #                                           f"{CONFIG['TRAINING_DATASET']}_{CONFIG['DATASET_VERSION']}.jsonl"))
    #
    # ref_disassembler_dataset = divide_ref_spans(dataset)
    # output_file = os.path.join(paths.annotations_folder, "Archives", f"ref_disassembler_{CONFIG['TRAINING_DATASET']}_{CONFIG['DATASET_VERSION']}.jsonl")
    #
    # dump_to_jsonl(output_file, ref_disassembler_dataset)
    #
    # new_dataset = load_jsonl_dataset(output_file)
    # new_output_file = os.path.join(paths.annotations_folder, f"Balanced_{CONFIG['TRAINING_DATASET']}_{CONFIG['DATASET_VERSION']}.jsonl")
    # new_data = balance_examples(new_dataset)
    # dump_to_jsonl(new_output_file, new_data)

    new_output_file = os.path.join(paths.annotations_folder, f"Balanced_{CONFIG['TRAINING_DATASET']}_{CONFIG['DATASET_VERSION']}.jsonl")
    cleaned_file = os.path.join(paths.annotations_folder, f"Cleaned_{CONFIG['TRAINING_DATASET']}_{CONFIG['DATASET_VERSION']}.jsonl")
    dataset = load_jsonl_dataset(cleaned_file)
    new_data = balance_examples(dataset)
    dump_to_jsonl(new_output_file, new_data)


