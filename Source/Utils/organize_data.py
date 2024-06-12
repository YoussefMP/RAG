from io_operations import *
import paths


def remove_training_data():

    files_names = list_folder_content(paths.extracted_refs_folder)
    TEXTS = []

    for filename in files_names:
        TEXTS += load_text_file_content_as_list(os.path.join(paths.extracted_refs_folder, filename))

    training_data = load_jsonl_dataset(paths.annotations_file)

    print(len(TEXTS))

    removed = 0
    for text in TEXTS:
        if text.strip("\n") in training_data["text"]:
            TEXTS.remove(text)
            removed += 1

    print(removed)
    print(len(TEXTS))

    with open(os.path.join(paths.extracted_refs_folder, "refs_dataset_without_training.txt"), "w", encoding="utf-8") as f:
        for text in TEXTS:
            f.write(f"{text}")
    f.close()


def convert_export_to_import_format():

    def convert_entities_to_labels(entities):
        result = []
        try:
            for entity in entities:
                result.append([entity["start_offset"], entity["end_offset"], entity["label"]])
        except Exception as e:
            print("GGSDG", e)
        return result

    files = ["annotated_dataset_long.jsonl"]

    for file_name in files:
        file_name = os.path.join(paths.annotations_folder, file_name)

        data = load_jsonl_dataset(file_name)

        content = []
        for entry in data:

            jsonl = {"id": entry["id"],
                     "text": entry["text"],
                     "label": convert_entities_to_labels(entry["entities"]),
                     "Comments": []
                     }
            content.append(jsonl)

        dump_to_jsonl(file_name.split(".")[0] + "_import.jsonl", content)


remove_training_data()
