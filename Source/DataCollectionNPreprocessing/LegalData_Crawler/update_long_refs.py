import csv
from transformers import AutoTokenizer
from Source.Utils.io_operations import list_files_as_paths, load_text_file_content_as_list
from Source.Utils.io_operations import load_jsonl_dataset, dump_to_jsonl
from Source.Utils import paths
import matplotlib.pyplot as plt
import json
import os


# Took it manually from Dataset VR5.2
START_ID = 37570
E_START_ID = 200000
R_START_ID = 5847


def bar_plot(x, y):
    # Create a bar plot for length_data
    plt.figure(figsize=(10, 6))

    # Plot the bar graph
    plt.bar(x, y)

    # Set the x-axis and y-axis labels
    plt.xlabel('Length of Text')
    plt.ylabel('Number of Occurrences')

    # Set the title of the plot
    plt.title('Distribution of Text Lengths')

    # Show the plot
    plt.show()


def replace_examples(old_data, new_data, mapping):
    global START_ID
    global E_START_ID
    global R_START_ID
    
    res = []

    for entry in old_data:

        old_text = entry["text"]
        entry["entities"].sort(key=lambda entity: entity["start_offset"])

        if entry["id"] in mapping:
            
            for coordinates in mapping[entry["id"]]:
                entities_mapping = {}
                new_entry = {"id": START_ID, "text": "", "comments": [], "entities": [], "relations": []}
                START_ID += 1

                new_text = new_data[coordinates[0]][coordinates[1]-2]["text"]
                new_entry["text"] = new_text

                annotated_parts = {}
                for entity in entry["entities"]:
                    if old_text[entity["start_offset"]:entity["end_offset"]] in new_text:

                        start_offset_new_text = new_text.find(old_text[entity["start_offset"]:entity["end_offset"]])
                        end_offset_new_text = start_offset_new_text + len(old_text[entity["start_offset"]:entity["end_offset"]])

                        skip = False
                        while (start_offset_new_text, end_offset_new_text) in annotated_parts:
                            # check if the reference is mentioned again in the same text
                            start_offset_new_text = new_text[end_offset_new_text:].find(old_text[entity["start_offset"]:entity["end_offset"]])
                            if start_offset_new_text == -1:
                                skip = True
                                break

                            start_offset_new_text += end_offset_new_text
                            end_offset_new_text = start_offset_new_text + len(old_text[entity["start_offset"]:entity["end_offset"]])

                        if skip:
                            continue

                        for key in annotated_parts.keys():
                            if (key[0] <= start_offset_new_text and key[1] >= end_offset_new_text) or \
                                    (start_offset_new_text <= key[0] and end_offset_new_text >= key[1]) or \
                                    (key[0] == start_offset_new_text) or (key[1] == end_offset_new_text):

                                if annotated_parts[key]["label"] == entity["label"]:
                                    if end_offset_new_text-start_offset_new_text > key[1] - key[0]:
                                        # remove the old entity
                                        removed = False
                                        for e in new_entry["entities"]:
                                            if e["id"] == annotated_parts[key]["id"]:
                                                new_entry["entities"].remove(e)
                                                removed = True
                                                # del entities_mapping[e["id"]]
                                        if removed:
                                            break
                                        del annotated_parts[key]
                                        break
                                    else:
                                        skip = True

                        if skip:
                            continue

                        new_entry["entities"].append({"start_offset": start_offset_new_text,
                                                      "end_offset": end_offset_new_text,
                                                      "label": entity["label"],
                                                      "id": E_START_ID})
                        entities_mapping[entity["id"]] = E_START_ID

                        annotated_parts[(start_offset_new_text, end_offset_new_text)] =\
                            {"start_offset": start_offset_new_text,
                             "end_offset": end_offset_new_text,
                             "label": entity["label"],
                             "id": E_START_ID
                             }
                        E_START_ID += 1

                # for relation in entry["relations"]:
                #     if relation["from_id"] in entities_mapping and relation["to_id"] in entities_mapping:
                #         new_entry["relations"].append({
                #             "from_id": entities_mapping[relation["from_id"]],
                #             "to_id": entities_mapping[relation["to_id"]],
                #             "type": relation["type"],
                #         })

                res.append(new_entry)

    return res


def compare_and_extract(old_data, new_data):

    records = {}
    for title, book_dataset in new_data.items():
        records[title] = [i+2 for i in range(len(book_dataset))]

    res = {}

    for dp in old_data:

        text = dp["text"]

        for title, book_dataset in new_data.items():
            for dpid, new_dp in enumerate(book_dataset):
                dpid += 2
                dp_text = new_dp["text"]
                if text.startswith(dp_text[:int(new_dp["b_length"])]):
                    if dp_text[int(new_dp["b_length"]):int(new_dp["b_length"])+1+int(new_dp["s_length"])].strip() in text:

                        if int(new_dp["b_length"])+1+int(new_dp["s_length"]) <= int(new_dp["b_length"]):
                            print("Hold'up")

                        if dp["id"] in res.keys():
                            res[dp["id"]].append((title, dpid))
                        else:
                            res[dp["id"]] = [(title, dpid)]

                        try:
                            records[title].remove(dpid)
                        except Exception as e:
                            pass

                elif dp_text[int(new_dp["b_length"]):int(new_dp["b_length"])+1+int(new_dp["s_length"])].strip() in text:
                    try:
                        records[title].remove(dpid)
                    except Exception as e:
                        pass

    count = 0
    for k, v in res.items():
        print(f"{len(v)} -> {k} -> {v}")
        count += len(v)

    print(f"Total matches: {count}")

    return res


if __name__ == '__main__':

    # model_name = "FacebookAI/xlm-roberta-large"
    # roberta_tokenizer = AutoTokenizer.from_pretrained(model_name)
    # roberta_tokenizer = None

    dataset_name = "Annotated_dataset_VR5.2.jsonl"
    dataset = load_jsonl_dataset(os.path.join(paths.annotations_folder, dataset_name))

    new_refs_files = list_files_as_paths(paths.german_law_books, [".csv"])
    new_refs_dataset = {}

    length_data = {}

    for file in new_refs_files:
        book = file[file.rfind("\\")+1:].replace("extracted_refs_", "").replace(".csv", "")
        new_refs_dataset[book] = []

        with open(file, 'r', encoding="utf-8") as csvfile:
            reader = csv.DictReader(csvfile, delimiter='|')

            for row in reader:
                new_refs_dataset[book].append(row)

                # tokens = roberta_tokenizer.tokenize(row["text"])
                # if len(tokens) in length_data.keys():
                #     length_data[len(tokens)] += 1
                # else:
                #     length_data[len(tokens)] = 1

    # bar_plot(list(length_data.keys()), list(length_data.values()))

    examples_to_replace = compare_and_extract(dataset, new_refs_dataset)
    new_entries = replace_examples(dataset, new_refs_dataset, examples_to_replace)
    
    # Assuming new_entries is the list of dicts obtained from replace_examples function
    dump_to_jsonl(os.path.join(paths.temp_files_path, "new_entries.jsonl"), new_entries)



