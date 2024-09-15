import csv
from transformers import AutoTokenizer
from Source.Utils.io_operations import list_files_as_paths, load_text_file_content_as_list
from Source.Utils.io_operations import load_jsonl_dataset
from Source.Utils import paths
import matplotlib.pyplot as plt
import os


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

    # for k, v in res.items():
    #     print(f"{len(v)} -> {k} -> {v}")
    #
    # print(len(res))

    return True


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

    compare_and_extract(dataset, new_refs_dataset)
