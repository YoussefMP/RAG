import csv
from Source.Utils.io_operations import list_files_as_paths, load_text_file_content_as_list
from Source.Utils.io_operations import load_jsonl_dataset
from Source.Utils import paths
import os


def compare_and_extract(old_data, new_data):
    nid = [i+1 for i in range(len(new_data))]
    res = {}
    duplicates = []

    for dp in old_data:

        text = dp["text"]
        found_one = False

        for dpid, new_dp in enumerate(new_data):
            dp_text = new_dp["text"]
            if text.startswith(dp_text[:int(new_dp["b_length"])]):

                if dp_text[int(new_dp["b_length"]):int(new_dp["b_length"])+1+int(new_dp["s_length"])] in text:

                    if dp["id"] in res.keys():
                        res[dp["id"]].append(dpid)
                    else:
                        res[dp["id"]] = [dpid]

                    if dpid+1 in nid:
                        nid.remove(dpid+1)
                    else:
                        try:
                            duplicates.append(dpid+1)
                        except TypeError as e:
                            print("DAFUQ")
                # print(new_dp["text"])
                # found_one = True

        # if found_one:
        #     print(text)
        #     print("\n--------------------------------\n")

    for k, v in res.items():
        print(f"%s: %s" % (k, v))

    print(set(nid))
    print(set(duplicates))
    return True


if __name__ == '__main__':

    # model_name = "FacebookAI/xlm-roberta-large"
    # roberta_tokenizer = AutoTokenizer.from_pretrained(model_name)
    # # roberta_tokenizer = None
    #
    dataset_name = "Annotated_dataset_VR5.2.jsonl"
    dataset = load_jsonl_dataset(os.path.join(paths.annotations_folder, dataset_name))

    new_refs_files = list_files_as_paths(paths.german_law_books, [".csv"])
    new_refs_dataset = []
    for file in new_refs_files:
        with open(file, 'r', encoding="utf-8") as csvfile:
            reader = csv.DictReader(csvfile, delimiter='|')

            for row in reader:
                new_refs_dataset.append(row)

    compare_and_extract(dataset, new_refs_dataset)
