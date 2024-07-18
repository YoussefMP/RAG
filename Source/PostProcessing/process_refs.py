import faulthandler
from Utils import paths
import json
import os
import re
from Utils.io_operations import dump_to_jsonl

on_the_fly = False
DEBUG = False

CONFIG = {"MODEL_VERSION": "v0.7",
          "THRESHOLD": 15,
          }


KEYWORDS = {
    "Kapitel(?:s|n)?",
    "Absatz(?:es|e|.)?",
    "Absätze(?:n)?",
    "Artikel(?:s|n)?",
    "Nr\\.?|Nummer(?:n)?",
    "§",
    "Satz(?:es|e|en|.)?",
    "Sätze(?:n)?",
    "Halbsatz"
}
pattern = re.compile(r'(?:\W|^)(?:' + '|'.join(KEYWORDS) + r')(?:\W|$)?', re.IGNORECASE)


def extract_and_consolidate(entry):
    """
    This method concatenates the uncertain predictions with the certain ones based on proximity threshold.
    :param entry: dictionary containing "text" and "label" fields
    :return: 
    """
    got_one = False

    text = entry["text"]
    labels = entry["label"]

    # Initialize an empty list to store the final combined labels
    combined_labels = []

    # Sort labels based on the first item of each entry
    sorted_labels = sorted(labels, key=lambda x: x[0])

    lid = 0
    while lid < len(sorted_labels):

        start, end, tag = sorted_labels[lid]
        last_tag = tag
        if lid == len(sorted_labels) - 1:
            combined_labels.append([start, end, tag])
            break

        jump = False
        it = lid + 1

        # it iterated over the following labels based on the span we determine whether to fuse the blocks or not
        while it < len(sorted_labels) and not jump:
            # n_ is for next
            n_start, n_end, n_tag = sorted_labels[it]

            # Set a limit of 7 chars of distance between blocks for combinations
            # If the two predictions were made with certainty we keep them separate
            # except if the second block contains the book or law to which it is being referenced
            # got_one for debugging purposes
            # last_tag to keep track of the last fused block
            if (end + CONFIG["THRESHOLD"] >= n_start):
                    # and (not (last_tag == n_tag == "Ref") or ("gesetz" in text or "buch" in text))):
                got_one = True
                end = n_end
                last_tag = n_tag
                if last_tag == "Ref" or n_tag == "Ref":
                    tag = "Ref"
                # so not to add the last element twice if it was fused
                if it == len(sorted_labels)-1:
                    it += 1
            else:
                jump = True
            lid = it
            it += 1

        combined_labels.append([start, end, tag])

    if got_one and DEBUG:
        print("________________________________________________________________")
        print(text)
        for e in sorted_labels:
            print(text[e[0]:e[1]], "  //  ", e[0], ", ", e[1], ", ", e[2])
        print("\t\t+++++++++++\t\t")
        for ce in combined_labels:
            print(text[ce[0]:ce[1]], "  //  ", ce[0], ", ", ce[1], ", ", ce[2])
        print("________________________________________________________________")
        input("")

    entry["label"] = combined_labels
    return entry


def clean_uncertain_predictions(entry):
    """
    This method removes uncertain predictions that do not meet certain criteria.
    :param entry: dictionary containing "text" and "label" fields
    :return:
    """
    keywords = ["gesetz", "Art", "bis", "abs", "buch", "titel", "abschnitt"]

    labels = entry["label"]
    sorted_labels = sorted(labels, key=lambda x: x[0])

    to_delete = []  # "Uncertain" classified spans that are faulty positives that should be deleted

    for ref_id, ref in enumerate(sorted_labels):
        s, e, label = ref
        pre, suc = None, None   # Placeholders for the following and preceding labeled spans

        if label == "Uncertain":
            ref_span = entry["text"][s:e]
            to_delete.append(ref_id)
            if is_certain(ref_span) or any([kw in ref_span for kw in keywords]):
                # Loop to find the first valid preceding classified span
                it_finder = ref_id-1
                while it_finder in to_delete and it_finder >= 0:
                    it_finder -= 1
                if it_finder >= 0 and ref[0] - sorted_labels[it_finder][1] < 25:
                    pre = sorted_labels[it_finder]

                # Loop to find the first valid following classified span
                it_finder = ref_id+1
                while it_finder in to_delete and it_finder < len(sorted_labels):
                    it_finder += 1
                if ref_id < len(sorted_labels) -1:
                    suc = sorted_labels[ref_id+1]

                # Consolidating the "uncertain" classified span with another span
                if pre is None and suc is None:
                    continue
                if suc is not None and pre is not None:
                    if ref[0] - pre[1] < suc[0] - ref[1]:
                        pre[1] = ref[1]
                    else:
                        suc[0] = ref[0]
                elif suc is None:
                    pre[1] = ref[1]
                elif pre is None:
                    suc[0] = ref[0]

    # parse the to_delete list in reverse order
    for d_id in to_delete[::-1]:
        # delete d_id for sorted_labels
        del sorted_labels[d_id]

    entry["label"] = sorted_labels
    return entry


def print_labeled_sequence(tokenizer, input_ids, labels):
    """
    Prints the input sequence along with the labels.
    :param tokenizer: tokenizer object
    :param input_ids: input sequence
    :param labels: labels
    :return:
    """
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    for token, label in zip(tokens, labels):
        print(f"{token}: {label}")


def is_certain(span):
    return bool(pattern.search(span))


if __name__ == '__main__' and not on_the_fly:

    faulthandler.enable()

    # Cleaned list of entries
    cleaned_entries = []

    # Define the path to the output file
    output_file = os.path.join(paths.annotations_folder, "V3_Cleaned_dataset.jsonl")

    # Define the path to the model output folder
    files_folder = os.path.join(paths.model_output_folder, CONFIG["MODEL_VERSION"])
    # list only files in the model output folder
    files = [f for f in os.listdir(files_folder) if os.path.isfile(os.path.join(files_folder, f))]

    for file in files:
        # Load the dataset from the JSON file
        with open(os.path.join(files_folder, file), 'r', encoding="utf-8") as f:
            for line in f.readlines():
                entry_object = json.loads(line)

                # processed_entry = clean_uncertain_predictions(extract_and_consolidate(entry_object))
                # cleaned_entries.append(processed_entry)



        f.close()

    # Write the cleaned entries to the output file
    dump_to_jsonl(output_file, cleaned_entries)
