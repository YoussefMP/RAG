import json 
from Utils import paths
import os 

on_the_fly = False
DEBUG = False

CONFIG = {"MODEL_VERSION": "v0.7",
          "THRESHOLD": 7,
          }


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
            # got_one for debugging purposes
            # last_tag to keep track of the last fused block
            if end + CONFIG["THRESHOLD"] >= n_start and not (last_tag == n_tag == "Ref"):
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


def clean_uncertain_predictions():
    return



if __name__ == '__main__' and not on_the_fly:

    # Define the path to the model output folder
    files_folder = os.path.join(paths.model_output_folder, CONFIG["MODEL_VERSION"])
    
    # list only files in the model output folder
    files = [f for f in os.listdir(files_folder) if os.path.isfile(os.path.join(files_folder, f))]
        
    for file in files:
        # Load the dataset from the JSON file
        with open(os.path.join(files_folder, file), 'r', encoding="utf-8") as f:
            for line in f.readlines():
                entry_object = json.loads(line)
                extract_and_consolidate(entry_object)
        break
