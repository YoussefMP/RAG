import os
from Utils import paths
import torch
import json


e0 = torch.load("emissions_0.pt")
e = torch.load("emissions_1.pt")

print(torch.allclose(e0, e))


def print_results():

    path = os.path.join(paths.model_output_folder, "v0.7\\ref_annotations_{}_t{}_{}_0.jsonl")
    with open(path, "r", encoding="utf-8") as f:

        for line in f.readlines():
            data = json.loads(line)
            print(data["text"])
            print(data["label"])

            for seq in data["label"]:
                print(data["text"][seq[0]:seq[1]], end="\t//\t")

            print("\n\n")

print_results()