"""
Script that handles all FileIO operation and some extras that will come later.
"""

import os
import re
import json
import pandas as pd
from datasets import Dataset


#######################
# File Methods   ######
#######################
def write_to_file(path: str, content: str):
    with open(path, "w", encoding="utf-8") as out_file:
        out_file.write(content)
    out_file.close()


def load_text_file_content_as_list(out_path: str) -> list[str]:
    with open(out_path, 'r', encoding='utf-8') as outfile:
        content = outfile.readlines()

    return content


def load_txt_file_content_as_str(out_path: str) -> str:
    with open(out_path, 'r', encoding='utf-8') as outfile:
        content = outfile.read()
    return content


def save_node_as_document(base_folder, path, content):

    if len(path.split("/")) > 1:
        page_path = os.path.join(base_folder, path.replace("/", "\\"))
        page_path = make_os_conform(page_path)
        if content:
            os.makedirs(page_path.rsplit("\\", 1)[0], exist_ok=True)
            write_to_file(page_path + ".txt", content)
    else:
        if not os.path.exists(os.path.join(base_folder, path)):
            path = make_os_conform(path)
            os.makedirs(os.path.join(base_folder, path))


def dump_to_json(path, content):
    """Given a path and a content dumps the content to a json file."""

    create_folder_and_subfolders(path)

    with open(path, 'w', encoding='utf-8') as outfile:
        json.dump(content, outfile, ensure_ascii=False, indent=4)
    outfile.close()


def dump_to_jsonl(path, content):

    if not os.path.exists(path):
        create_folder_and_subfolders(path)

    with open(path, 'w', encoding="utf-8") as file:
        for entry in content:
            json.dump(entry, file, ensure_ascii=False)
            file.write('\n')
    file.close()


def load_jsonl_dataset(file_path):
    data = []
    with open(file_path, "r", encoding='utf8') as file:
        for line in file:
            data.append(json.loads(line))

    # convert the data list into a dataframe
    df = pd.DataFrame(data, columns=["id", "text", "entities"])

    # Convert the DataFrame to a Dataset
    dataset = Dataset.from_pandas(df)

    return dataset


#######################
# Os methods    #######
#######################
def make_os_conform(path):
    # Remove invalid characters
    name = re.sub(r'[/:*?"<>|]', '_', path)

    # Replace spaces with underscores
    name = name.replace(' ', '_')
    return name


def create_folder_and_subfolders(file_path):
    """
    Given a file path, checks if all directories exist and creates them if not.
    :param file_path:
    """
    dirname = os.path.dirname(file_path)
    if not os.path.exists(dirname):
        os.makedirs(dirname)


def generate_child_file_path(path: str) -> str:
    """
    Creates a text file in the folder with the same name as the parent folder.
    :param path:
    :return:
    """
    parent_folder, title = path.rsplit("\\", 1)
    file_path = f"{parent_folder}\\{title}.txt"
    return file_path


def list_folder_content(folder_path):
    if os.path.exists(folder_path):
        return os.listdir(folder_path)
    else:
        print(f"Could not find Folder {folder_path}")
        return []


