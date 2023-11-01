"""
Script that handles all IO operation and some extras that will come later.
"""

import os
import re


#######################
# File Methods   ######
#######################
def write_to_file(path: str, content: str):
    with open(path, "w", encoding="utf-8") as out_file:
        out_file.write(content)
    out_file.close()


def load_file_content_as_list(out_path: str) -> list[str]:
    with open(out_path, 'r', encoding='utf-8') as outfile:
        content = outfile.readlines()

    return content


def load_file_content_as_str(out_path: str) -> str:
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


#######################
# Os methods    #######
#######################
def make_os_conform(path):
    # Remove invalid characters
    name = re.sub(r'[/:*?"<>|]', '_', path)

    # Replace spaces with underscores
    name = name.replace(' ', '_')
    return name


def generate_child_file_path(path: str) ->str:
    parent_folder, title = path.rsplit("\\", 1)
    file_path = f"{parent_folder}\\{title}.txt"
    return file_path
