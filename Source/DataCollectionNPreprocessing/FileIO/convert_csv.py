from pyarabic import araby
from nltk.tokenize.treebank import TreebankWordDetokenizer
from Utils.paths import *
from nltk.tokenize import word_tokenize
import csv
import os
import re


def pre_clean(t_chunk):
    araby.strip_diacritics(t_chunk)

    pattern4 = r'\s{2,}'
    t_chunk = re.sub(pattern4, " ", t_chunk)

    t_chunk = t_chunk.replace("ـ", "")
    t_chunk = t_chunk.replace("", "")

    if t_chunk.isspace():
        return ""
    return t_chunk


def remove_table_of_content(text):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    sid = find_start_page(sentences)

    new_text = ""
    if re.search(r'### (\d+) صفحة ###', sentences[sid - 1]):
        if not sentences[sid - 1].split('###')[-1].isspace():
             new_text += sentences[sid - 1].split('###')[-1]

    for i in range(sid, len(sentences)):
        new_text += sentences[i]

    return new_text


def detect_table_of_content(sentence):
    pattern = r'(\b\w+\b)\s+\1'
    matches = re.findall(pattern, sentence)

    if matches:
        return True
    return False


def extra_cleaning_step(chunks):
    page = 0
    sid = []
    for cid in range(len(chunks)):

        chunk = TreebankWordDetokenizer().detokenize(chunks[cid])

        tc = detect_table_of_content(chunk)
        if tc:
            sid.append(cid)
        if re.search(r'NEWPAGE', chunk):
            page += 1
        if page >= 2:
            break
    sid.reverse()
    for tdi in sid:
        chunks.pop(tdi)

    return chunks


def find_start_page(sentences: list):
    start_id = 0
    sid = 0
    page = 0
    sid_set = False

    while sid < len(sentences) and page <= 2:
        if detect_table_of_content(sentences[sid]):
            while not re.search(r'### (\d+) صفحة ###', sentences[sid]):
                sid += 1
            page += 1
            sid_set = False
        else:
            if not sid_set:
                start_id = sid
                sid_set = True
            if re.search(r'### (\d+) صفحة ###', sentences[sid]):
                page += 1

        sid += 1

    if re.search(r'### (\d+) صفحة ###', sentences[sid - 1]):
        if not sentences[sid - 1].split('###')[-1].isspace():
            start_id -= 1

    return start_id


def chunk_text(text, chunk_size=350, overlap=50, padding="[PAD]") -> list:
    chunks = []
    start = 0

    tokens = word_tokenize(text)

    while start < len(tokens):
        end = start + chunk_size
        chunk = tokens[start:end]
        # Append the chunk to the list
        chunks.append(chunk)
        # Move the starting point forward with the specified chunk_size
        start += chunk_size - overlap

    # Pad the last chunk to ensure it has the same size
    last_chunk = chunks[-1]
    last_chunk += [padding for _ in range(chunk_size - len(last_chunk))]

    return chunks


def make_file_csv(file_path, out_file_path):

    file = open(file_path, "r", encoding="utf-8")

    input_text = pre_clean(remove_table_of_content(file.read()))
    input_text = input_text.replace(r'### (\d+) صفحة ###', "[NEWPAGE]")

    chunks = chunk_text(input_text, )
    chunks = extra_cleaning_step(chunks)

    with open(out_file_path, 'w', newline='', encoding='utf-8') as csvfile:
        csv_writer = csv.writer(csvfile, quoting=csv.QUOTE_MINIMAL)

        for chunk in chunks:
            sentence = TreebankWordDetokenizer().detokenize(chunk)
            csv_writer.writerow([sentence])

    csvfile.close()
    return True


def main(in_path, out_path):

    # for y_folder in os.listdir(in_path):
    for y_folder in ["2000"]:

        year_folder_path = os.path.join(in_path, y_folder)
        out_year_folder_path = os.path.join(out_path, y_folder)
        os.makedirs(out_year_folder_path, exist_ok=True)

        for file_name in os.listdir(year_folder_path):
            file_path = os.path.join(year_folder_path, file_name)
            output_file_path = os.path.join(out_year_folder_path, f"XX{file_name.replace('_extracted.txt', '.csv')}")
            make_file_csv(file_path, output_file_path)


if __name__ == "__main__":

    law_files_path = os.path.join(jotr_documents_path, "Loi_txt")
    output_path = os.path.join(jotr_documents_path, "Loi_csv")
    main(law_files_path, output_path)
