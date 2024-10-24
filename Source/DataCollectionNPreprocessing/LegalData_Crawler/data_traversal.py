import os
import json
from Utils import paths
from Utils.io_operations import dump_to_jsonl
from Utils.decorators import deprecated
import re
import csv


DEBUG = False


def load_json_books(condition=None):
    """
    Returns the json files in which the law books are saved.
    :return:
    """
    files = list(os.listdir(paths.crawl_results_folder))

    if DEBUG:
        files = files[:1]

    for file in files:
        if file.endswith(".json"):
            json_data = json.load(open(os.path.join(paths.crawl_results_folder, file), "r", encoding="utf-8"))
        else:
            continue

        yield json_data, file


def check_for_ref(sentence):
    patterns = {
        # ChatGPTs
        'AbsatzCombined': r'Absatz(?:es)?\s\d+',
        'ArtikelCombined': r'Artikels?\s\d+(\sAbs\.\s\d+)?(\sSatz\s\d+)?(\sNr\.)?',
        'ParagraphCombined': r'§§?\s\d+[a-z]?(?:\sAbsatz\s\d+\sSatz\s\d+)?(?:\sbis\s\d+)?',
        'Absätze': r'Absätzen?\s\d+',
        'Artikeln': r'Artikeln\s\d+\sund\s\d+',
        'Directive': r'Richtlinie\s95/\d+/EG\s*\(Datenschutz-Grundverordnung\)',
        'BracketedParagraph': r'\(\§\s\d+\sAbs\.\s\d+\)',

        # Mine
        'Line': r'Nummern?\s\d+',
        'Satz': r'Satz(:?es)\s\d+',
        'EU': r'\(EU\)\s(Nr\.)?\s\d+/\d+',
        'EG': r'\(EG\)\s(Nr\.)?\s\d+/\d+',
        "Agreement": r'Übereinkommen'
    }

    if "richtlinie" in sentence:
        return True

    for key, pattern in patterns.items():
        matches = re.findall(pattern, sentence)
        if matches:
            return True

    return False


def extract_sentences(data, negative=False):
    sentences = []

    for sentence in data["sentences"]:

        if "lines" in list(sentence.keys()):
            for line in sentence["lines"]:
                if "text" in list(line.keys()):
                    if check_for_ref(f"{data['text']} {sentence['text']} {line['text']}") and not negative:
                        sentences.append((len(data['text']), len(sentence["text"]), len(line["text"]), -1,
                                          f"{data['text']} {sentence['text']} {line['text']}"))
                    elif negative and not check_for_ref(f"{data['text']} {sentence['text']} {line['text']}"):
                        sentences.append(f"{data['text']} {sentence['text']} {line['text']}")

                elif "lines" in list(line.keys()):
                    for sub_line in line["lines"]:
                        if "text" in list(sub_line.keys()):
                            if check_for_ref(f"{data['text']} {sentence['text']} {line['text']} {sub_line['text']}")\
                                    and not negative:
                                sentences.append((len(data['text']), len(sentence["text"]), len(line["text"]),
                                                  len(sub_line["text"]),
                                                  f"{data['text']} {sentence['text']} {line['text']} {sub_line['text']}"))
                            elif negative and\
                                    not check_for_ref(f"{data['text']} {sentence['text']} {line['text']} {sub_line['text']}"):

                                sentences.append(f"{data['text']} {sentence['text']} {line['text']} {sub_line['text']}")

        else:
            if check_for_ref(f"{data['text']} {sentence['text']}") and not negative:
                sentences.append((len(data['text']), len(sentence["text"]), -1, -1,
                                  f"{data['text']} {sentence['text']}"))
            elif negative and not check_for_ref(f"{data['text']} {sentence['text']}"):
                sentences.append(f"{data['text']} {sentence['text']}")

    return sentences


def extract_negatives(data, path, text_count, result):
    """
    :param data:
    :param path:
    :param text_count:
    :param result:
    :return:
    """
    try:
        for key in data.keys():
            if "\\" in key:
                print(f"Warning: {key}")

            if "Inhaltsübersicht" in key:
                continue

            path += f"{key}\\"

            if "paragraphs" in key:
                for paragraph in data["paragraphs"]:
                    if "sentences" in list(paragraph.keys()):
                        result += extract_sentences(paragraph, negative=True)
                        # result[path] = extract_sentences(paragraph)
                    elif not check_for_ref(paragraph["text"]):
                        result += [paragraph["text"]]

                return result

            result = extract_negatives(data[key], path, text_count, result)
            path = path[:path[:-1].rfind("\\")+1]
    except AttributeError as e:
        pass

    return result


@deprecated
def extract_refs(data, path, text_count, result):
    """
    :param data:
    :param path:
    :param text_count:
    :param result:
    :return:
    """
    sid = 0

    if "texts" in list(data.keys()):

        for sentence in data["texts"]:
            if "richtlinie" in sentence.lower():
                result.append(sentence)
                continue

            if check_for_ref(sentence):
                result.append(sentence)
    else:

        for key in data.keys():
            if "\\" in key:
                print(f"Warning: {key}")
            path += f"{key}\\"

            result = extract_refs(data[key], path, text_count, result)
            path = path[:path[:-1].rfind("\\")+1]

    return result


def extract_long_refs(data, path, text_count, result):
    """
    :param data:
    :param path:
    :param text_count:
    :param result:
    :return:
    """
    try:
        for key in data.keys():
            if "\\" in key:
                print(f"Warning: {key}")

            if "Inhaltsübersicht" in key:
                continue

            path += f"{key}\\"

            if "paragraphs" in key:
                for paragraph in data["paragraphs"]:
                    if "sentences" in list(paragraph.keys()):
                        result += extract_sentences(paragraph)
                        # result[path] = extract_sentences(paragraph)

                return result

            result = extract_long_refs(data[key], path, text_count, result)
            path = path[:path[:-1].rfind("\\")+1]
    except AttributeError as e:
        pass

    return result


if __name__ == "__main__":
    DATA = load_json_books()

    for book, title in DATA:

        references_dataset = extract_negatives(book, "", 0, [])
        print(f"Extracted {len(references_dataset)} references from {title}")

        # save references to csv file using the method save_refs_to_csv in io_operations
        # path os.path.join(paths.german_law_books, f"extracted_refs_{title.replace('.json', '')}.csv")

        # format data to fit jsonl format
        jsonl_result = []
        sid = 0
        for text_example in references_dataset:
            jsonl_result.append({"id": sid, "text": text_example, "entities": [], "comments": [], "relations": []})
            sid += 1
        # save the negatives to jsonl file to include them in the dataset
        result_path = os.path.join(paths.annotations_folder, "negatives_VR5.2.jsonl")
        dump_to_jsonl(result_path, jsonl_result)
