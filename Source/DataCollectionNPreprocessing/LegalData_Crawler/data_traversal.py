import os
import json
from Utils import paths
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
        'AbsatzCombined': r'Absatz(?:es)?\s\d+',
        'ArtikelCombined': r'Artikels?\s\d+(\sAbs\.\s\d+)?(\sSatz\s\d+)?(\sNr\.)?',
        'ParagraphCombined': r'§§?\s\d+[a-z]?(?:\sAbsatz\s\d+\sSatz\s\d+)?(?:\sbis\s\d+)?',
        'Absätze': r'Absätze\s\d+\sbis\s\d+',
        'Artikeln': r'Artikeln\s\d+\sund\s\d+',
        'Directive': r'Richtlinie\s95/\d+/EG\s*\(Datenschutz-Grundverordnung\)',
        'BracketedParagraph': r'\(\§\s\d+\sAbs\.\s\d+\)',
        'EU': r'\(EU\)\s\d+/\d+'
    }

    for key, pattern in patterns.items():
        matches = re.findall(pattern, sentence)
        if matches:
            return True

    return False


def extract_refs(data, path, text_count, result):
    """

    :param data:
    :param path:
    :param text_count:
    :param result:
    :return:
    """

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


def extract_sentences(data):
    sentences = []

    for sentence in data["sentences"]:

        if "lines" in list(sentence.keys()):
            for line in sentence["lines"]:
                if "text" in list(line.keys()):
                    if check_for_ref(f"{data['text']} {sentence['text']} {line['text']}"):
                        sentences.append((len(data['text']), len(sentence["text"]), f"{data['text']} {sentence['text']} {line['text']}"))

                elif "lines" in list(line.keys()):
                    for sub_line in line["lines"]:
                        if "text" in list(sub_line.keys()):
                            if check_for_ref(f"{data['text']} {sentence['text']} {line['text']} {sub_line['text']}"):
                                sentences.append((len(data['text']), len(sentence["text"]), f"{data['text']} {sentence['text']} {line['text']} {sub_line['text']}"))

        else:
            if check_for_ref(f"{data['text']} {sentence['text']}"):
                sentences.append((len(data['text']), len(sentence["text"]), f"{data['text']} {sentence['text']}"))

    return sentences


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

        # references_dataset = extract_refs(book, "", 0, [])
        references_dataset = extract_long_refs(book, "", 0, [])
        print(f"Extracted {len(references_dataset)} references from {title}")

        # result_file_path = os.path.join(paths.german_law_books, f"extracted_refs_{title.replace('.json', '')}.txt")
        # with open(result_file_path, "w", encoding="utf-8") as f:
        #     for text in references_dataset:
        #         f.write(f"{text}\n")
        # f.close()

        # Save to CSV file
        result_file_path = os.path.join(paths.german_law_books, f"extracted_refs_{title.replace('.json', '')}.csv")
        with open(result_file_path, 'w', newline="", encoding="utf-8") as csvfile:
            fieldnames = ['b_length', "s_length", 'text']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter="|")

            # Write headers
            writer.writeheader()

            # Write data
            for row in references_dataset:
                writer.writerow({'b_length': row[0], "s_length": row[1], 'text': row[2]})

