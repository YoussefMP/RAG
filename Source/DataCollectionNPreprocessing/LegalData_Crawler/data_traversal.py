import os
import json
from Utils import paths
import re

DEBUG = False


def load_data():
    files = list(os.listdir(paths.crawl_results_folder))

    if DEBUG:
        files = files[:1]

    for file in files:
        if file.endswith(".json"):
            json_data = json.load(open(os.path.join(paths.crawl_results_folder, file), "r", encoding="utf-8"))
        else:
            continue

        yield json_data, file


def postorder_traversal(data, path, text_count, result):

    if "texts" in list(data.keys()):
        # keywords = ["abs.", "absatz", "§", "§§", "artikel", "satz",
        #             "absatzes", "absätze", "(eu)", "richtlinie", "abschnitt",
        #             ]

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

        for sentence in data["texts"]:
            if "richtlinie" in sentence.lower():
                result.append(sentence)
                continue
            for key, pattern in patterns.items():
                matches = re.findall(pattern, sentence)
                if matches:
                    try:
                        result.append(sentence)
                    except TypeError:
                        print("FVDWSF")
                    break
    else:

        for key in data.keys():
            if "\\" in key:
                print(f"Warning: {key}")
            path += f"{key}\\"
            result = postorder_traversal(data[key], path, text_count, result)
            path = path[:path[:-1].rfind("\\")+1]

    return result


if __name__ == "__main__":
    DATA = load_data()

    for book, title in DATA:
        references_dataset = postorder_traversal(book, "", 0, [])
        print(f"Extracted {len(references_dataset)} references from {title}")

        result_file_path = os.path.join(paths.german_law_books, f"extracted_refs_{title.replace('.json', '')}.txt")
        with open(result_file_path, "w", encoding="utf-8") as f:
            for sentence in references_dataset:
                f.write(f"{sentence}\n")
        f.close()


