from Source.DataCollectionNPreprocessing.FileIO.convert_csv import pre_clean,remove_table_of_content
from Utils.paths import *

jotr = True


texts = ""
if jotr:
    folders = [str(i+2000) for i in range(15)]
    folders += ["2019", "2020", "2021", "2023"]
    out_file = "Jotr_docs_collection.txt"
else:
    folders = ["economie-tunisie", "global-news", "national-news", "politic-tunisie", "tech", "regional-news"]
    out_file = "mosaique_articles_collection.txt"

for folder in folders:

    if jotr:
        fp = os.path.join(jotr_documents_path, "Loi_txt", folder)
    else:
        fp = os.path.join(mosaique_articles_path, folder)

    for fn in os.listdir(fp):
        file_path = os.path.join(fp, fn)

        file = open(file_path, "r", encoding="utf-8")

        input_text = pre_clean(remove_table_of_content(file.read()))
        input_text = input_text.replace(r'###', "\n\n")
        texts += f"{input_text}\n\n"


output_path = os.path.join(resources_folder_path, "Custom_data")
os.makedirs(output_path, exist_ok=True)
with open(os.path.join(output_path, out_file), "w", encoding="utf-8") as of:
    of.write(texts.replace(" ", " "))
