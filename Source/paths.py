import os


# C:\Users\maist\Desktop\Projects\RAG\Source
source_folder_path = os.path.abspath(os.path.dirname(__file__))

#################
# Resources  ####
#################
resources_folder_path = os.path.join(source_folder_path, "1_Resouces")
config_folder_path = os.path.join(source_folder_path, "0_conf")
web_config_path = os.path.join(source_folder_path, "DataCollectionNPreprocessing/WebScraper/web_configs")

# Mosaique
###########
mosaique_articles_path = os.path.join(resources_folder_path, "Mosaique_articles")
mosaique_json_config = os.path.join(source_folder_path,
                                    "DataCollectionNPreprocessing/WebScraper/web_configs/mosaique.json")

# JOTR
###########
jotr_documents_path = os.path.join(resources_folder_path, "JOTR_files")
jotr_json_config = os.path.join(source_folder_path, "DataCollectionNPreprocessing/WebScraper/web_configs/pist.json")
# TODO: add paths with placeholders for the extracted files and the csv files

# OLDP
###########
german_law_books = os.path.join(resources_folder_path, "German Law Books")


#################
# Configuration #
#################
log_files_folder = os.path.join(source_folder_path, "Logging\\log_files")



