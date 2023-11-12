"""
This script is for crawling the pist.tn website and downloading all the records saved on it.
The records are saved in pdf files.
This script also extracts any meta_data about the file.

08.11.2023
V 1.0.0
"""

from Source.DataCollectionNPreprocessing.FileIO import pdf_reader as PR
from Source.paths import *
from Source.Logging import loggers
from bs4 import BeautifulSoup
import requests
import os.path
import json
import re


logger = loggers.get_logger("jotr_crawler", "jotr_crawler.log")

link = "https://www.pist.tn/search?ln={}&cc={}&jrec={}&of=hd&rg={}"


def update_start_record(s_rec: int) -> None:
    """
    The records are iterated in groups of 10 (for now). This function saved the record reached, so when starting the
    script at later point it does not have to revisit all records, but starts from the saved int in the file.
    :param s_rec: Record reached
    :return:
    """
    with open(os.path.join(web_config_path, "start_rec.txt"), "w") as srf:
        srf.write(str(s_rec))
    srf.close()


def get_start_record() -> int:
    """
    This return the record at which the retrieval of the webpages should start at
    :return: int : the start record
    """
    srf = os.path.join(web_config_path, "start_rec.txt")
    if os.path.exists(srf):
        f = open(srf, "r")
        try:
            s_rec = int(f.read())
        except ValueError:
            return 1
        return s_rec
    else:
        return 1


def set_last_rec(elements: list) -> int:
    """
    This returns the total number of records for a section
    :param elements: the HTML elements that might contain the number of total records
    :return: int : total number of records
    """
    pattern = r'jrec=(\d+)'
    max_jrec = -1

    # Iterate through the elements and find the greatest number
    for element in elements:
        match = re.search(pattern, str(element))
        if match:
            jrec_value = int(match.group(1))
            if jrec_value > max_jrec:
                max_jrec = jrec_value
                max_jrec_element = element

    return max_jrec


def file_downloaded(pdf_url, section, year_folder):
    """
    Checks if file is already downloaded
    :param pdf_url:
    :param section:
    :param year_folder:
    :return:
    """
    doc_name = pdf_url.split("/")[-1]

    # year_folder = doc_name.split(".")[0][-4:]
    local_file_path = os.path.join(jotr_documents_path, section, year_folder, doc_name)

    return os.path.exists(local_file_path)


def download_file(pdf_url, section, meta_data: dict, year_folder) -> str:
    """
    I think you can guess from the name what this does.
    :param year_folder:
    :param pdf_url: url of the record's pdf file
    :param section: Section under which the file is saved
    :param meta_data:
    :return:
    """
    # Send an HTTP GET request to the PDF URL
    response = requests.get(pdf_url)

    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        # Specify the local file path where you want to save the PDF
        doc_name = pdf_url.split("/")[-1]

        os.makedirs(os.path.join(jotr_documents_path, section, year_folder), exist_ok=True)

        local_file_path = os.path.join(jotr_documents_path, section, year_folder, doc_name)

        # Save the PDF content to the local file
        with open(local_file_path, 'wb') as pdf_file:
            pdf_file.write(response.content)
        pdf_file.close()

        with open(local_file_path.replace(".pdf", ".txt"), "w", encoding="utf-8") as meta_file:
            for key, val in meta_data.items():
                meta_file.write(f"{key}: {val}\n")
        meta_file.close()
        logger.debug(f"\t\t\t\tPDF file downloaded and saved to {local_file_path}")
    else:
        year_folder = ""
        doc_name = ""
        logger.warning(f"Failed to download the PDF file. => {pdf_url}")

    return os.path.join(year_folder, doc_name)


def get_info_text(info, wc, lang) -> str:
    """
    this is a helper method for extracting metadata. It returns the key to extract the information needed from
    the element.
    :param info:
    :param wc:
    :param lang:
    :return:
    """
    for item in wc["texts"]:
        if item.get(info):
            for text_item in item.get(info):
                if text_item.get(lang):
                    return text_item.get(lang)


def extract_metadata_from_box(element, wc, lang) -> dict:
    """
    The name kinda spoils the plot, don't you think ?
    :param element:
    :param wc: stand for web_config and not the toilet
    :param lang:
    :return:
    """
    record_info = {}

    # Find the element with the title
    element_soup = BeautifulSoup(str(element), "html.parser")

    title_element = element_soup.find("div", id="HB")
    if title_element:
        title_text = title_element.find('big')
        if title_text:
            record_info['Title'] = title_text.text.strip()

    # Find elements with strong tags
    strong_elements = element_soup.find_all('strong')

    for element in strong_elements:
        text = element.text.strip()
        if get_info_text("date", wc, lang) in text:
            date_value = element.find_next('span').text.strip()
            record_info['date'] = date_value

        elif get_info_text("number", wc, lang) in text:
            doc_number_value = element.find_next('span').text.strip()
            record_info['number'] = doc_number_value

        elif get_info_text("jort_nb", wc, lang) in text:
            jort_number_value = element.find_next('span').text.strip()
            record_info['jort_nb'] = jort_number_value

        elif get_info_text("fr_page", wc, lang) in text:
            fr_version_page = element.find_next('span').text.strip()
            record_info['fr_page'] = fr_version_page

        elif get_info_text("ar_page", wc, lang) in text:
            ar_version_page = element.find_next('span').text.strip()
            record_info['ar_page'] = ar_version_page

        elif get_info_text("keywords", wc, lang) in text:
            keywords_value = element.find_next('span').text.strip()
            keywords_list = [keyword.strip() for keyword in keywords_value.split(';')]
            record_info['Keywords'] = keywords_list

    return record_info


# def extract_pdf_link_2(element, lang):


def extract_pdf_link(element, lang):
    """
    Again self-explanatory.
    :param element:
    :param lang:
    :return:
    """

    # I resoup the element just so I can use the find_all method. There might be a better way to do this.
    element_soup = BeautifulSoup(str(element), "html.parser")
    # Find the element with the title
    links_elements = element_soup.find_all("a", href=True)

    for link_element in links_elements:

        element_text = link_element.text.lower()
        try:
            previous_element_text = link_element.previous_sibling.previous_sibling.text.lower()
        except AttributeError:
            pass

        if lang == "ar" and ("arabe" in element_text or "arabe" in previous_element_text):
            return link_element["href"], link_element
        elif lang == "fr" and ("fran" in element_text or "fran" in previous_element_text):
            return link_element["href"], link_element

    return None, links_elements


def pist_crawl(base_url, wc, language):
    """
    :param base_url: home page link
    :param wc: contains the base_url, the extensions to the link, and some other data that helps crawl
                the website
    :param language:
    :return:
    """
    logger.info(f"Iterating the different sections")
    for section in wc["sections"]:

        logger.info(f"\tStarting with {section}")
        folder_name, tag = list(section.items())[0]

        doc_folder = os.path.join(jotr_documents_path, folder_name)
        logger.info(f"\tCreating folder to save downloaded documents {doc_folder}")
        os.makedirs(doc_folder, exist_ok=True)

        start_rec = get_start_record()
        logger.info(f"\tGetting the start record index \n\t\t start_record = {start_rec}")
        end_rec = -1        # Total number of records
        increment = 10
        total_download_files = []

        while end_rec == -1 or (start_rec <= end_rec != -1):
            files_to_add = []
            c_link = link.format(language, tag, start_rec, increment)

            logger.info(f"{'#'*30}")
            logger.info(f"\t\tSending request to get the next {increment} from {start_rec} records")
            response = requests.get(c_link)
            if response.status_code == 200:

                soup = BeautifulSoup(response.text, 'html.parser')
                if end_rec == -1:
                    end_rec = set_last_rec(soup.find_all("a", class_="img", href=True))

                record_boxes = soup.find_all("div", "detailedrecordbox")
                record_minipanels = soup.find_all("div", "detailedrecordminipanel")

                logger.info(f"\t\t\tStarting the download loop ...")
                for bid in range(len(record_boxes)):
                    pdf_link, element_of_link = extract_pdf_link(record_minipanels[bid], language)
                    meta_data = extract_metadata_from_box(record_boxes[bid], wc, language)
                    meta_data["link"] = base_url + pdf_link
                    year = re.findall(r'\b\d{4}\b', meta_data["date"])[0]
                    try:
                        if int(year) >= 2000:
                            # if not file_downloaded(base_url + pdf_link, folder_name, year):
                            if (not pdf_link.split("/")[-1] in total_download_files and
                                    not file_downloaded(base_url + pdf_link, folder_name, year)):

                                logger.debug(f"\t\t\tDownloading the file at {pdf_link}")
                                file_to_add = download_file(base_url + pdf_link, folder_name, meta_data, year)

                                if file_to_add:
                                    logger.info(f"\t\t\t\t Added one more file ==> {pdf_link.split('/')[-1]}")
                                    total_download_files.append(pdf_link.split("/")[-1])
                                    files_to_add.append(file_to_add)
                            else:
                                # logger.debug(f"\t\t\tFile {pdf_link} already downloaded")
                                pass
                    except TypeError:
                        logger.exception("Error while trying to download file")
                        logger.error(f"Values that might have cause the problem {base_url} \n {pdf_link} \n {folder_name} \n {element_of_link}")
                        new = input("ther was an error go checl the logs ")
            else:
                print(f"Failed to load page {c_link}")

            # call extraction method for the downloaded files
            if files_to_add:
                PR.main(folder_name, files_to_add)

            start_rec += increment
            update_start_record(start_rec)


def main(file_path):
    logger.info(f"parsing the web config file at {file_path}")
    web_config_file = open(file_path, "r", encoding="utf-")
    web_config = json.load(web_config_file)
    base_url = web_config["base_url"]
    web_config_file.close()

    language = "ar"
    logger.info(f"starting the crawl - base_url = {base_url}")
    pist_crawl(base_url, web_config, language)


if __name__ == '__main__':
    main(jotr_json_config)
