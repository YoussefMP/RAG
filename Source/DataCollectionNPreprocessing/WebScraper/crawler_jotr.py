import os.path

from bs4 import BeautifulSoup
import requests
import json


def download_file(pdf_url):

    # Send an HTTP GET request to the PDF URL
    response = requests.get(pdf_url)

    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        # Specify the local file path where you want to save the PDF
        doc_name = pdf_url.split("/")[-1]
        local_file_path = f".\\..\\..\\1_Resouces\\JOTR_files\\{doc_name}"  # Replace with your local file path

        if not os.path.exists(local_file_path):
            # Save the PDF content to the local file
            with open(local_file_path, 'wb') as pdf_file:
                pdf_file.write(response.content)

            print(f"PDF file downloaded and saved to {local_file_path}")
        else:
            print(f"{doc_name} already exists")
    else:
        print(f"Failed to download the PDF file. => {pdf_url}")


def get_info_text(info, wc, lang):
    for item in wc["texts"]:
        if item.get(info):
            for text_item in item.get(info):
                if text_item.get(lang):
                     return text_item.get(lang)


def extract_metadata_from_box(element, wc, lang):
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


def extract_pdf_link(element, lang):

    # Find the element with the title
    element_soup = BeautifulSoup(str(element), "html.parser")

    links_elements = element_soup.find_all("a", href=True)

    for link_element in links_elements:
        if lang == "ar" and "Arabe" in link_element.text:
            return link_element["href"]
        elif lang == "fr" and "Fran" in link_element.text:
            return link_element["href"]


def pist_crawl(base_url, wc, language):

    for section in wc["sections"]:

        tag, link = list(section.items())[0]
        link = f"https://www.pist.tn/search?ln={language}&cc={tag}&action_search=Recherche&rg=10&of=hd"

        response = requests.get(link)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')

            record_boxes = soup.find_all("div", "detailedrecordbox")
            record_minipanels = soup.find_all("div", "detailedrecordminipanel")

            for bid in range(len(record_boxes)):

                meta_data = extract_metadata_from_box(record_boxes[bid], wc, language)
                pdf_link = extract_pdf_link(record_minipanels[bid], language)
                download_file(base_url + pdf_link)
        else:
            print(f"Failed to load page {link}")

    return None

def main(file_path):
    web_config_file = open(file_path, "r", encoding="utf-")
    web_config = json.load(web_config_file)
    base_url = web_config["base_url"]

    language = "ar"
    pist_crawl(base_url, web_config, language)


if __name__ == '__main__':
    main(".\\web_configs\\pist.json")