from Utils.io_operations import dump_to_json
import requests
from Source.Logging.loggers import get_logger
from bs4 import BeautifulSoup
from Utils import paths
import os
import zipfile
from requests.exceptions import ConnectionError, Timeout
import time

# Make a GET request to the website
BASE_URL = "https://www.gesetze-im-internet.de/"
logger = get_logger("oldp_crawler", "gesetze_im_internet_crawler.log")


def request_with_backoff(url):
    """
    Given a URL this method sends a GET request to it with error handling for unstable connections.
    It handles the error using exponential backoff/.
    :param url:
    :return:
    """
    got_response = False
    tries = 0
    response = None
    while not got_response:
        try:
            # Make a GET request to the law text page
            response = requests.get(url)
            got_response = True
            tries += 1
        except ConnectionError:
            if tries+1 % 500 == 0:
                logger.warning("Connection error occurred. Please check your network and try again. 100")
            time.sleep(0.25 * tries)
        except Timeout:
            print("The request timed out. Try increasing the timeout value.")
        except Exception as e:
            print(f"An error occurred: {e}")

    return response


def handle_sentences(dl_tag, subpoints=None):
    """
    :param dl_tag:
    :return:
    """
    sentences = []
    ids = []

    dt_elements = dl_tag.find_all("dt")
    dd_elements = dl_tag.find_all("dd")

    for index, (dt, dd) in enumerate(zip(dt_elements, dd_elements), 1):
        sentence_object = {}
        lines_object = {}
        sid = dt.text.replace(".", "") if "." in dt.text else dt.text.replace(")", "")

        lines = dd.find("dl")
        if lines:
            lines_object, r_ids = handle_sentences(lines, True)
            ids += r_ids
            if subpoints:
                logger.error("Third level of recursion detected")

        if sid not in ids:
            ids.append(sid)
            sentence_object["id"] = sid
            sentence_object["text"] = dd.text.split(lines.text)[0].strip() if lines else dd.text

            if lines_object:
                sentence_object["lines"] = lines_object
            sentences.append(sentence_object)

            if lines and dd.text.split(lines.text)[1].strip():
                if len(dd.text.split(lines.text)) > 2:
                    logger.error("Another edge case to be handled")
                sentences.append({"id": sid, "text": dd.text.split(lines.text)[1].strip()})

    return sentences, ids


def handle_article_content(article_content):
    """
    This method parses the HTML content of the website and extracts the links to the law texts and saves them to a list.
    :param article_content:
    :return: List of law paragraphs.
    """
    paragraphs = []

    # extract paragraphs from element
    paragraph_elements = article_content.find_all("div", {"class": "jurAbsatz"})
    if not paragraph_elements:
        paragraph_elements = article_content.find_all("span", {"class": "SP"})

    for pid, paragraph in enumerate(paragraph_elements):

        paragraph_dict = {}
        dl = paragraph.find("dl")
        if dl:
            text_before_dl = paragraph.text.split(dl.text)[0].strip()
            paragraph_dict["id"] = pid
            paragraph_dict["text"] = text_before_dl
            paragraph_dict["sentences"], _ = handle_sentences(dl)

            if paragraph.text.split(dl.text)[1].strip():
                paragraphs.append(paragraph_dict)
                paragraphs.append({"id": pid, "text": paragraph.text.split(dl.text)[1].strip()})
                continue
        else:
            paragraph_dict["text"] = paragraph.text
            paragraph_dict["id"] = pid

        paragraphs.append(paragraph_dict)
    return paragraphs


def download_book_content(books):
    """
    This method parses the HTML content of the website and extracts the links to the law texts and saves them to a file.
    :param books: list of law books.
    :return:
    """

    depths = ["Buch", "Abschnitt", "Titel", "Untertitel", "Kapitel"]

    logger.info("Starting download of books")
    for book in books:
        logger.info(f"\tStarting download of book {book}")
        # Make a GET request to the book page
        book_base_url = BASE_URL + book
        book_response = request_with_backoff(book_base_url)
        # Parse the HTML content
        soup = BeautifulSoup(book_response.text, 'html.parser')

        # Find all the td elements that have a child a with a href attribute
        main_element = soup.find('div', {'id': 'paddingLR12'})
        headers = main_element.find_all('td')

        if book == "stgb":
            book_content = {"Begin": {},
                            "Allgemeiner teil": {},
                            "Besonderer Teil": {}
                            }
        else:
            book_content = {"Begin": {}}
        depth_stack = []
        keys_stack = []
        dict_iterator = book_content["Begin"]
        for i, header in enumerate(headers):

            if not header.find('a') or (header.has_attr("colspan") and header['colspan'] == "3"):
                # Depth is known at this level
                if header.has_attr("colspan") and header['colspan'] == "3":
                    chapter_title = header.find('b').text

                    for depth in depths:
                        if depth in chapter_title:

                            if depth in depth_stack:

                                if depth != depth_stack[-1]:
                                    depth_stack = depth_stack[:depth_stack.index(depth)+1]
                                    keys_stack = keys_stack[:depth_stack.index(depth)+1]
                                keys_stack[-1] = headers[i+2].find('b').text

                                dict_iterator = book_content
                                for key in keys_stack:
                                    try:
                                        dict_iterator = dict_iterator[key]
                                    except KeyError:
                                        dict_iterator[key] = {}
                                        dict_iterator = dict_iterator[key]

                            else:
                                depth_stack.append(depth)
                                keys_stack.append(headers[i+2].find('b').text)

                                dict_iterator = book_content
                                for key_index, key in enumerate(keys_stack):
                                    try:
                                        dict_iterator = dict_iterator[key]
                                    except KeyError:
                                        dict_iterator[key] = {}
                                        dict_iterator = dict_iterator[key]

                            break

                    try:
                        logger.info('\t' * (len(keys_stack)+2) + f"Reached depth {len(keys_stack)} - at {depth_stack[-1]} = {keys_stack[-1]}")
                    except IndexError:
                        pass
                continue

            elif header.has_attr("colspan") and header['colspan'] == "2":
                continue

            elif header.find('a').has_attr("href"):
                link_element = header.find('a')
                href = link_element['href']

                article_response = request_with_backoff(book_base_url + f"/{href}")

                # Parse the HTML content of the law text page
                article_soup = BeautifulSoup(article_response.text, 'html.parser')      # features="%(parser)s"
                article_title = link_element.text

                if "weggefallen" in article_title:
                    dict_iterator[article_title] = {}
                    dict_iterator[article_title]["paragraphs"] = []
                    continue

                # save the text of the law text to a list
                article_content = article_soup.find("div", {"class": "jnhtml"})
                # book_content[chapter_title][article_title]["texts"] = handle_article_content(article_content)
                dict_iterator[article_title] = {}
                dict_iterator[article_title]["paragraphs"] = handle_article_content(article_content)
                dict_iterator[article_title]["url"] = book_base_url + f"/{href}"

        book_file_path = os.path.join(paths.crawl_results_folder, f"{book}.json")
        dump_to_json(book_file_path, book_content)


def download_book_content_as_xml(books):

    for book in books:
        # Download the compressed file
        url = BASE_URL + book + "/xml.zip"
        xml_file_folder = os.path.join(paths.crawl_results_folder, "xml")
        if not os.path.exists(xml_file_folder):
            os.makedirs(xml_file_folder)
        compressed_file_path = os.path.join(xml_file_folder, f"{book}.zip")

        response = requests.get(url, stream=True)
        with open(compressed_file_path, 'wb') as f:
            f.write(response.content)
        f.close()

        # Step 2: Decompress the file
        decompressed_file_folder = os.path.join(xml_file_folder, book)
        if not os.path.exists(decompressed_file_folder):
            os.makedirs(decompressed_file_folder)
        with zipfile.ZipFile(compressed_file_path, 'r') as zip_ref:
            zip_ref.extractall(decompressed_file_folder)

        logger.info(f"Decompressed file saved to {decompressed_file_folder}")
        os.remove(compressed_file_path)


if __name__ == "__main__":
    # BOOKS = ["gg", "zpo", "stpo", "stgb"]
    # BOOKS = ["stpo", "stgb"]
    BOOKS = ["bgb", "gg", "zpo", "stpo", "stgb"]

    # download_book_content_as_xml(BOOKS)
    download_book_content(BOOKS)
