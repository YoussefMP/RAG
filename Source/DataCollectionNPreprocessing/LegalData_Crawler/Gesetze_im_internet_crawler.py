from Utils.io_operations import dump_to_json
import requests
from Source.Logging.loggers import get_logger
from bs4 import BeautifulSoup
from Utils import paths
import os


# Make a GET request to the website
BASE_URL = "https://www.gesetze-im-internet.de/"
logger = get_logger("oldp_crawler", "gesetze_im_internet_crawler.log")


def handle_article_content(article_content):
    """
    This method parses the HTML content of the website and extracts the links to the law texts and saves them to a list.
    :param article_content:
    :return: List of law paragraphs.
    """

    # extract paragraphs from element
    paragraph_elements = article_content.find_all("div", {"class": "jurAbsatz"})
    if not paragraph_elements:
        paragraph_elements = article_content.find_all("span", {"class": "SP"})

    paragraphs = []
    for paragraph in paragraph_elements:
        paragraphs.append(paragraph.text)

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
        book_response = requests.get(book_base_url)
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
                        print("F")

                continue

            elif header.has_attr("colspan") and header['colspan'] == "2":
                continue

            elif header.find('a').has_attr("href"):
                link_element = header.find('a')
                href = link_element['href']

                # Make a GET request to the law text page
                article_response = requests.get(book_base_url + f"/{href}")

                # Parse the HTML content of the law text page
                article_soup = BeautifulSoup(article_response.text, 'html.parser')      # features="%(parser)s"
                article_title = link_element.text

                if "weggefallen" in article_title:
                    dict_iterator[article_title] = {}
                    dict_iterator[article_title]["texts"] = []
                    continue

                # save the text of the law text to a list
                article_content = article_soup.find("div", {"class": "jnhtml"})
                # book_content[chapter_title][article_title]["texts"] = handle_article_content(article_content)
                dict_iterator[article_title] = {}
                dict_iterator[article_title]["texts"] = handle_article_content(article_content)
                dict_iterator[article_title]["url"] = book_base_url + f"/{href}"

        book_file_path = os.path.join(paths.crawl_results_folder, f"{book}.json")
        dump_to_json(book_file_path, book_content)


if __name__ == "__main__":
    BOOKS = ["bgb", "gg", "zpo", "stpo"]#, "stgb"]
    # BOOKS = ["stgb"]

    download_book_content(BOOKS)