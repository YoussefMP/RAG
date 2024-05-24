from Source.DataCollectionNPreprocessing.FileIO.io_operations import dump_to_json
import requests
from Source.Logging.loggers import get_logger
from bs4 import BeautifulSoup
from Source import paths
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

    logger.info("Starting download of books")
    for book in books:
        logger.info(f"Starting download of book {book}")
        # Make a GET request to the book page
        book_base_url = BASE_URL + book
        book_response = requests.get(book_base_url)
        # Parse the HTML content
        soup = BeautifulSoup(book_response.text, 'html.parser')

        # Find all the td elements that have a child a with a href attribute
        main_element = soup.find('div', {'id': 'paddingLR12'})
        headers = main_element.find_all('td')

        chapter_count = 1
        chapter_title = "Begin"

        book_content = {"Begin": {}}

        for i, header in enumerate(headers):

            if not header.find('a') or (header.has_attr("colspan") and header['colspan'] == "3"):
                continue

            elif header.has_attr("colspan") and header['colspan'] == "2":
                logger.info(f"\tFinished download of chapter {chapter_title}")
                chapter_title = header.find('b').text
                book_content[chapter_title] = {}
                chapter_count += 1
                logger.info(f"\tStarting download of chapter {i} - {chapter_title}")
                continue

            elif header.find('a').has_attr("href"):
                link_element = header.find('a')
                href = link_element['href']

                # Make a GET request to the law text page
                article_response = requests.get(book_base_url + f"/{href}")
                book_content[chapter_title][article_title]["url"] = book_base_url + f"/{href}"

                # Parse the HTML content of the law text page
                article_soup = BeautifulSoup(article_response.text, 'html.parser')      # features="%(parser)s"
                article_title = link_element.text

                if "weggefallen" in article_title:
                    book_content[chapter_title][article_title]["texts"] = []
                    continue

                # save the text of the law text to a list
                article_content = article_soup.find("div", {"class": "jnhtml"})
                book_content[chapter_title][article_title]["texts"] = handle_article_content(article_content)

        book_file_path = os.path.join(paths.german_law_books, f"{book}.json")
        dump_to_json(book_file_path, book_content)


if __name__ == "__main__":
    BOOKS = ["gg", "bgb", "zpo", "stpo", "stgb"]

    download_book_content(BOOKS)

