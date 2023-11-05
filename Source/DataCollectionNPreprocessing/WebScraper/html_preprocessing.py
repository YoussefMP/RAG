"""
Script for preprocessing html content.
Cleaning tag and structuring unstructured data.
"""
######################################################################################################
__DEBUG__ = False
######################################################################################################
from bs4 import BeautifulSoup
import re


def process_html_list(list_element: str) -> str:
    """
    This method structures lists in confluence pages.
    We send the header Or introductory sentence of the table and part of the list to an LLM and ask it to structure it
    in to a coherent text in a specific format.
    We then apply the received format to the rest of the list.
    :param list_element: HTML table element
    :return: cleaned text based on the table data
    """
    # TODO: Implement this
    return ""


def process_html_table(table: str) -> str:
    """
    This method structures tables in confluence pages.
    We send the columns of the table and the first row to an LLM and ask it to structure it in to a coherent text in
    a specific format.
    We then apply the received format to the rest of the rows.
    :param table: HTML table element
    :return: cleaned text based on the table data
    """
    # TODO: Implement this
    return ""


def extra_preprocessing(cleaned_text):
    cleaned_text = re.sub(r'\n+', '\n', cleaned_text)
    cleaned_text = cleaned_text.replace('\xa0', ' ')

    return cleaned_text


def process_html_content(content: str) -> str:

    soup = BeautifulSoup(content, 'html.parser')

    tags_to_keep = ['h1', 'h2', 'h3', 'p', 'ul', 'ol', 'li', 'table', 'tr', 'td']

    all_tags = soup.find_all()

    # Loop through all tags in the document
    for tag in all_tags:
        if tag.name in tags_to_keep:
            # Remove the names and IDs from the attributes of the tag
            for attr in list(tag.attrs):
                if attr not in ['href']:  # You can add more attributes to retain
                    del tag[attr]
        else:
            # Remove the <span> tags and keep their content
            if tag.name == 'span':
                tag.unwrap()

    # Get the human-readable text
    cleaned_text = soup.get_text()
    cleaned_text = extra_preprocessing(cleaned_text)

    return cleaned_text




# if __name__ == "__main__" and __DEBUG__:
    # file_path = ".\\..\\..\\Resources\\Confluence\\Erasmus+ and European Solidarity Corps guides\\Applicant Guides - Submission phase\\Apply for grant or accreditation.txt"
    # text = load_and_process_content(file_path)


