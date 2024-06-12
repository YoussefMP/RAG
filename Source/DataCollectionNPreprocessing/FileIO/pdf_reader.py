"""
This script is for extracting the text and tables from the pdf files.

08.11.2023
V 1.0.0.
"""
import os
os.environ['PATH'] += os.pathsep + r'C:\Program Files\Tesseract-OCR'

from Source.Logging.loggers import *
from Utils.paths import *
from pdf2image import convert_from_path
import pytesseract
import fitz

from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
from io import StringIO


logger = get_logger("pdf_reader", "jotr_crawler.log")


def extract_scanned_text(pdf_file_path, out_file_path):
    pages = convert_from_path(pdf_file_path, 550)  # 500 is the dpi resolution

    with open(out_file_path, "w", encoding="utf-8") as of:
        for pid in range(len(pages)):
            text = pytesseract.image_to_string(pages[pid], lang='ara')
            of.write(text.replace("\u200e", " ").replace("\u200f", " "))
            of.write(f"\n\n ### {pid} صفحة ### \n\n")
    of.close()


def extract_searchable_text(pdf_file_path, out_file_path):
    #
    # pdf_file = open(pdf_file_path, "rb")
    # pdf_reader = PyPDF2.PdfReader(pdf_file)
    #
    # with open(out_file_path, "w", encoding="utf-8") as of:
    #     for pid in range(len(pdf_reader.pages)):
    #         page = pdf_reader.pages[pid]
    #         text = page.extract_text()
    #
    #         of.write(text)
    #         of.write(f"\n\n ### {pid} صفحة ### \n\n")
    # of.close()
    # pdf_file.close()

    ###############################################################################
    rsrcmgr = PDFResourceManager()
    retstr = StringIO()
    codec = 'utf-8'
    laparams = LAParams()
    device = TextConverter(rsrcmgr, retstr, codec=codec, laparams=laparams)
    fp = open(pdf_file_path, 'rb')
    interpreter = PDFPageInterpreter(rsrcmgr, device)
    password = ""
    maxpages = 0
    caching = True
    pagenos = set()

    for page in PDFPage.get_pages(fp, pagenos, maxpages=maxpages, password=password, caching=caching,
                                  check_extractable=True):
        interpreter.process_page(page)

    text = retstr.getvalue()

    fp.close()
    device.close()
    retstr.close()
    print(text)
    input("GREGBR")
    ################################################################################
    # doc = fitz.open(pdf_file_path)
    # text = ""
    # for page_num in range(doc.page_count):
    #     page = doc[page_num]
    #     text += page.get_text()
    #     print(text)
    #     input("wait here ....")


def is_selectable_text(pdf_path):
    """
    Check if the pdf file under the given path is made of selectable text or is made of scanned pages
    :param pdf_path:
    :return:
    """
    try:
        doc = fitz.open(pdf_path)
        num_pages = len(doc)
        is_selectable = False  # Assume it's a scanned PDF by default

        for page_num in range(num_pages):
            page = doc[page_num]
            blocks = page.get_text("blocks")

            if len(blocks) > 0:
                is_selectable = True  # Found selectable text, so it's not a scanned PDF
                break

        doc.close()

        return is_selectable  # Return True for searchable text, False for scanned images

    except Exception as e:
        print(f"Error analyzing PDF: {e}")
        return False  # Error occurred, assuming it's not selectable text


def main(section, list_of_files):
    """

    :param section:
    :param list_of_files:
    :return:
    """
    logger.info(f"\t\t\tRegistering the \"{section}\" files => {list_of_files}")

    for file in list_of_files:

        pdf_file_path = os.path.join(jotr_documents_path, section, file)

        results_folder_path = os.path.join(jotr_documents_path, f"{section}_txt")
        os.makedirs(os.path.join(results_folder_path, file[:4]), exist_ok=True)

        out_txt_file = os.path.join(results_folder_path, file.replace(".pdf", "_extracted.txt"))

        if not os.path.exists(out_txt_file):
            logger.debug(f"\t\t\t\tExtracting text for {file} ")
            # if int(file[:4]) >= 2000:
            # extract_searchable_text(pdf_file_path, out_txt_file)
            # else:
            extract_scanned_text(pdf_file_path, out_txt_file)
        else:
            logger.debug(f"\t\t\t\tText for {file} is already extracted")

