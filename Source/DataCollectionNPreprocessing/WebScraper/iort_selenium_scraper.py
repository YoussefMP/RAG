import os.path

import selenium
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import Select
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common import exceptions as selenium_exceptions
import json
from Source.paths import *
from Source.Logging import loggers
from time import sleep

logger = loggers.get_logger("iort_logger", "iort_scraper.log")

# Set the path to your web driver executable
driver_path = '/path/to/chromedriver'

# Initialize the browser
chrome_options = webdriver.ChromeOptions()
driver = webdriver.Chrome(options=chrome_options)
search_url = 'http://www.iort.gov.tn/'

types = ["قانون", "مرسوم", "أمر", "قرار", "رأي"]


def update_start_record(s_type, s_year):
    with open(os.path.join(web_config_path, "start_rec.txt"), "w") as srf:
        srf.write(f"{s_type}-{s_year}")
    srf.close()


def get_start_record():
    """
    This return the record at which the retrieval of the webpages should start at
    :return: int : the start record
    """
    srf = os.path.join(web_config_path, "start_rec.txt")
    if os.path.exists(srf):
        f = open(srf, "r")
        try:
            start_record = f.read()
            s_type, s_year = int(start_record.split("-")[0]), int(start_record.split("-")[1])
        except IndexError or ValueError:
            return 2, 2
        return s_type, s_year
    else:
        return 2, 2


def refresh_link():
    driver.get("http://www.iort.gov.tn/")
    navigation = driver.find_element("name", "M7")
    navigation.click()

    # Wait for the search results to load (you may need to adjust the wait time)
    driver.implicitly_wait(10)

    navigation = driver.find_element("name", "A31")
    navigation.click()

    # Wait for the search results to load (you may need to adjust the wait time)
    driver.implicitly_wait(10)


def write_data(doc_type, file_name, data):
    # Saving the returned data to the file
    file_folder = os.path.join(jotr_documents_path, doc_type)
    os.makedirs(file_folder, exist_ok=True)
    with open(os.path.join(file_folder, file_name), 'w', encoding='utf-8') as json_file:
        json.dump(data, json_file, ensure_ascii=False, indent=2)
    json_file.close()


def load_next_batch(current_page, jumpto=None):

    retries = 0
    while retries < 2:
        try:
            element_a6 = driver.find_element(By.ID, 'A6')
        except selenium.common.exceptions.NoSuchElementException:
            wait = WebDriverWait(driver, 10)
            element_a6 = wait.until(EC.presence_of_element_located((By.ID, 'A6')))
        except selenium.common.exceptions.TimeoutException:
            logger.info("Waiting after TimeOut Exception, and retrying")
            sleep(1)
        retries += 1

    pages = element_a6.find_elements(By.TAG_NAME, "a")

    for page in pages:

        if jumpto:
            index = str(jumpto * 20)
            new_url = page.get_attribute("href")[:-2] + str(index)
            driver.get(new_url)
            driver.implicitly_wait(10)
            return

        if page.text.isdigit() and int(page.text) == current_page+1:
            driver.get(page.get_attribute("href"))
            # page.click()
            driver.implicitly_wait(10)
            return


def extract_meta_data(title):
    fields_map = {"A10": "text_title", "A14": "text_year", "A9": "text_nb", "A13": "date",
                  "A6": "journal_date", "A5": "journal_nb", "A12": "type"}

    try:
        meta_data = {"text_title": driver.find_element("id", "A10").text}
        for element_name in list(fields_map.keys())[1:]:
            field = driver.find_element("id", element_name)
            meta_data[fields_map[element_name]] = field.get_attribute("value")
        return meta_data
    except selenium.common.exceptions.NoSuchElementException:
        (logger.error(f"Error extracting metadata => {title}"))
        return None
    except selenium.common.exceptions.WebDriverException:
        (logger.error(f"Error extracting metadata => {title}"))
        return None
    except Exception as e:
        (logger.error(f"Error extracting metadata => {title}"))
        return None


def handle_content(title):
    driver.switch_to.window(driver.window_handles[1])               # Switch to the new tab
    wait = WebDriverWait(driver, 10)

    meta_data = {}

    try:
        meta_data = extract_meta_data(title)                 # Extract metadata
        tzA21_element = wait.until(EC.presence_of_element_located((By.ID, 'tzA21')))
    except (selenium.common.exceptions.TimeoutException or selenium.common.exceptions.StaleElementReferenceException
            or selenium.common.exceptions.NoSuchElementException):
        driver.close()
        driver.switch_to.window(driver.window_handles[0])
        if meta_data:
            logger.exception(f"Could not load text of {meta_data['title']}")
            return "", meta_data
        else:
            (logger.error(f"Error extracting metadata => {title}"))
            return None, None

    html_text = tzA21_element.get_attribute('innerHTML')            # Get the inner HTML of the element
    soup = BeautifulSoup(html_text, 'html.parser')          # Parse the HTML content
    consolidated_text = soup.get_text(separator=' ', strip=True)    # Extract text without HTML tags

    driver.close()
    driver.switch_to.window(driver.window_handles[0])
    return consolidated_text, meta_data


def parse_year_docs(nb_results, year, typeId):
    year_data = {year: []}
    processed_batches = 0
    results_exist = True

    while results_exist:
        # Find and click on the element that triggers the dynamic content
        elements = driver.find_elements("name", "A2")

        if processed_batches % 50 == 0:
            if os.path.exists(
                    os.path.join(jotr_documents_path, types[typeId], f'{year}_{int((processed_batches / 50)+1)}.json')
            ):
                logger.info("\t Found file of the records for the next 50 years")
                processed_batches += 50
                load_next_batch(processed_batches)
                continue
            if processed_batches > 0:
                write_data(types[typeId], f'{year}_{int(processed_batches / 50)}.json', year_data)
                year_data = {year: []}

        for element in elements:
            try:
                element.click()
            except selenium.common.exceptions.StaleElementReferenceException:
                logger.error(f"Error parsing docs of year {year} after processing {processed_batches * 20}")
            try:
                text, meta_data = handle_content(element.text)
                year_data[year].append({"text": text, "metadata": meta_data})
            except (selenium.common.exceptions.StaleElementReferenceException
                    or selenium.common.exceptions.NoSuchElementException
                    or selenium.common.exceptions.TimeoutException):
                logger.error(f"Error While handling content of {element.text}")
                logger.exception("Stack: ")
        processed_batches += 1
        logger.info(f"processed page nb ====> {processed_batches}")

        if nb_results <= 20 or processed_batches*20 > nb_results:
            results_exist = False

        elif processed_batches*20 < nb_results:
            load_next_batch(processed_batches)

    return year_data


def start_parsing():

    years = 18
    retry = 0

    s_type, s_year = get_start_record()

    for a_type in range(s_type, len(types)+2):
        logger.info(f"Start the parising of files of type {types[a_type-2]}")

        if s_year == years:
            s_year = 2

        for year in range(s_year, years+1):

            # Update start record so on next iteration we don't start from scratch
            update_start_record(a_type, year)

            logger.info(f"\tStart the parising of files of year {2025-year}")
            while retry < 3:
                try:
                    # Find the dropdown (select) element and select an option
                    dropdown = Select(driver.find_element(By.ID, 'A9'))  # Replace with the actual ID of the dropdown
                    dropdown.select_by_value(str(a_type))  # Replace '2' with the value of the option you want to select
                    break
                except selenium_exceptions.NoSuchElementException:
                    refresh_link()
                    retry += 1

            # Find the dropdown (select) element and select an option
            dropdown = Select(driver.find_element(By.ID, 'A8'))  # Replace with the actual ID of the dropdown
            dropdown.select_by_value(str(year))  # Replace '2' with the value of the option you want to select

            # Find and click the search button
            search_button = driver.find_element(By.NAME, 'A40')  # Replace with the actual name of the button
            search_button.click()

            # Wait for the search results to load (you may need to adjust the wait time)
            driver.implicitly_wait(10)

            # getting the number of results returned from the search
            start_element = driver.find_element(By.CLASS_NAME, 'l-1')
            input_element = start_element.find_element(By.XPATH, './following-sibling::td/input[@type="TEXT"]')
            input_value = input_element.get_attribute('value')

            data = parse_year_docs(int(input_value), 2025-year, a_type-2)
            write_data(types[a_type - 2], f'{str(2025 - year)}.json', data)

            refresh_link()


if __name__ == "__main__":
    logger.info("Let the spider loose !!!.....")
    driver.get(search_url)    # Navigate to the website

    start_parsing()
