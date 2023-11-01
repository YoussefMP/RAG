__DEBUG__ = True


from Source.DataCollectionNPreprocessing.Preprocessing.html_preprocessing import extra_preprocessing
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.by import By
from selenium import webdriver
from bs4 import BeautifulSoup
import requests
import argparse
import json


try:
    chrome_options = webdriver.ChromeOptions()
    chrome_options.add_argument("--disable-application-cache")
    chrome_options.add_argument("--disable-gpu-shader-disk-cache")
    # chrome_options.add_argument("--disable-local-storage")
    chrome_options.add_argument("--disable-offline-load-stale-cache")
    chrome_options.add_argument("--disable-session-crashed-bubble")
    chrome_options.add_argument("--disable-tcmalloc")
    chrome_options.add_argument("--disable-threaded-compositing")
    chrome_options.add_argument("--disable-web-security")
    chrome_options.add_argument("--disk-cache-size=0")
    chrome_options.add_argument("--media-cache-size=0")
    chrome_options.add_argument("--v8-cache-options=off")
    # Start a Selenium WebDriver (make sure you have installed a compatible driver for your browser)
    driver = webdriver.Chrome(options=chrome_options)
    # driver = webdriver.Chrome()
except KeyboardInterrupt:
    print("Program interrupted by user.")


def get_static_page_content(link: str) -> (str, str, str):
    """
    :param link: link to the articles page
    :return:
    """
    response = requests.get(link)
    soup = BeautifulSoup(response.text, "html.parser")
    script_element = soup.find("script", {"id": "__NEXT_DATA__"})

    if script_element:
        # Extract the JSON data from the script element
        json_data = json.loads(script_element.string)

        # For example, if 'content' is a field within the JSON data, you can access it like this:
        pagedata = json_data["props"]["pageProps"]["pagedata"]

        title = pagedata["article"]["title"]
        date = pagedata["article"]["startPublish"]["date"]
        content = pagedata["article"]["description"]

        # 'content' will now contain the content you want from the script element
        content = BeautifulSoup(content, "html.parser").get_text()
    else:
        print("Script element with ID '__NEXT_DATA__' not found on the article page.")

    return title, date, content


def mosaique_crawler(base_url, wc):
    for section in wc["sections"]:
        tag, link = list(section.items())[0]
        driver.get(base_url + link)

        # Wait for the dynamic content to load (you may need to adjust the timeout)
        wait = WebDriverWait(driver, 10)
        wait.until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, "figure")))

        # Once the dynamic content has loaded, you can retrieve the page source
        page_source = driver.page_source

        # Now, you can use Beautiful Soup to parse the page source
        soup = BeautifulSoup(page_source, 'html.parser')

        # Find the div elements with the specified class
        section = soup.find('section')
        articles = section.find_all("div", class_="row")[1]

        # Extract the article links from 'a' elements within the 'figure' elements
        article_links = []
        for article in articles.children:
            a = article.find('figure').find("a")
            if a and 'href' in a.attrs:
                article_links.append(a['href'])

                title, date, content  = get_static_page_content(base_url + a['href'])

                with open(f".\\article_{len(article_links)}.txt", "w", encoding="utf-8") as of:
                    of.write(title + "\n")
                    of.write(date + "\n")
                    of.write(extra_preprocessing(content))
                of.close()
            break
        break

    # Don't forget to close the WebDriver when you're done
    driver.quit()


def main(file_path):
    web_config_file = open(file_path, "r", encoding="utf-")
    wc = json.load(web_config_file)
    base_url = wc["base_url"]
    mosaique_crawler(base_url, wc)


if __name__ == "__main__":
    if __DEBUG__:
        main(".\\web_configs\\mosaique.json")
    else:
        parser = argparse.ArgumentParser(description="Web Crawler.")
        parser.add_argument("--file_path", type=str, help="path to the website config_file")
        args = parser.parse_args()
        main(args.file_path)

