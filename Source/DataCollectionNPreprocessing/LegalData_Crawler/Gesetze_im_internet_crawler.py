import requests
from bs4 import BeautifulSoup

# Make a GET request to the website
url = "https://www.gesetze-im-internet.de/Teilliste_A.html"
response = requests.get(url)

# Parse the HTML content
soup = BeautifulSoup(response.text, 'html.parser')

# Find all the links in the page
links = soup.find_all('a', href=True)

# Extract the first 10 law texts
law_texts = []
for i, link in enumerate(links):
    if i >= 10:
        break

    href = link['href']

    if "/index.html" in href:
        # Make a GET request to the law text page
        law_response = requests.get(href)

        # Parse the HTML content of the law link
        law_soup = BeautifulSoup(law_response.text, 'html.parser')
        law_soup_links = law_soup.find_all('a', href=True)

        for _, link in law_soup_links:
            if "BNJR" in link['href'] and ".html" in link['href']:
                # law_text = requests.get(link['href'])
                # law_soup = BeautifulSoup(law_response.text, 'html.parser')
                # law_text = law_soup.find('div', class_='gesetzestext').text
                # law_texts.append(law_text)
                break


        # Extract the law text content
        law_text = law_soup.find('div', class_='gesetzestext').text
        law_texts.append(law_text)
    else:
        continue

# Print the first 10 law texts
for i, law_text in enumerate(law_texts):
    print(f"Law Text {i+1}:")
    print(law_text)