from selenium import webdriver

try:
    chrome_options = webdriver.ChromeOptions()
    # chrome_options.add_argument("--disable-application-cache")
    # chrome_options.add_argument("--disable-gpu-shader-disk-cache")
    # chrome_options.add_argument("--disable-local-storage")
    # chrome_options.add_argument("--disable-offline-load-stale-cache")
    # chrome_options.add_argument("--disable-session-crashed-bubble")
    # chrome_options.add_argument("--disable-tcmalloc")
    # chrome_options.add_argument("--disable-threaded-compositing")
    # chrome_options.add_argument("--disable-web-security")
    # chrome_options.add_argument("--disk-cache-size=0")
    # chrome_options.add_argument("--media-cache-size=0")
    # chrome_options.add_argument("--v8-cache-options=off")
    # Start a Selenium WebDriver (make sure you have installed a compatible driver for your browser)
    driver = webdriver.Chrome(options=chrome_options)
    # driver = webdriver.Chrome()
except KeyboardInterrupt:
    print("Program interrupted by user.")
