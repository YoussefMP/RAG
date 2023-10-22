class PageNode:
    def __init__(self, title, path, pid, content=""):
        self.title = title
        self.content = content
        self.path = path
        self.id = pid
        self.sub_pages = []  # List to store child pages

    def add_page(self, page):
        self.sub_pages.append(page)

    def display(self, level=0):
        # Display the page title and content with proper indentation
        print(" " * level + f"{self.title}: {self.content}")

        # Display child pages with increased indentation
        for page in self.sub_pages:
            page.display(level + 2)
