"""

V 1.0 > 21.10.2023
"""
#########################################################################################################
__DEBUG__ = True
#########################################################################################################
from space_model_handler import PageNode
from atlassian import Confluence
from bs4 import BeautifulSoup
import argparse
import os


class ConfluenceManager:

    def __init__(self, user: str, token: str, base_url: str, space_name: str = None) -> None:
        # Setting the confluence API - "user"
        self.confluence = Confluence(
            url=base_url,
            username=user,
            password=token,
            cloud=True)

        # Preparing the skeleton for downloading the confluence space
        resources_folder = ".\\..\\..\\Resources"
        self.confluence_folder = resources_folder + "\\Confluence"
        # TODO: if Space is None iterate all Spaces

        if not os.path.exists(resources_folder):
            os.makedirs(resources_folder)
        if not os.path.exists(self.confluence_folder):
            os.makedirs(self.confluence_folder)

    def get_root_page_id_by_space_key(self, space_key: str = None) -> str:
        """
        :param space_key: Identifier of space
        :return: ID of the root page
        """
        space = self.confluence.get_space(space_key, expand='homepage')
        return space['homepage']['id']

    def create_page_path(self, path, content):
        # TODO: Clean Folder names
        if len(path.split("/")) > 1:
            page_path = self.confluence_folder + "\\" + path.replace("/", "\\")
            if content:
                os.makedirs(page_path.rsplit("\\", 1)[0], exist_ok=True)
                self.write_page_content(page_path, content)
        else:
            if not os.path.exists(self.confluence_folder + "\\" + path):
                os.makedirs(self.confluence_folder + "\\" + path)

    @staticmethod
    def write_page_content(page_path, content):
        parent_folder, title = page_path.rsplit("\\", 1)
        with open(f"{parent_folder}\\{title}.txt", "w", encoding="utf-8") as p_file:
            p_file.write(content)
        p_file.close()

    def parse_confluence_space(self, root_page_id: str) -> PageNode:
        """
        Parses the confluence space as a tree. The algorithm used is In-Order Traversal.
        :param root_page_id:
        :return: root page
        """
        root_page = self.confluence.get_page_by_id(page_id=root_page_id, expand="body.view")
        # Create a root PageNode for the Confluence space
        title = root_page["title"]
        content = root_page['body']['view']['value']
        root_page = PageNode(
            title=title,
            path=title,
            pid=root_page_id,
            content=content
        )

        self.create_page_path(title, content)

        # Function to recursively build the tree
        def build_tree(page_id, parent_page):
            children = self.confluence.get_page_child_by_type(page_id, type="page")

            for child in children:

                child_page = self.confluence.get_page_by_id(page_id=child['id'], expand="body.view")
                child_title = child_page['title']
                child_content = child_page['body']['view']['value']
                child_path = parent_page.path + f"/{child_title}"

                # Create a new PageNode for the child
                child_node = PageNode(
                    title=child_title,
                    content=child_content,
                    pid=child["id"],
                    path=child_path
                )

                # Recursively build the tree for this child
                build_tree(child["id"], child_node)
                self.create_page_path(child_path, child_content)
                parent_page.add_page(child_node)

        # Start building the tree from the root
        build_tree(root_page_id, root_page)
        return root_page


def main(email, api_token, url, space=None):

    conf_man = ConfluenceManager(email, api_token, url, space)
    root_page_id = conf_man.get_root_page_id_by_space_key(space)
    tree = conf_man.parse_confluence_space(root_page_id)


if __name__ == "__main__":
    if not __DEBUG__:
        parser = argparse.ArgumentParser(description="Script that accepts API token, email, and URL.")
        parser.add_argument("api_token", type=str, help="Your API token")
        parser.add_argument("email", type=str, help="Your email address")
        parser.add_argument("url", type=str, help="URL to process")
        parser.add_argument("spaceKey", type=str, help="Key of the space we want to retrieve", required=False)

        args = parser.parse_args()
        main(args.api_token, args.email, args.url)
    else:

        e = "y.ameur.mail@gmail.com"
        # at = os.getenv("CONF_TOKEN")
        with open("./../_conf/Conf_api_token.txt", "r", encoding="utf-8") as token_file:
            at = token_file.readline()


        u = 'https://wikis.ec.europa.eu'
        skey = "NAITDOC"

        main(e, at, u, skey)
