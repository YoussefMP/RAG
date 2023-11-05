"""

V 1.0 > 21.10.2023
"""
#########################################################################################################
__DEBUG__ = True

#########################################################################################################
from Source.DataCollectionNPreprocessing.WebScraper.html_preprocessing import process_html_content
from Source.DataCollectionNPreprocessing.IO.io_operations import save_node_as_document
from space_model_handler import PageNode
from atlassian import Confluence
from tqdm import tqdm
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

        # getting the total number of pages
        self.total_pages = len(self.confluence.get_all_pages_from_space_trash(space_name,
                                                                          limit=500,
                                                                          status="current",
                                                                          content_type="page"
                                                                          )
                               )

        # Preparing the skeleton for downloading the confluence space
        resources_folder = ".\\..\\..\\Resources"
        self.confluence_folder = resources_folder + "\\Confluence"
        # TODO: if Space is None iterate all Spaces

        os.makedirs(resources_folder, exist_ok=True)
        os.makedirs(self.confluence_folder, exist_ok=True)

    def get_root_page_id_by_space_key(self, space_key: str = None) -> str:
        """
        :param space_key: Identifier of space
        :return: ID of the root page
        """
        space = self.confluence.get_space(space_key, expand='homepage')
        return space['homepage']['id']

    def parse_confluence_space(self, root_page_id: str, pbar: tqdm) -> PageNode:
        """
        Parses the confluence space as a tree. The algorithm used is In-Order Traversal.
        :param root_page_id:
        :param pbar:
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
        )

        content = process_html_content(content)
        save_node_as_document(self.confluence_folder, title, content)

        # Function to recursively build the tree
        def build_tree(page_id, parent_page, pbar):
            # TODO: Implement login to skip page if path exists
            children = self.confluence.get_page_child_by_type(page_id, type="page")

            for child in children:

                child_page = self.confluence.get_page_by_id(page_id=child['id'], expand="body.view")
                child_title = child_page['title']
                child_content = child_page['body']['view']['value']
                child_path = parent_page.path + f"/{child_title}"

                # Create a new PageNode for the child
                child_node = PageNode(
                    title=child_title,
                    pid=child["id"],
                    path=child_path
                )

                # Recursively build the tree for this child
                build_tree(child["id"], child_node, pbar)
                child_content = process_html_content(child_content)
                save_node_as_document(self.confluence_folder, child_path, child_content)
                parent_page.add_page(child_node)
                pbar.update(1)

        # Start building the tree from the root
        build_tree(root_page_id, root_page, pbar)
        return root_page


def main(email, api_token, url, space=None):

    conf_man = ConfluenceManager(email, api_token, url, space)
    root_page_id = conf_man.get_root_page_id_by_space_key(space)

    with tqdm(total=conf_man.total_pages, unit=' nodes', unit_scale=True) as progress_bar:
        tree = conf_man.parse_confluence_space(root_page_id, progress_bar)


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
        with open("../../0_conf/Conf_api_token.txt", "r", encoding="utf-8") as token_file:
            at = token_file.readline()

        u = "https://rag-1-prototype.atlassian.net/wiki"
        skey = "RAGPrototy"
        u = 'https://wikis.ec.europa.eu'
        skey = "NAITDOC"

        main(e, at, u, skey)
