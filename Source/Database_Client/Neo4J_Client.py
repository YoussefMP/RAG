from neo4j import GraphDatabase


class DBNeo4JManager:
    """This class manages all operations to the Neo4J database.
     The queries are constructed using the cipher language.
     """
    def __init__(self, **kwargs):

        self.driver = GraphDatabase.driver(kwargs["uri"], auth=(kwargs["username"], kwargs["password"]))
        self.driver.verify_connectivity()

