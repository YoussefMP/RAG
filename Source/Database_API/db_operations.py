from Source.Logging.loggers import get_logger
import logging
import psycopg2
import pinecone
import os
import time


logger = get_logger("__DB__", "indexing.log")


class DbLocalManager:

    def __init__(self, name, user, password, host, port):
        # Connect with username and password
        self.conn = psycopg2.connect(
            dbname=name,
            user=user,
            password=password,
            host=host,
            port=port
        )

        self.cur = self.conn.cursor()

    def insert(self, vectors_data):
        try:
            self.cur.executemany("""
                INSERT INTO vectors_table (document_id, vector_column_name)
                VALUES (%s, %s)
            """, vectors_data)

            self.conn.cursor()
        except Exception as e:
            print("Log Exception")


class DBPineconeManager:
    def __init__(self, index_name, env, api_key, dimensions, metadata_config):
        # initialize connection to pinecone (get API key at app.pinecone.io)
        api_key = os.getenv("PINECONE_API_KEY") or api_key
        # find your environment next to the api key in pinecone console
        env = os.getenv("PINECONE_ENVIRONMENT") or env

        pinecone.init(api_key=api_key, environment=env)
        logger.info(f"\t{pinecone.whoami()}")

        self.index_name = index_name
        self.dimensions = dimensions
        self.metadata_config = metadata_config
        self.index = None

    def create_index(self,):
        logger.info("\tCreating index...")
        if self.index_name not in pinecone.list_indexes():
            pinecone.create_index(
                self.index_name,
                dimension=self.dimensions,
                metric='cosine',
                metadata_config=self.metadata_config
            )

            logger.info("\tWaiting for index to be initialized...")
            while not pinecone.describe_index(self.index_name).status['ready']:
                time.sleep(1)

        self.index = pinecone.Index(self.index_name)
        logger.info(f"\t{self.index.describe_index_stats()}")

    def upsert_batch_with_metadata(self, batch_data):

        cuids = batch_data[0]
        vectors = batch_data[1]
        metadata = batch_data[2]

        data = list(zip(cuids, vectors, metadata))

        logger.debug("\tUpserting Batch...")
        upsert_response = self.index.upsert(
            vectors=data
        )
        logger.debug(f"\t\tResponse: {upsert_response}")

