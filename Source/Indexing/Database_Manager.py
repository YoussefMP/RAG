import psycopg2
from psycopg2 import sql


class DBManager:
    def __init__(self, dbname, user, password, host, port):
        # Establish a connection to your PostgreSQL database
        self.conn = psycopg2.connect(
                dbname=dbname,
                user=user,
                password=password,
                host=host,
                port=port
        )

        # Create a cursor object to execute SQL queries
        self.cursor = self.conn.cursor()

    @staticmethod
    def upsert_data():
        return True
