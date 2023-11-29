import psycopg2


class DbManager:

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


