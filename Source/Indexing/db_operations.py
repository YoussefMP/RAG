import psycopg2


class VecBase:
    def __init__(self, name, host):
        self.name = name
        self.host = host

    def create_tables(self):
        conn = psycopg2.connect(
            dbname=self.name,
            host=self.host
        )
        cur = conn.cursor()

        # Define your table creation SQL statements here

        conn.commit()
        cur.close()
        conn.close()

    def insert_metadata(self, title, link):
        conn = psycopg2.connect(
            dbname=self.name,
            host=self.host
        )
        cur = conn.cursor()

        # Define your metadata insertion SQL statement here
        cur.execute("""
            INSERT INTO metadata_table (title, link)
            VALUES (%s, %s) RETURNING document_id;
        """, (title, link))

        document_id = cur.fetchone()[0]

        conn.commit()
        cur.close()
        conn.close()

        return document_id

    def insert_chunk(self, vector, document_id):
        conn = psycopg2.connect(
            dbname=self.host,
            host=self.host
        )
        cur = conn.cursor()

        # Define your chunk insertion SQL statement here

        conn.commit()
        cur.close()
        conn.close()

