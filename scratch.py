import psycopg2
from psycopg2 import sql, ProgrammingError
from pgvector.psycopg2 import register_vector
from config import DB_DSN

def main():
    conn = psycopg2.connect(DB_DSN)

    # For CREATE EXTENSION, autocommit is often safer
    conn.autocommit = True

    try:
        with conn.cursor() as cur:
            # 1) Ensure pgvector / vector type exists
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            print("CREATE EXTENSION vector; executed (or already exists).")

        # 2) Now that the type exists, register it with pgvector
        register_vector(conn)
        print("pgvector: vector type registered successfully.")

        # 3) Any normal queries you want
        with conn.cursor() as cur:
            cur.execute("SELECT version();")
            print("Postgres version:", cur.fetchone()[0])

            cur.execute("SELECT current_database();")
            print("Database:", cur.fetchone()[0])

            cur.execute("SELECT 'ok'::text;")
            print("pgvector extension ready:", cur.fetchone()[0])

    except ProgrammingError as e:
        print("ProgrammingError:", e)
        print(
            "Hint: you may not have permission to CREATE EXTENSION.\n"
            "Try running `CREATE EXTENSION IF NOT EXISTS vector;` as a superuser "
            "in this same database, or enable pgvector via your cloud provider’s UI."
        )
    finally:
        conn.close()

if __name__ == "__main__":
    main()
