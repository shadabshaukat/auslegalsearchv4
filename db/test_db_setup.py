"""
Test script to verify DB connection and create/check tables for auslegalsearchv2.
Run this file directly: python db/test_db_setup.py
It will:
- Print the database URL (DB_URL) in use.
- Create (if missing) the 'documents' and 'embeddings' tables per the SQLAlchemy models.
- List all tables present after setup.
"""
from sqlalchemy import inspect
from db.connector import engine
from db.store import create_all_tables

def main():
    print("Testing DB connection and table setup...")
    # This will print the DB_URL from connector.py at import
    create_all_tables()
    inspector = inspect(engine)
    print("Tables currently in your database:")
    for t in inspector.get_table_names():
        print(" -", t)

if __name__ == "__main__":
    main()
