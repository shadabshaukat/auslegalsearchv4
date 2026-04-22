"""
Oracle 23ai Database Connector for auslegalsearchv3.
- Provides a connector class for querying Oracle 23ai databases.
- Auth/credentials/config are loaded from environment variables or parameters.
- Results are returned as native Python objects.
- Follows a similar pattern to the Oracle reference agentic_rag repo.
"""

import os

try:
    import oracledb
except ImportError:
    oracledb = None  # For environments without oracledb installed

class Oracle23AIConnector:
    def __init__(
        self,
        user: str = None,
        password: str = None,
        dsn: str = None,
        wallet_location: str = None
    ):
        """
        Args:
            user (str): Oracle DB username
            password (str): Oracle DB password
            dsn (str): Oracle DB connection string
            wallet_location (str, optional): For Autonomous DBs w/ wallet, path to wallet zip
        """
        self.user = user or os.environ.get("ORACLE_DB_USER")
        self.password = password or os.environ.get("ORACLE_DB_PASSWORD")
        self.dsn = dsn or os.environ.get("ORACLE_DB_DSN")
        self.wallet_location = wallet_location or os.environ.get("ORACLE_WALLET_LOCATION")

        self.connection = self._init_connection()

    def _init_connection(self):
        if not oracledb:
            raise ImportError("oracledb (python-oracledb) package not installed. Please pip install oracledb")

        # With/without wallet
        connection_params = {
            "user": self.user,
            "password": self.password,
            "dsn": self.dsn
        }
        if self.wallet_location:
            os.environ["TNS_ADMIN"] = self.wallet_location

        try:
            conn = oracledb.connect(**connection_params)
            return conn
        except Exception as e:
            raise RuntimeError(f"Failed to connect to Oracle 23ai DB: {e}")

    def run_query(self, sql: str, params: tuple = ()):
        """
        Executes a SQL query and returns the results as a list of dicts (columns to values).
        """
        try:
            cursor = self.connection.cursor()
            cursor.execute(sql, params)
            columns = [col[0] for col in cursor.description]
            results = [dict(zip(columns, row)) for row in cursor.fetchall()]
            cursor.close()
            return results
        except Exception as e:
            return {"error": str(e)}

    def close(self):
        if self.connection:
            self.connection.close()
