"""
Oracle 26ai Database Connector for auslegalsearchv4.
- Provides a connector class for querying Oracle 26ai databases.
- Auth/credentials/config are loaded from environment variables or parameters.
- Results are returned as native Python objects.
"""

import os

try:
    import oracledb
except ImportError:
    oracledb = None


class Oracle26AIConnector:
    def __init__(
        self,
        user: str = None,
        password: str = None,
        dsn: str = None,
        wallet_location: str = None,
    ):
        self.user = user or os.environ.get("ORACLE_DB_USER")
        self.password = password or os.environ.get("ORACLE_DB_PASSWORD")
        self.dsn = dsn or os.environ.get("ORACLE_DB_DSN")
        self.wallet_location = wallet_location or os.environ.get("ORACLE_WALLET_LOCATION")
        self.connection = self._init_connection()

    def _init_connection(self):
        if not oracledb:
            raise ImportError("oracledb (python-oracledb) package not installed. Please pip install oracledb")
        connection_params = {
            "user": self.user,
            "password": self.password,
            "dsn": self.dsn,
        }
        if self.wallet_location:
            os.environ["TNS_ADMIN"] = self.wallet_location
        try:
            return oracledb.connect(**connection_params)
        except Exception as e:
            raise RuntimeError(f"Failed to connect to Oracle 26ai DB: {e}")

    def run_query(self, sql: str, params: tuple = ()):
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
