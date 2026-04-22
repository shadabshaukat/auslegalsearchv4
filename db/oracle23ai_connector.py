"""Backward-compat wrapper for legacy Oracle 23ai module path.

Canonical connector now lives in db.oracle26ai_connector.
"""

from db.oracle26ai_connector import Oracle26AIConnector

# Backward-compatible class name and module path
Oracle23AIConnector = Oracle26AIConnector
