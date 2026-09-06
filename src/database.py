import sqlite3
from pathlib import Path
from typing import Generator, Any
import pandas as pd


class Database:
    def __init__(self, db_path: str | Path = ":memory:"):
        self.db_path = str(db_path)
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        self._init_db()

    def _init_db(self):
        # ponytail: WAL + NORMAL synchronous for fast writes and crash safety
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.execute("PRAGMA temp_store=MEMORY")
        with self.conn:
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS transactions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    step INTEGER NOT NULL,
                    type TEXT NOT NULL,
                    amount REAL NOT NULL,
                    nameOrig TEXT NOT NULL,
                    oldbalanceOrg REAL DEFAULT 0.0,
                    newbalanceOrig REAL DEFAULT 0.0,
                    nameDest TEXT NOT NULL,
                    oldbalanceDest REAL DEFAULT 0.0,
                    newbalanceDest REAL DEFAULT 0.0,
                    isFraud INTEGER DEFAULT 0,
                    isFlaggedFraud INTEGER DEFAULT 0
                )
            """)
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_orig_step ON transactions(nameOrig, step)")
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_dest_step ON transactions(nameDest, step)")
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_step ON transactions(step)")

    def insert_transaction(
        self,
        step: int,
        type_: str,
        amount: float,
        name_orig: str,
        name_dest: str,
        oldbalance_org: float = 0.0,
        newbalance_orig: float = 0.0,
        oldbalance_dest: float = 0.0,
        newbalance_dest: float = 0.0,
        is_fraud: int = 0,
        is_flagged_fraud: int = 0
    ):
        with self.conn:
            self.conn.execute("""
                INSERT INTO transactions (
                    step, type, amount, nameOrig, oldbalanceOrg, newbalanceOrig,
                    nameDest, oldbalanceDest, newbalanceDest, isFraud, isFlaggedFraud
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                step, type_, amount, name_orig, oldbalance_org, newbalance_orig,
                name_dest, oldbalance_dest, newbalance_dest, is_fraud, is_flagged_fraud
            ))

    def insert_batch(self, rows: list[tuple]):
        with self.conn:
            self.conn.executemany("""
                INSERT INTO transactions (
                    step, type, amount, nameOrig, oldbalanceOrg, newbalanceOrig,
                    nameDest, oldbalanceDest, newbalanceDest, isFraud, isFlaggedFraud
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, rows)

    def ingest_csv(self, csv_path: str | Path, chunksize: int = 50000, max_rows: int | None = None) -> int:
        total_inserted = 0
        for chunk in pd.read_csv(csv_path, chunksize=chunksize, nrows=max_rows):
            rows = [
                (
                    int(r.step),
                    str(r.type),
                    float(r.amount),
                    str(r.nameOrig),
                    float(getattr(r, "oldbalanceOrg", 0.0)),
                    float(getattr(r, "newbalanceOrig", 0.0)),
                    str(r.nameDest),
                    float(getattr(r, "oldbalanceDest", 0.0)),
                    float(getattr(r, "newbalanceDest", 0.0)),
                    int(getattr(r, "isFraud", 0)),
                    int(getattr(r, "isFlaggedFraud", 0))
                )
                for r in chunk.itertuples(index=False)
            ]
            self.insert_batch(rows)
            total_inserted += len(rows)
        return total_inserted

    def get_account_history(self, account_id: str, limit: int | None = None) -> list[dict[str, Any]]:
        cursor = self.conn.cursor()
        query = """
            SELECT * FROM transactions
            WHERE nameOrig = ? OR nameDest = ?
            ORDER BY step DESC, id DESC
        """
        params: list[Any] = [account_id, account_id]
        if limit is not None:
            query += " LIMIT ?"
            params.append(limit)
        cursor.execute(query, params)
        return [dict(row) for row in cursor.fetchall()]

    def get_recent_transactions(self, limit: int = 50000) -> list[dict[str, Any]]:
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT * FROM (
                SELECT * FROM transactions
                ORDER BY step DESC, id DESC
                LIMIT ?
            ) ORDER BY step ASC, id ASC
        """, (limit,))
        return [dict(row) for row in cursor.fetchall()]

    def stream_transactions(self, batch_size: int = 10000) -> Generator[list[dict[str, Any]], None, None]:
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM transactions ORDER BY step ASC, id ASC")
        while True:
            rows = cursor.fetchmany(batch_size)
            if not rows:
                break
            yield [dict(r) for r in rows]

    def count(self) -> int:
        cursor = self.conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM transactions")
        return cursor.fetchone()[0]

    def close(self):
        try:
            self.conn.close()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __del__(self):
        self.close()
