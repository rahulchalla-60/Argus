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

            # Option B: Latest risk score per account + history for significant changes
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS risk_scores (
                    account_id TEXT PRIMARY KEY,
                    risk_score REAL NOT NULL,
                    step INTEGER NOT NULL,
                    model_type TEXT,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS risk_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    account_id TEXT NOT NULL,
                    risk_score REAL NOT NULL,
                    step INTEGER NOT NULL,
                    model_type TEXT
                )
            """)
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_risk_hist_acc ON risk_history(account_id, step)")

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

    def save_risk_score(
        self,
        account_id: str,
        risk_score: float,
        step: int,
        model_type: str = "xgboost",
        threshold: float = 0.30,
        delta_threshold: float = 0.10
    ) -> bool:
        # Option B: Persist only if risk_score >= threshold or delta >= delta_threshold
        cursor = self.conn.cursor()
        cursor.execute("SELECT risk_score FROM risk_scores WHERE account_id = ?", (account_id,))
        row = cursor.fetchone()

        prev_score = row["risk_score"] if row else None
        should_save = False

        if risk_score >= threshold:
            should_save = True
        elif prev_score is not None and abs(risk_score - prev_score) >= delta_threshold:
            should_save = True

        if should_save:
            with self.conn:
                self.conn.execute("""
                    INSERT OR REPLACE INTO risk_scores (account_id, risk_score, step, model_type, updated_at)
                    VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
                """, (account_id, risk_score, step, model_type))

                self.conn.execute("""
                    INSERT INTO risk_history (account_id, risk_score, step, model_type)
                    VALUES (?, ?, ?, ?)
                """, (account_id, risk_score, step, model_type))

        return should_save

    def get_account_risk(self, account_id: str) -> dict[str, Any] | None:
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM risk_scores WHERE account_id = ?", (account_id,))
        row = cursor.fetchone()
        return dict(row) if row else None

    def get_high_risk_accounts(self, min_score: float = 0.80) -> list[dict[str, Any]]:
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT * FROM risk_scores
            WHERE risk_score >= ?
            ORDER BY risk_score DESC, step DESC
        """, (min_score,))
        return [dict(r) for r in cursor.fetchall()]

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
