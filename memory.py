
import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parent / "data" / "agent.db"
DB_PATH.parent.mkdir(parents=True, exist_ok=True)


def initialize():
    with sqlite3.connect(DB_PATH) as db:
        db.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                fact TEXT UNIQUE NOT NULL
            )
        """)


def add_memory(fact):
    with sqlite3.connect(DB_PATH) as db:
        db.execute(
            "INSERT OR IGNORE INTO memories (fact) VALUES (?)",
            (fact,),
        )


def get_memories():
    with sqlite3.connect(DB_PATH) as db:
        return [
            row[0]
            for row in db.execute(
                "SELECT fact FROM memories ORDER BY id"
            ).fetchall()
        ]
