"""
Prediction History Database Module
Uses SQLite for persistent storage of prediction history.
"""

import sqlite3
import json
import os
from datetime import datetime

DB_PATH = os.getenv("PREDICTION_HISTORY_DB", "history/predictions.db")


def get_connection():
    """Get a database connection with row factory for dict access."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Initialize the predictions table if it doesn't exist."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            text TEXT NOT NULL,
            predicted_emotion TEXT NOT NULL,
            confidence REAL NOT NULL,
            all_probabilities TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()


def save_prediction(text, predicted_emotion, confidence, all_probabilities=None):
    """Save a prediction to the history database."""
    conn = get_connection()
    cursor = conn.cursor()
    probs_json = json.dumps(all_probabilities) if all_probabilities else None
    cursor.execute(
        """
        INSERT INTO predictions (text, predicted_emotion, confidence, all_probabilities)
        VALUES (?, ?, ?, ?)
        """,
        (text, predicted_emotion, confidence, probs_json),
    )
    conn.commit()
    conn.close()


def get_history(limit=50, offset=0):
    """Retrieve prediction history with optional limiting and pagination."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT id, text, predicted_emotion, confidence, all_probabilities, created_at
        FROM predictions
        ORDER BY created_at DESC
        LIMIT ? OFFSET ?
        """,
        (limit, offset),
    )
    rows = cursor.fetchall()
    conn.close()

    history = []
    for row in rows:
        entry = {
            "id": row["id"],
            "text": row["text"],
            "predicted_emotion": row["predicted_emotion"],
            "confidence": row["confidence"],
            "created_at": row["created_at"],
        }
        if row["all_probabilities"]:
            entry["all_probabilities"] = json.loads(row["all_probabilities"])
        history.append(entry)

    return history


def get_history_count():
    """Get total number of predictions in history."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM predictions")
    count = cursor.fetchone()[0]
    conn.close()
    return count


def cleanup_old_predictions(max_count=500):
    """Remove oldest predictions when history exceeds max count."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        DELETE FROM predictions
        WHERE id IN (
            SELECT id FROM predictions
            ORDER BY created_at ASC
            LIMIT (SELECT COUNT(*) FROM predictions) - ?
        )
        """,
        (max_count,),
    )
    conn.commit()
    conn.close()