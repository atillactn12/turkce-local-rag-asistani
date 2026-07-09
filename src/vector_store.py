"""Chunk ve embedding verilerini yerel SQLite dosyasında saklar."""

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path


class SQLiteVectorStore:
    """Projeye özel, küçük ve anlaşılır SQLite veri katmanı."""

    def __init__(self, db_path: str = "data/index/rag_index.sqlite") -> None:
        self.db_path = Path(db_path)

    def _connect(self) -> sqlite3.Connection:
        """Her işlem için yeni ve kısa ömürlü bir SQLite bağlantısı açar.

        Streamlit uygulamaları rerun/thread değiştirebildiği için bağlantıyı
        sınıf içinde uzun süre saklamak SQLite'ın "created in a thread" hatasına
        yol açabilir. Bu yüzden bağlantı her metotta açılır ve context manager
        tarafından otomatik kapatılır.
        """
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        return sqlite3.connect(self.db_path)

    def initialize(self) -> None:
        """Veritabanı klasörünü ve gerekli tabloları hazırlar."""
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS chunks (
                    chunk_id TEXT PRIMARY KEY,
                    file_name TEXT,
                    file_path TEXT,
                    page_number INTEGER,
                    text TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS embeddings (
                    chunk_id TEXT NOT NULL,
                    provider TEXT NOT NULL,
                    vector_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY (chunk_id, provider)
                )
                """
            )

    def clear(self) -> None:
        """Önceki aktif index verilerini siler, veritabanı dosyasını korur."""
        with self._connect() as connection:
            connection.execute("DELETE FROM embeddings")
            connection.execute("DELETE FROM chunks")

    def add_chunks(self, chunks: list[dict]) -> None:
        """Chunkları kaynak bilgileriyle SQLite'a kaydeder."""
        rows = [
            (
                chunk.get("chunk_id"),
                chunk.get("file_name"),
                chunk.get("file_path"),
                chunk.get("page_number"),
                chunk.get("text", ""),
            )
            for chunk in chunks
        ]
        with self._connect() as connection:
            connection.executemany(
                """
                INSERT OR REPLACE INTO chunks
                (chunk_id, file_name, file_path, page_number, text)
                VALUES (?, ?, ?, ?, ?)
                """,
                rows,
            )

    def get_all_chunks(self) -> list[dict]:
        with self._connect() as connection:
            cursor = connection.execute(
                "SELECT chunk_id, file_name, file_path, page_number, text FROM chunks"
            )
            return [
                {
                    "chunk_id": row[0],
                    "file_name": row[1],
                    "file_path": row[2],
                    "page_number": row[3],
                    "text": row[4],
                }
                for row in cursor.fetchall()
            ]

    def save_embedding(
        self,
        chunk_id: str,
        vector: list[float],
        provider: str,
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO embeddings
                (chunk_id, provider, vector_json, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (
                    chunk_id,
                    provider,
                    json.dumps(vector),
                    datetime.now(timezone.utc).isoformat(),
                ),
            )

    def get_embeddings(self, provider: str) -> list[dict]:
        with self._connect() as connection:
            cursor = connection.execute(
                """
                SELECT chunk_id, provider, vector_json, created_at
                FROM embeddings
                WHERE provider = ?
                """,
                (provider,),
            )
            return [
                {
                    "chunk_id": row[0],
                    "provider": row[1],
                    "vector": json.loads(row[2]),
                    "created_at": row[3],
                }
                for row in cursor.fetchall()
            ]

    def close(self) -> None:
        """Bağlantılar metot içinde kapandığı için geriye dönük uyum amaçlı no-op."""
        return None
