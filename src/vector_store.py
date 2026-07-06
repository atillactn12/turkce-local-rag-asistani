"""Chunk ve embedding verilerini yerel SQLite dosyasında saklar."""

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path


class SQLiteVectorStore:
    """Projeye özel, küçük ve anlaşılır SQLite veri katmanı."""

    def __init__(self, db_path: str = "data/index/rag_index.sqlite") -> None:
        self.db_path = Path(db_path)
        self.connection: sqlite3.Connection | None = None

    def initialize(self) -> None:
        """Veritabanı klasörünü ve gerekli tabloları hazırlar."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        if self.connection is None:
            self.connection = sqlite3.connect(self.db_path)

        self.connection.execute(
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
        self.connection.execute(
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
        self.connection.commit()

    def _get_connection(self) -> sqlite3.Connection:
        if self.connection is None:
            self.initialize()
        if self.connection is None:  # Type checker için güvenli son kontrol.
            raise RuntimeError("SQLite bağlantısı oluşturulamadı.")
        return self.connection

    def clear(self) -> None:
        """Önceki aktif index verilerini siler, veritabanı dosyasını korur."""
        connection = self._get_connection()
        connection.execute("DELETE FROM embeddings")
        connection.execute("DELETE FROM chunks")
        connection.commit()

    def add_chunks(self, chunks: list[dict]) -> None:
        """Chunkları kaynak bilgileriyle SQLite'a kaydeder."""
        connection = self._get_connection()
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
        connection.executemany(
            """
            INSERT OR REPLACE INTO chunks
            (chunk_id, file_name, file_path, page_number, text)
            VALUES (?, ?, ?, ?, ?)
            """,
            rows,
        )
        connection.commit()

    def get_all_chunks(self) -> list[dict]:
        connection = self._get_connection()
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
        connection = self._get_connection()
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
        connection.commit()

    def get_embeddings(self, provider: str) -> list[dict]:
        connection = self._get_connection()
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
        if self.connection is not None:
            self.connection.close()
            self.connection = None
