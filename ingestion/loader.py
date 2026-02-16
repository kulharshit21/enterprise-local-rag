"""
Document loaders for various file formats.
Supports PDF, Markdown, CSV, and HTML documents.
"""

import os
import csv
import uuid
from typing import List, Dict, Any
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class Document:
    """Represents a loaded document with metadata."""

    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    doc_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    source_file: str = ""
    doc_type: str = ""


class PDFLoader:
    """Load text and tables from PDF files using pdfplumber."""

    def load(
        self,
        file_path: str,
        role_access: List[str] = None,
        sensitivity: str = "internal",
    ) -> Document:
        import pdfplumber

        role_access = role_access or ["viewer"]
        pages_text = []

        with pdfplumber.open(file_path) as pdf:
            for i, page in enumerate(pdf.pages):
                text = page.extract_text() or ""
                pages_text.append(text)

                # Extract tables as structured text
                tables = page.extract_tables()
                for table in tables:
                    if table:
                        table_text = self._format_table(table)
                        pages_text.append(f"\n[TABLE]\n{table_text}\n[/TABLE]\n")

        content = "\n\n".join(pages_text)
        return Document(
            content=content,
            metadata={
                "role_access": role_access,
                "sensitivity": sensitivity,
                "file_type": "pdf",
                "page_count": len(pages_text),
                "timestamp": datetime.utcnow().isoformat(),
            },
            source_file=file_path,
            doc_type="pdf",
        )

    @staticmethod
    def _format_table(table: List[List]) -> str:
        """Format a table as readable text."""
        rows = []
        for row in table:
            cleaned = [str(cell).strip() if cell else "" for cell in row]
            rows.append(" | ".join(cleaned))
        return "\n".join(rows)


class MarkdownLoader:
    """Load content from Markdown files."""

    def load(
        self,
        file_path: str,
        role_access: List[str] = None,
        sensitivity: str = "internal",
    ) -> Document:
        role_access = role_access or ["viewer"]

        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        return Document(
            content=content,
            metadata={
                "role_access": role_access,
                "sensitivity": sensitivity,
                "file_type": "markdown",
                "timestamp": datetime.utcnow().isoformat(),
            },
            source_file=file_path,
            doc_type="markdown",
        )


class CSVLoader:
    """Load structured data from CSV files."""

    def load(
        self,
        file_path: str,
        role_access: List[str] = None,
        sensitivity: str = "internal",
    ) -> Document:
        role_access = role_access or ["viewer"]
        rows = []

        with open(file_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames or []

            for row in reader:
                row_text = " | ".join(f"{k}: {v}" for k, v in row.items() if v)
                rows.append(row_text)

        content = f"Headers: {', '.join(headers)}\n\n" + "\n".join(rows)
        return Document(
            content=content,
            metadata={
                "role_access": role_access,
                "sensitivity": sensitivity,
                "file_type": "csv",
                "row_count": len(rows),
                "headers": headers,
                "timestamp": datetime.utcnow().isoformat(),
            },
            source_file=file_path,
            doc_type="csv",
        )


class HTMLLoader:
    """Load content from HTML files by stripping tags."""

    def load(
        self,
        file_path: str,
        role_access: List[str] = None,
        sensitivity: str = "internal",
    ) -> Document:
        from bs4 import BeautifulSoup

        role_access = role_access or ["viewer"]

        with open(file_path, "r", encoding="utf-8") as f:
            soup = BeautifulSoup(f.read(), "html.parser")

        # Remove script and style elements
        for tag in soup(["script", "style"]):
            tag.decompose()

        content = soup.get_text(separator="\n", strip=True)

        return Document(
            content=content,
            metadata={
                "role_access": role_access,
                "sensitivity": sensitivity,
                "file_type": "html",
                "timestamp": datetime.utcnow().isoformat(),
            },
            source_file=file_path,
            doc_type="html",
        )


class DocumentLoader:
    """Unified document loader dispatching by file extension."""

    LOADERS = {
        ".pdf": PDFLoader,
        ".md": MarkdownLoader,
        ".markdown": MarkdownLoader,
        ".csv": CSVLoader,
        ".html": HTMLLoader,
        ".htm": HTMLLoader,
    }

    def load(
        self,
        file_path: str,
        role_access: List[str] = None,
        sensitivity: str = "internal",
    ) -> Document:
        """Load a document based on its file extension."""
        ext = os.path.splitext(file_path)[1].lower()
        loader_class = self.LOADERS.get(ext)

        if not loader_class:
            raise ValueError(
                f"Unsupported file type: {ext}. Supported: {list(self.LOADERS.keys())}"
            )

        loader = loader_class()
        return loader.load(file_path, role_access, sensitivity)

    def load_directory(
        self,
        dir_path: str,
        role_access: List[str] = None,
        sensitivity: str = "internal",
    ) -> List[Document]:
        """Load all supported documents from a directory."""
        documents = []

        for root, _, files in os.walk(dir_path):
            for fname in sorted(files):
                ext = os.path.splitext(fname)[1].lower()
                if ext in self.LOADERS:
                    fpath = os.path.join(root, fname)
                    try:
                        doc = self.load(fpath, role_access, sensitivity)
                        documents.append(doc)
                    except Exception as e:
                        print(f"Warning: Failed to load {fpath}: {e}")

        return documents
