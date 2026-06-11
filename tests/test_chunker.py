from pathlib import Path

from app.retrieval.chunker import chunk_file


def test_chunk_file_returns_complete_function_chunks(tmp_path: Path) -> None:
    source = """import os


def first():
    value = 1
    return value


class Example:
    def method(self):
        return 2


def second():
    return 3
"""
    file_path = tmp_path / "sample.py"
    file_path.write_text(source, encoding="utf-8")

    chunks = chunk_file(file_path)

    assert [chunk["metadata"]["name"] for chunk in chunks] == ["first", "Example", "second"]
    assert chunks[0]["text"].strip() == "def first():\n    value = 1\n    return value"
    assert chunks[1]["metadata"]["node_type"] == "class"
    assert chunks[1]["text"].strip().startswith("class Example:")
    assert chunks[2]["text"].strip() == "def second():\n    return 3"


def test_chunk_file_uses_file_level_chunk_when_no_definitions(tmp_path: Path) -> None:
    source = """import os

VALUE = 1
"""
    file_path = tmp_path / "constants.py"
    file_path.write_text(source, encoding="utf-8")

    chunks = chunk_file(file_path)

    assert len(chunks) == 1
    assert chunks[0]["metadata"]["node_type"] == "file"
    assert chunks[0]["text"] == source