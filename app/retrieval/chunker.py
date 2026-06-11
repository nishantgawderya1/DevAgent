from __future__ import annotations

from ast import AST, AsyncFunctionDef, ClassDef, FunctionDef, Import, ImportFrom, Module, parse
from pathlib import Path
from typing import TypedDict


class ChunkMetadata(TypedDict):
    file_path: str
    name: str
    node_type: str
    start_line: int
    end_line: int
    imports: list[str]


class CodeChunk(TypedDict):
    id: str
    text: str
    metadata: ChunkMetadata


def chunk_file(file_path: str | Path) -> list[CodeChunk]:
    path = Path(file_path)
    source = path.read_text(encoding="utf-8")

    if path.suffix == ".py":
        return _chunk_python_file(path, source)

    return _chunk_file_level(path, source)


def _chunk_python_file(path: Path, source: str) -> list[CodeChunk]:
    module = parse(source)
    lines = source.splitlines()
    imports = _collect_imports(module)
    chunks: list[CodeChunk] = []

    for node in module.body:
        if isinstance(node, (FunctionDef, AsyncFunctionDef, ClassDef)):
            chunks.extend(_chunk_python_node(path, lines, node, imports))

    if chunks:
        return chunks

    return [_build_chunk(path, path.name, "file", 1, len(lines) or 1, source, imports)]


def _chunk_python_node(path: Path, lines: list[str], node: AST, imports: list[str]) -> list[CodeChunk]:
    if isinstance(node, (FunctionDef, AsyncFunctionDef)):
        return [_build_function_chunk(path, lines, node, imports)]

    if isinstance(node, ClassDef):
        return [
            _build_chunk(
                path,
                node.name,
                "class",
                node.lineno,
                _node_end_line(node),
                _slice_source(lines, node.lineno, _node_end_line(node)),
                imports,
            )
        ]

    return []


def _build_function_chunk(path: Path, lines: list[str], node: FunctionDef | AsyncFunctionDef, imports: list[str]) -> CodeChunk:
    start_line = node.lineno
    end_line = _node_end_line(node)
    text = _slice_source(lines, start_line, end_line)
    return _build_chunk(path, node.name, "function", start_line, end_line, text, imports)


def _build_chunk(
    path: Path,
    name: str,
    node_type: str,
    start_line: int,
    end_line: int,
    text: str,
    imports: list[str],
) -> CodeChunk:
    return {
        "id": f"{path.as_posix()}::{name}",
        "text": text,
        "metadata": {
            "file_path": path.as_posix(),
            "name": name,
            "node_type": node_type,
            "start_line": start_line,
            "end_line": end_line,
            "imports": imports,
        },
    }


def _chunk_file_level(path: Path, source: str) -> list[CodeChunk]:
    lines = source.splitlines()
    imports = _extract_import_lines(lines)
    return [_build_chunk(path, path.name, "file", 1, len(lines) or 1, source, imports)]


def _collect_imports(module: Module) -> list[str]:
    imports: list[str] = []
    for node in module.body:
        if isinstance(node, Import):
            for alias in node.names:
                imports.append(alias.name)
        elif isinstance(node, ImportFrom):
            module_name = node.module or ""
            for alias in node.names:
                imports.append(f"{module_name}:{alias.name}" if module_name else alias.name)
    return imports


def _extract_import_lines(lines: list[str]) -> list[str]:
    return [line.strip() for line in lines if line.lstrip().startswith(("import ", "from "))]


def _node_end_line(node: AST) -> int:
    end_line = getattr(node, "end_lineno", None)
    if end_line is None:
        return getattr(node, "lineno", 1)
    return end_line


def _slice_source(lines: list[str], start_line: int, end_line: int) -> str:
    return "\n".join(lines[start_line - 1 : end_line])
