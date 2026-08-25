from __future__ import annotations

from pathlib import Path
from typing import TypedDict

import tree_sitter_javascript as ts_javascript
import tree_sitter_python as ts_python
from tree_sitter import Language, Parser


class ChunkMetadata(TypedDict):
    file_path: str
    name: str
    node_type: str
    start_line: int
    end_line: int
    imports: list[str]
    calls: list[str]


class CodeChunk(TypedDict):
    id: str
    text: str
    metadata: ChunkMetadata


_PY_LANGUAGE = Language(ts_python.language())
_JS_LANGUAGE = Language(ts_javascript.language())

_LANGUAGES = {".py": _PY_LANGUAGE, ".js": _JS_LANGUAGE}

# Node types that wrap a definition we want to expose as a standalone chunk.
_IMPORT_TYPES = {
    ".py": {"import_statement", "import_from_statement"},
    ".js": {"import_statement"},
}

# The call-expression node type per grammar; both expose the callee on the
# "function" field.
_CALL_TYPES = {".py": "call", ".js": "call_expression"}


def chunk_file(file_path: str | Path, root: str | Path | None = None) -> list[CodeChunk]:
    """Chunk one source file.

    ``root`` is the repository root. Paths in chunk metadata are recorded
    relative to it, because those paths reach the writer's prompt and end up in
    a unified diff -- and ``git apply`` only accepts repo-relative paths.
    Recording the absolute checkout path produced diffs that could never apply.
    """
    path = Path(file_path)
    source = path.read_text(encoding="utf-8")
    suffix = path.suffix
    display = _display_path(path, root)

    if suffix in _LANGUAGES:
        chunks = _chunk_with_tree_sitter(display, source, suffix)
        if chunks:
            return chunks

    return [_build_file_chunk(display, source)]


def _display_path(path: Path, root: str | Path | None) -> Path:
    """Path as the model should see it: relative to the repository root."""
    if root is None:
        return path
    try:
        return path.resolve().relative_to(Path(root).resolve())
    except ValueError:
        # Outside the root: fall back to the bare name so no full path leaks.
        return Path(path.name)


def _chunk_with_tree_sitter(path: Path, source: str, suffix: str) -> list[CodeChunk]:
    source_bytes = source.encode("utf-8")
    parser = Parser(_LANGUAGES[suffix])
    root = parser.parse(source_bytes).root_node

    imports = _collect_imports(root, source_bytes, suffix)
    chunks: list[CodeChunk] = []

    for top_node in root.named_children:
        core = _core_definition(top_node, suffix)
        if core is None:
            continue
        name = _definition_name(core, source_bytes)
        if name is None:
            continue
        node_type = "class" if "class" in core.type else "function"
        chunks.append(
            _build_chunk(
                path,
                name,
                node_type,
                top_node.start_point[0] + 1,
                top_node.end_point[0] + 1,
                _node_text(top_node, source_bytes),
                imports,
                _collect_calls(top_node, source_bytes, suffix),
            )
        )

    return chunks


def _core_definition(node, suffix: str):
    """Return the underlying function/class node for a top-level statement, or None."""
    node_type = node.type

    if suffix == ".py":
        if node_type in {"function_definition", "class_definition"}:
            return node
        if node_type == "decorated_definition":
            return _first_child_of_types(node, {"function_definition", "class_definition"})
        return None

    # JavaScript
    if node_type in {"function_declaration", "generator_function_declaration", "class_declaration"}:
        return node
    if node_type == "export_statement":
        for child in node.named_children:
            inner = _core_definition(child, suffix)
            if inner is not None:
                return inner
        return None
    if node_type in {"lexical_declaration", "variable_declaration"}:
        # const handler = (req) => ... / var fn = function () { ... }
        declarator = _first_child_of_types(node, {"variable_declarator"})
        if declarator is not None:
            value = declarator.child_by_field_name("value")
            if value is not None and value.type in {"arrow_function", "function", "function_expression"}:
                return node
        return None
    return None


def _definition_name(core, source_bytes: bytes) -> str | None:
    if core.type in {"lexical_declaration", "variable_declaration"}:
        declarator = _first_child_of_types(core, {"variable_declarator"})
        if declarator is None:
            return None
        name_node = declarator.child_by_field_name("name")
        return _node_text(name_node, source_bytes) if name_node is not None else None

    name_node = core.child_by_field_name("name")
    return _node_text(name_node, source_bytes) if name_node is not None else None


def _collect_imports(root, source_bytes: bytes, suffix: str) -> list[str]:
    import_types = _IMPORT_TYPES[suffix]
    return [
        _node_text(node, source_bytes).strip()
        for node in root.named_children
        if node.type in import_types
    ]


def _collect_calls(node, source_bytes: bytes, suffix: str) -> list[str]:
    call_type = _CALL_TYPES[suffix]
    calls: list[str] = []
    seen: set[str] = set()
    stack = [node]

    while stack:
        current = stack.pop()
        if current.type == call_type:
            callee = current.child_by_field_name("function")
            name = _callee_name(callee, source_bytes) if callee is not None else None
            if name is not None and name not in seen:
                seen.add(name)
                calls.append(name)
        stack.extend(current.named_children)

    return sorted(calls)


def _callee_name(node, source_bytes: bytes) -> str | None:
    if node.type == "identifier":
        return _node_text(node, source_bytes)
    if node.type == "attribute":  # Python: pkg.mod.func -> func
        attribute = node.child_by_field_name("attribute")
        return _node_text(attribute, source_bytes) if attribute is not None else None
    if node.type == "member_expression":  # JavaScript: obj.method -> method
        prop = node.child_by_field_name("property")
        return _node_text(prop, source_bytes) if prop is not None else None
    return None


def _build_chunk(
    path: Path,
    name: str,
    node_type: str,
    start_line: int,
    end_line: int,
    text: str,
    imports: list[str],
    calls: list[str],
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
            "calls": calls,
        },
    }


def _build_file_chunk(path: Path, source: str) -> CodeChunk:
    lines = source.splitlines()
    imports = [line.strip() for line in lines if line.lstrip().startswith(("import ", "from "))]
    return _build_chunk(path, path.name, "file", 1, len(lines) or 1, source, imports, [])


def _first_child_of_types(node, types: set[str]):
    for child in node.named_children:
        if child.type in types:
            return child
    return None


def _node_text(node, source_bytes: bytes) -> str:
    return source_bytes[node.start_byte : node.end_byte].decode("utf-8")
