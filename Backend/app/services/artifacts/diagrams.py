import json
import math
import re
import textwrap
from pathlib import Path
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from ...schemas import DiagramPayload
from ..llm.service import get_llm_provider
from ..storage_service import project_directory


def _node_id(value: str, index: int) -> str:
    clean = re.sub(r"[^A-Za-z0-9_]", "_", value).strip("_") or f"node_{index}"
    return f"n{index}_{clean[:40]}"


def _svg_escape(value: str) -> str:
    return value.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def mermaid_from_graph(nodes: list[dict], edges: list[dict]) -> str:
    lines = ["flowchart TD"]
    for node in nodes:
        lines.append(f"    {node['id']}[\"{node['label'].replace(chr(34), '')}\"]")
    for edge in edges:
        lines.append(f"    {edge['source']} -->|{edge.get('label', '')}| {edge['target']}")
    return "\n".join(lines)


def _normalise_source_path(value: str) -> str:
    """Return a comparable repository-relative path."""
    return value.replace("\\", "/").lstrip("./").lower()


def _python_import_modules(path: str, content: str) -> set[str]:
    """Extract absolute and relative Python modules, including from-imports."""
    modules: set[str] = set()
    package = Path(path).parent.as_posix().replace("/", ".")
    if package == ".":
        package = ""

    for match in re.finditer(r"^\s*from\s+([.\w]+)\s+import\s+([^\n]+)", content, re.MULTILINE):
        module, imported = match.groups()
        dots = len(module) - len(module.lstrip("."))
        base = module[dots:]
        parent_parts = package.split(".") if package else []
        if dots:
            parent_parts = parent_parts[:max(0, len(parent_parts) - dots + 1)]
        absolute = (
            ".".join([*parent_parts, base]) if dots else base
        ).strip(".")
        if absolute:
            modules.add(absolute)
        # ``from package import module`` points to both the package and the
        # imported child module. Resolving both handles either source layout.
        for name in re.findall(r"[A-Za-z_]\w*", imported.split("#", 1)[0]):
            if name not in {"as", "type"}:
                modules.add(".".join(part for part in (absolute, name) if part))

    for match in re.finditer(r"^\s*import\s+([^\n]+)", content, re.MULTILINE):
        for item in match.group(1).split(","):
            module = re.split(r"\s+as\s+", item.strip(), maxsplit=1)[0].strip()
            if re.fullmatch(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*", module):
                modules.add(module)
    return modules


def _resolve_import_target(source_path: str, reference: str, known_paths: dict[str, str], suffix: str) -> str | None:
    """Resolve an import/reference to an indexed source file."""
    reference = reference.replace("\\", "/").strip()
    source = _normalise_source_path(source_path)
    candidates: list[str] = []

    if suffix == ".py":
        module = reference.lstrip(".").replace("/", ".")
        source_parts = source.rsplit("/", 1)[0].split("/") if "/" in source else []
        if reference.startswith("."):
            dots = len(reference) - len(reference.lstrip("."))
            base = source_parts[:max(0, len(source_parts) - dots + 1)]
            module = ".".join([*base, module]).strip(".")
        module_path = module.replace(".", "/")
        candidates.extend((f"{module_path}.py", f"{module_path}/__init__.py"))
    elif suffix in {".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs"}:
        if reference.startswith("."):
            base = f"{source.rsplit('/', 1)[0]}/{reference}" if "/" in source else reference
        else:
            base = reference
        base = _normalise_source_path(base).rstrip("/")
        candidates.extend(f"{base}{extension}" for extension in (".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs"))
        candidates.extend(f"{base}/{name}" for name in ("index.js", "index.jsx", "index.ts", "index.tsx"))
    else:
        base = _normalise_source_path(reference)
        candidates.extend((base, f"{base}.h", f"{base}.hpp", f"{base}.c", f"{base}.cpp"))

    for candidate in candidates:
        if candidate in known_paths:
            return known_paths[candidate]
    # Package aliases such as ``@/components/Button`` and Python package
    # imports can still be matched by their suffix when the repository root
    # is not part of the import name.
    for candidate in candidates:
        suffix_candidate = candidate.lstrip("./")
        matches = [path for normalised, path in known_paths.items() if normalised.endswith("/" + suffix_candidate) or normalised == suffix_candidate]
        if len(matches) == 1:
            return matches[0]
    return None


def _architecture_explanation(nodes: list[dict], edges: list[dict]) -> str:
    visible = [node for node in nodes if node.get("kind") not in {"output", "note"}]
    if not visible:
        return "No architecture relationships were supported by the indexed repository evidence."

    source_nodes = [node for node in visible if node.get("kind") == "source"]
    capabilities = [node["label"] for node in visible if node.get("kind") == "capability"]
    runtimes = [node["label"] for node in visible if node.get("kind") == "runtime"]

    def names(items: list[str], limit: int = 6) -> str:
        shown = items[:limit]
        suffix = f" and {len(items) - limit} more" if len(items) > limit else ""
        return ", ".join(shown) + suffix

    groups: dict[str, list[str]] = {
        "entry points and configuration": [],
        "application logic": [],
        "data and persistence": [],
        "presentation and routing": [],
        "other supporting components": [],
    }
    for node in source_nodes:
        path = node["label"]
        lower = path.lower()
        if any(token in lower for token in ("settings", "config", "manage.py", "main.py", "app.py", "application")):
            group = "entry points and configuration"
        elif any(token in lower for token in ("model", "migration", "schema", "database", "repository", "persistence")):
            group = "data and persistence"
        elif any(token in lower for token in ("template", "html", "component", "page", "view", "url", "route")):
            group = "presentation and routing"
        elif any(token in lower for token in ("service", "controller", "worker", "pipeline", "signal", "form")):
            group = "application logic"
        else:
            group = "other supporting components"
        groups[group].append(path)

    sections = [
        "This is a static architecture map of the repository, built from indexed source files and dependency signals. "
        "It is best read as a map of responsibility and dependency direction, not as a chronological execution trace.",
    ]
    for title, items in groups.items():
        if items:
            sections.append(f"{title.capitalize()}: {names(items)}.")
    if capabilities:
        sections.append(f"Detected capabilities: {names(capabilities)}. These capabilities are linked to the source files that appear to provide them.")
    if runtimes:
        sections.append(f"Runtime orchestration: {names(runtimes)}. These nodes indicate where the application is packaged or coordinated.")

    imports = [edge for edge in edges if edge.get("label") == "imports"]
    provides = [edge for edge in edges if edge.get("label") == "provides"]
    orchestrates = [edge for edge in edges if edge.get("label") == "orchestrates"]
    flow_parts = []
    if imports:
        flow_parts.append(f"{len(imports)} import dependencies connect modules")
    if provides:
        flow_parts.append(f"{len(provides)} capability links connect implementation files to detected features")
    if orchestrates:
        flow_parts.append(f"{len(orchestrates)} runtime links connect containers or launchers to application code")
    if flow_parts:
        sections.append("The arrows show that " + "; ".join(flow_parts) + ". Follow an arrow from its source to see which component depends on or provides the target.")
    else:
        sections.append("No explicit dependency arrows were detected in the indexed source evidence.")
    sections.append("Only relationships supported by file paths or source references are shown; external services, indirect calls, and runtime behavior may not appear.")
    return " ".join(sections)


async def enrich_diagram_explanation(payload: DiagramPayload) -> DiagramPayload:
    """Generate a grounded walkthrough from the diagram structure itself."""
    graph = json.dumps({"nodes": payload.nodes, "edges": payload.edges}, ensure_ascii=False)
    prompt = f"""Explain this {payload.diagram_type.value} diagram for a developer.
Return JSON only: {{"explanation": "..."}}.

Write a useful architecture walkthrough, not a catalogue of filenames. Start with the purpose of the diagram and its limitations. Group nodes by responsibility (entry/configuration, application logic, data/persistence, presentation/routing, capabilities, and runtime). For each group, name the most important components and explain their likely role from the node label and kind. Explain the arrow types: imports mean dependency direction, provides means a file supports a detected capability, and orchestrates means a runtime/container coordinates code. Describe the most plausible high-level flow only when the graph supports it. For a database schema, explain tables and columns and describe relationships only when explicitly present. Use only the supplied diagram structure; do not invent functions, business rules, database relationships, or runtime behavior. If a role cannot be determined, say so briefly. Return 2-4 concise paragraphs with concrete details, not a generic disclaimer.

DIAGRAM STRUCTURE:
{graph}
"""
    try:
        result = json.loads(await get_llm_provider().generate(prompt, max_tokens=2500))
        explanation = result.get("explanation") if isinstance(result, dict) else None
        if isinstance(explanation, str) and explanation.strip():
            payload.explanation = explanation.strip()
    except Exception:
        pass
    return payload


async def architecture_diagram(db: AsyncSession, project_id: UUID) -> DiagramPayload:
    """Build a repository-wide graph using generic source and dependency signals."""
    del db
    chunks_path = project_directory(str(project_id)) / "processed" / "chunks" / "chunks.jsonl"
    files: dict[str, str] = {}
    if chunks_path.is_file():
        with chunks_path.open("r", encoding="utf-8") as source:
            for line in source:
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                path = str(item.get("file_path") or "")
                if path:
                    files[path] = files.get(path, "") + "\n" + str(item.get("content") or "")

    nodes: list[dict] = []
    edges: list[dict] = []
    ids: dict[str, str] = {}

    def add_node(key: str, label: str, kind: str = "source") -> str:
        if key not in ids:
            ids[key] = _node_id(key, len(nodes))
            nodes.append({"id": ids[key], "label": label, "kind": kind})
        return ids[key]

    def add_edge(source: str | None, target: str | None, label: str) -> None:
        if not source or not target or source == target:
            return
        if not any(edge["source"] == source and edge["target"] == target for edge in edges):
            edges.append({"source": source, "target": target, "label": label})

    ignored = {".md", ".rst", ".txt", ".pdf", ".doc", ".docx", ".lock"}
    compose_names = {"docker-compose.yml", "docker-compose.yaml", "compose.yml", "compose.yaml"}
    runtime_paths = [path for path in files if Path(path).name.lower() in compose_names]
    source_paths = [
        path for path in files
        if Path(path).suffix.lower() not in ignored and Path(path).name.lower() not in compose_names
    ]
    known_paths = {_normalise_source_path(path): path for path in source_paths}
    key_names = {"main", "app", "application", "server", "index", "route", "api", "pipeline", "service", "models", "settings"}
    capability_patterns = {
        "OCR / document vision": r"\b(?:ocr|paddleocr|tesseract|easyocr)\b",
        "Machine learning": r"\b(?:torch|pytorch|tensorflow|sklearn|scikit-learn|keras|transformers|yolo)\b",
        "Blockchain / smart contract": r"\b(?:blockchain|ethereum|web3|solidity|ganache)\b",
        "Database / persistence": r"\b(?:sqlalchemy|django\.db|mongoose|sequelize|prisma|postgres|mysql|sqlite)\b",
    }

    def score(path: str) -> float:
        name = Path(path).stem.lower()
        value = 1.0 + min(len(files[path]) / 8000, 3)
        if name in key_names or any(token in name for token in key_names if token.strip()):
            value += 5
        if any(part.lower() in {"src", "app", "backend", "frontend", "server", "services"} for part in Path(path).parts):
            value += 2
        if any(part.lower() in {"test", "tests", "docs", "documentation", "node_modules"} for part in Path(path).parts):
            value -= 8
        return value

    selected = sorted(source_paths, key=score, reverse=True)[:18]
    for pattern in capability_patterns.values():
        selected.extend(path for path in source_paths if re.search(pattern, files[path], re.IGNORECASE) and path not in selected[:24])
    selected = list(dict.fromkeys(selected))[:24]

    # Expand from the important files into their actual internal dependencies.
    # The old implementation compared only filename stems (``service``), which
    # misses package imports such as ``from app.services.foo import run`` and
    # leaves most of the architecture as disconnected boxes.
    def import_references(path: str) -> set[str]:
        suffix = Path(path).suffix.lower()
        content = files[path]
        if suffix == ".py":
            return _python_import_modules(path, content)
        if suffix in {".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs"}:
            return set(re.findall(r"(?:from\s+|import\s+|import\s*\(|require\s*\()\s*[\"']([^\"']+)", content))
        if suffix in {".c", ".h", ".cc", ".cpp", ".cxx", ".hpp"}:
            return set(re.findall(r"#include\s*[<\"]([^>\"]+)", content))
        return set()

    max_nodes = 36
    cursor = 0
    while cursor < len(selected) and len(selected) < max_nodes:
        current = selected[cursor]
        cursor += 1
        for reference in import_references(current):
            target = _resolve_import_target(current, reference, known_paths, Path(current).suffix.lower())
            if target and target not in selected:
                selected.append(target)
                if len(selected) >= max_nodes:
                    break
    for path in selected:
        add_node(path, path)

    for path in selected:
        source_id = ids[path]
        for reference in import_references(path):
            target = _resolve_import_target(path, reference, known_paths, Path(path).suffix.lower())
            if target in ids and target != path:
                add_edge(source_id, ids[target], "imports")

    for capability, pattern in capability_patterns.items():
        evidence = [path for path in selected if re.search(pattern, files[path], re.IGNORECASE)]
        if evidence:
            capability_id = add_node(f"capability:{capability}", capability, "capability")
            for path in evidence[:6]:
                add_edge(ids[path], capability_id, "provides")

    for path in runtime_paths:
        if Path(path).name.lower() not in compose_names:
            continue
        runtime_id = add_node(f"runtime:{path}", f"Container runtime ({path})", "runtime")
        for target in selected:
            if target != path and (Path(target).suffix.lower() in {".py", ".js", ".jsx", ".ts", ".tsx", ".java", ".go", ".rs"}):
                add_edge(runtime_id, ids[target], "orchestrates")

    # Keep the graph visually navigable even when a valid source file has no
    # import edge (for example a standalone worker or a config module). The
    # repository node represents an observed path hierarchy, not an invented
    # runtime dependency. It also joins multiple legitimate dependency islands
    # into one architecture graph.
    if nodes:
        adjacency = {node["id"]: set() for node in nodes}
        for edge in edges:
            adjacency.setdefault(edge["source"], set()).add(edge["target"])
            adjacency.setdefault(edge["target"], set()).add(edge["source"])
        components: list[set[str]] = []
        remaining = set(adjacency)
        while remaining:
            start = remaining.pop()
            component = {start}
            stack = [start]
            while stack:
                current = stack.pop()
                for neighbour in adjacency[current] & remaining:
                    remaining.remove(neighbour)
                    component.add(neighbour)
                    stack.append(neighbour)
            components.append(component)
        if len(components) > 1:
            repository_id = add_node("architecture:repository", "Repository structure", "group")
            for component in components:
                representative = next(iter(component))
                add_edge(repository_id, representative, "contains")

    if not nodes:
        nodes = [{"id": "n0_empty", "label": "No processed source evidence detected", "kind": "note"}]
    return DiagramPayload(title="Architecture and Data Flow", diagram_type="architecture_diagram",
                          mermaid=mermaid_from_graph(nodes, edges),
                          explanation=_architecture_explanation(nodes, edges), nodes=nodes, edges=edges)


def database_diagram(project_id: UUID) -> DiagramPayload:
    path = project_directory(str(project_id)) / "processed" / "chunks" / "chunks.jsonl"
    tables: dict[str, list[tuple[str, str]]] = {}
    foreign_keys: set[tuple[str, str, str]] = set()
    django_models: dict[str, str] = {}
    django_foreign_keys: list[tuple[str, str, str]] = []
    file_contents: dict[str, str] = {}

    def django_type(field_type: str) -> str:
        return {
            "AutoField": "INTEGER",
            "BigAutoField": "INTEGER",
            "IntegerField": "INTEGER",
            "PositiveIntegerField": "INTEGER",
            "DecimalField": "DECIMAL",
            "FloatField": "REAL",
            "BooleanField": "BOOLEAN",
            "DateField": "DATE",
            "DateTimeField": "DATETIME",
            "JSONField": "JSON",
            "TextField": "TEXT",
            "EmailField": "TEXT",
            "URLField": "TEXT",
            "CharField": "VARCHAR",
        }.get(field_type, "TEXT")

    def add_django_model(model_name: str, body: str) -> None:
        meta = re.search(r"class\s+Meta\s*:.*?db_table\s*=\s*[\"']([^\"']+)", body, re.DOTALL)
        table_name = meta.group(1) if meta else re.sub(r"(?<!^)(?=[A-Z])", "_", model_name).lower()
        django_models[model_name.lower()] = table_name
        tables.setdefault(table_name, [])

        # Keep FK discovery independent from the broad field parser below.
        # Model declarations commonly span several lines, and a chunk boundary
        # or an additional field option should not turn a real relationship into
        # an isolated table in the generated diagram.
        for column, quoted_target, bare_target in re.findall(
            r"^\s{4}(\w+)\s*=\s*models\.(?:ForeignKey|OneToOneField|ManyToManyField)\(\s*(?:[\"'](?:\w+\.)?([A-Za-z_]\w*)[\"']|([A-Za-z_]\w*))",
            body,
            re.MULTILINE,
        ):
            django_foreign_keys.append((table_name, (quoted_target or bare_target).lower(), column))

        field_pattern = re.compile(
            r"^\s{4}(\w+)\s*=\s*models\.(\w+)\((.*?)(?=^\s{4}\w+\s*=\s*models\.|^\s{4}class\s|\Z)",
            re.MULTILINE | re.DOTALL,
        )
        for column, field_type, declaration in field_pattern.findall(body):
            if field_type in {"ForeignKey", "OneToOneField", "ManyToManyField"}:
                target = re.search(r"models\.(?:ForeignKey|OneToOneField|ManyToManyField)\(\s*(?:[\"'](?:\w+\.)?([A-Za-z_]\w*)[\"']|([A-Za-z_]\w*))", declaration)
                if target:
                    django_foreign_keys.append((table_name, (target.group(1) or target.group(2)).lower(), column))
            # Django stores a ForeignKey as the target model's integer primary
            # key (the database column is normally named ``<field>_id``).
            # Rendering it as TEXT makes the relationship easy to miss in an
            # ER diagram, even though the model declaration is explicit.
            column_type = "INTEGER" if field_type == "ForeignKey" else django_type(field_type)
            if (column, column_type) not in tables[table_name]:
                tables[table_name].append((column, column_type))

    if path.is_file():
        with path.open("r", encoding="utf-8") as source:
            for line in source:
                try:
                    item = json.loads(line)
                    content = item.get("content", "")
                except json.JSONDecodeError:
                    continue
                file_key = str(item.get("file_path") or "")
                file_contents[file_key] = file_contents.get(file_key, "") + content
                for table in re.findall(r"__tablename__\s*=\s*[\"']([^\"']+)", content):
                    tables.setdefault(table, [])
                for model_match in re.finditer(
                    r"class\s+(\w+)\s*\(\s*models\.Model\s*\)\s*:(.*?)(?=^class\s+\w+\s*\(|\Z)",
                    content,
                    re.DOTALL | re.MULTILINE,
                ):
                    add_django_model(model_match.group(1), model_match.group(2))
                for class_match in re.finditer(
                    r"__tablename__\s*=\s*[\"']([^\"']+)[\"'](?P<body>.*?)(?=\nclass |\Z)", content, re.DOTALL
                ):
                    table = class_match.group(1)
                    tables.setdefault(table, [])
                    for column, declaration in re.findall(
                        r"^\s{4}(\w+)\s*:.*?mapped_column\(([^\n]*)\)", class_match.group("body"), re.MULTILINE
                    ):
                        column_type = (re.search(r"(?:String|Integer|Boolean|DateTime|UUID|JSON)", declaration) or ["unknown"])[0]
                        if (column, column_type) not in tables[table]:
                            tables[table].append((column, column_type))
                        target = re.search(r"ForeignKey\(\s*[\"']([^\"']+)[\"']", declaration)
                        if target:
                            target_table, _, target_column = target.group(1).partition(".")
                            foreign_keys.add((table, target_table, column))
                    for column, declaration in re.findall(
                        r"^\s{4}(\w+)\s*=\s*Column\(([^\n]*)\)", class_match.group("body"), re.MULTILINE
                    ):
                        column_type = (re.search(r"(?:String|Integer|Boolean|DateTime|Text|JSON|UUID)", declaration) or ["unknown"])[0]
                        if (column, column_type) not in tables[table]:
                            tables[table].append((column, column_type))
                        target = re.search(r"ForeignKey\(\s*[\"']([^\"']+)[\"']", declaration)
                        if target:
                            target_table, _, _ = target.group(1).rpartition(".")
                            foreign_keys.add((table, target_table, column))
                    # Also recognize conventional *_id fields when a foreign-key
                    # constraint is not present in the source model.
                    for column, _ in tables[table]:
                        if column.endswith("_id"):
                            stem = column[:-3]
                            candidates = [stem, f"{stem}s", stem.rstrip("s")]
                            target_table = next((candidate for candidate in candidates if candidate in tables), None)
                            if target_table and target_table != table:
                                foreign_keys.add((table, target_table, column))
                for table_match in re.finditer(
                    r"CREATE TABLE(?:\s+IF\s+NOT\s+EXISTS)?\s+(\w+)\s*\((?P<body>.*?)(?=\n[ \t]+\)(?:\s|$)|\)\s*;)",
                    content, re.IGNORECASE | re.DOTALL
                ):
                    table = table_match.group(1)
                    body = table_match.group("body")
                    tables.setdefault(table, [])
                    for column, declaration in re.findall(
                        r"^\s*(\w+)\s+([A-Za-z]+(?:\s*\([^\n]*?\))?(?:\s+[A-Za-z]+)*)(?=\s*(?:,|$))",
                        body, re.MULTILINE
                    ):
                        if column.upper() not in {"FOREIGN", "PRIMARY", "UNIQUE", "CHECK", "CONSTRAINT"}:
                            column_type = (re.search(r"(?:INTEGER|TEXT|REAL|BOOLEAN|DATETIME|DATE|BLOB|UUID)", declaration, re.IGNORECASE) or ["unknown"])[0]
                            if (column, column_type) not in tables[table]:
                                tables[table].append((column, column_type))
                    for column, target_table, _target_column in re.findall(
                        r"FOREIGN\s+KEY\s*\(\s*(\w+)\s*\)\s*REFERENCES\s+(\w+)\s*\(\s*(\w+)\s*\)",
                        body, re.IGNORECASE
                    ):
                        foreign_keys.add((table, target_table, column))
                    for column, target_table, _target_column in re.findall(
                        r"^\s*(\w+)\s+[^\n,]+?REFERENCES\s+(\w+)\s*\(\s*(\w+)\s*\)",
                        body, re.IGNORECASE | re.MULTILINE
                    ):
                        foreign_keys.add((table, target_table, column))
                for table_match in re.finditer(
                    r"op\.create_table\(\s*[\"'](\w+)[\"'](?P<body>.*?)(?=\n\s*op\.create_table|\n\s*for column|\n\s*def |\Z)",
                    content, re.DOTALL
                ):
                    table = table_match.group(1)
                    body = table_match.group("body")
                    tables.setdefault(table, [])
                    for column, declaration in re.findall(
                        r"sa\.Column\(\s*[\"'](\w+)[\"']\s*,\s*([^\n]*)", body
                    ):
                        column_type = (re.search(r"(?:String|Integer|Boolean|DateTime|Text|JSON|UUID)", declaration) or ["unknown"])[0]
                        if (column, column_type) not in tables[table]:
                            tables[table].append((column, column_type))
                    for columns, targets in re.findall(
                        r"sa\.ForeignKeyConstraint\(\s*\[([^\]]+)\]\s*,\s*\[([^\]]+)\]", body
                    ):
                        column = re.search(r"[\"'](\w+)[\"']", columns)
                        target = re.search(r"[\"'](\w+)\.(\w+)[\"']", targets)
                        if column and target:
                            foreign_keys.add((table, target.group(1), column.group(1)))
                # Django migrations provide a second source of schema evidence.
                for migration_match in re.finditer(
                    r"migrations\.CreateModel\(\s*name\s*=\s*[\"'](\w+)[\"'](?P<body>.*?)(?=\n\s*migrations\.(?:CreateModel|AddField|AlterField|DeleteModel)|\n\s*\])",
                    content,
                    re.DOTALL,
                ):
                    model_name = migration_match.group(1)
                    body = migration_match.group("body")
                    table_match = re.search(r"[\"']db_table[\"']\s*:\s*[\"']([^\"']+)", body)
                    table_name = table_match.group(1) if table_match else re.sub(r"(?<!^)(?=[A-Z])", "_", model_name).lower()
                    django_models[model_name.lower()] = table_name
                    tables.setdefault(table_name, [])
                    for field_match in re.finditer(
                        r"\(\s*[\"'](\w+)[\"']\s*,\s*models\.(\w+)\((.*?)(?=\n\s*\(\s*[\"']\w+[\"']\s*,|\n\s*\]\s*)",
                        body,
                        re.DOTALL,
                    ):
                        column, field_type, declaration = field_match.groups()
                        column_type = "INTEGER" if field_type == "ForeignKey" else django_type(field_type)
                        if (column, column_type) not in tables[table_name]:
                            tables[table_name].append((column, column_type))
                        if field_type == "ForeignKey":
                            target = re.search(r"to\s*=\s*[\"'](?:\w+\.)?(\w+)[\"']", declaration)
                            if target:
                                django_foreign_keys.append((table_name, target.group(1).lower(), column))

    # Chunks are created by line count, so a class or migration can be split
    # across several records. Re-run relationship discovery on reconstructed
    # files; this is what makes model declarations connect reliably.
    for content in file_contents.values():
        for model_match in re.finditer(
            r"class\s+(\w+)\s*\(\s*models\.Model\s*\)\s*:(.*?)(?=^class\s+\w+\s*\(|\Z)",
            content,
            re.DOTALL | re.MULTILINE,
        ):
            add_django_model(model_match.group(1), model_match.group(2))

        for class_match in re.finditer(
            r"__tablename__\s*=\s*[\"']([^\"']+)[\"'](?P<body>.*?)(?=\nclass\s|\Z)",
            content,
            re.DOTALL,
        ):
            table = class_match.group(1)
            tables.setdefault(table, [])
            for column, declaration in re.findall(
                r"^\s{4}(\w+)\s*(?::[^=\n]+)?=\s*(?:mapped_column|Column)\(([^\n]*)\)",
                class_match.group("body"),
                re.MULTILINE,
            ):
                target = re.search(r"ForeignKey\(\s*[\"']([^\"']+)[\"']", declaration)
                if target:
                    target_table, _, _ = target.group(1).rpartition(".")
                    foreign_keys.add((table, target_table or target.group(1), column))

        for table_match in re.finditer(
            r"CREATE TABLE(?:\s+IF\s+NOT\s+EXISTS)?\s+(\w+)\s*\((?P<body>.*?)(?=\n\s*\)(?:\s*;|\s*$)|\)\s*;)",
            content,
            re.IGNORECASE | re.DOTALL,
        ):
            table = table_match.group(1)
            body = table_match.group("body")
            for column, target_table, _target_column in re.findall(
                r"FOREIGN\s+KEY\s*\(\s*(\w+)\s*\)\s*REFERENCES\s+([\w.-]+)\s*\(\s*(\w+)\s*\)",
                body,
                re.IGNORECASE,
            ):
                foreign_keys.add((table, target_table, column))
            for column, target_table, _target_column in re.findall(
                r"^\s*(\w+)\s+[^\n,]*?REFERENCES\s+([\w.-]+)\s*\(\s*(\w+)\s*\)",
                body,
                re.IGNORECASE | re.MULTILINE,
            ):
                foreign_keys.add((table, target_table, column))

        for migration_match in re.finditer(
            r"migrations\.CreateModel\(\s*name\s*=\s*[\"'](\w+)[\"'](?P<body>.*?)(?=\n\s*migrations\.(?:CreateModel|AddField|AlterField|DeleteModel)|\Z)",
            content,
            re.DOTALL,
        ):
            model_name = migration_match.group(1)
            body = migration_match.group("body")
            table_match = re.search(r"[\"']db_table[\"']\s*:\s*[\"']([^\"']+)", body)
            table_name = table_match.group(1) if table_match else re.sub(r"(?<!^)(?=[A-Z])", "_", model_name).lower()
            django_models[model_name.lower()] = table_name
            tables.setdefault(table_name, [])
            for field_match in re.finditer(
                r"\(\s*[\"'](\w+)[\"']\s*,\s*models\.(?:ForeignKey|OneToOneField|ManyToManyField)\((.*?)(?=\n\s*\(\s*[\"']\w+[\"']\s*,|\n\s*\]\s*)",
                body,
                re.DOTALL,
            ):
                column, declaration = field_match.groups()
                target = re.search(r"to\s*=\s*[\"'](?:\w+\.)?(\w+)[\"']", declaration)
                if target:
                    django_foreign_keys.append((table_name, target.group(1).lower(), column))

    def table_key(value: str) -> str:
        value = value.strip().strip("`\"'").lower()
        return value.rsplit(".", 1)[-1]

    aliases: dict[str, set[str]] = {}
    for table in tables:
        forms = {table_key(table)}
        if forms:
            singular = next(iter(forms))
            forms.add(singular[:-1] if singular.endswith("s") else f"{singular}s")
        for form in forms:
            aliases.setdefault(form, set()).add(table)
    for model_name, table in django_models.items():
        if table in tables:
            aliases.setdefault(table_key(model_name), set()).add(table)

    def resolve_table(value: str) -> str | None:
        candidates = aliases.get(table_key(value), set())
        return next(iter(candidates)) if len(candidates) == 1 else None

    # Conventional ``owner_id`` columns are useful evidence even when a
    # migration omitted an explicit constraint. Resolve them only when the
    # corresponding table name is unambiguous.
    for child, columns in tables.items():
        for column, _column_type in columns:
            if column.lower().endswith("_id"):
                parent = resolve_table(column[:-3])
                if parent and parent != child:
                    foreign_keys.add((child, parent, column))

    resolved_foreign_keys: set[tuple[str, str, str]] = set()
    for child, parent, column in foreign_keys:
        resolved_child = resolve_table(child)
        resolved_parent = resolve_table(parent)
        if resolved_child and resolved_parent and resolved_child != resolved_parent:
            resolved_foreign_keys.add((resolved_child, resolved_parent, column))
    for child_table, target_model, column in django_foreign_keys:
        resolved_child = resolve_table(child_table)
        resolved_parent = resolve_table(django_models.get(target_model, target_model))
        if resolved_child and resolved_parent and resolved_child != resolved_parent:
            resolved_foreign_keys.add((resolved_child, resolved_parent, column))
    foreign_keys = resolved_foreign_keys
    if not tables:
        return DiagramPayload(title="Database Diagram", diagram_type="database_diagram",
                              mermaid='flowchart TD\n    empty["No database schema detected."]',
                              explanation="No database tables or schema relationships were detected in the processed repository.",
                              notes=["No database schema detected."])
    nodes = [{"id": _node_id(name, index), "label": name, "kind": "table",
              "columns": [{"name": col, "type": typ} for col, typ in columns]}
             for index, (name, columns) in enumerate(tables.items())]
    known_tables = {node["label"] for node in nodes}
    mermaid_table_names: dict[str, str] = {}
    used_mermaid_names: set[str] = set()
    for node in nodes:
        base = re.sub(r"[^A-Za-z0-9_]", "_", node["label"]).strip("_") or "table"
        if base[0].isdigit():
            base = f"table_{base}"
        candidate = base
        suffix = 2
        while candidate in used_mermaid_names:
            candidate = f"{base}_{suffix}"
            suffix += 1
        used_mermaid_names.add(candidate)
        mermaid_table_names[node["label"]] = candidate
    mermaid_lines = ["erDiagram"]
    for node in nodes:
        mermaid_lines.append(f"    {mermaid_table_names[node['label']]} {{")
        for column in node["columns"]:
            column_name = re.sub(r"[^A-Za-z0-9_]", "_", column["name"]).strip("_") or "column"
            mermaid_lines.append(f"        {column['type']} {column_name}")
        mermaid_lines.append("    }")
    emitted_relationships: set[tuple[str, str]] = set()
    for child, parent, _column in sorted(foreign_keys):
        if child in known_tables and parent in known_tables:
            relationship = (parent, child)
            if relationship not in emitted_relationships:
                mermaid_lines.append(
                    f"    {mermaid_table_names[parent]} ||--o{{ {mermaid_table_names[child]} : contains"
                )
                emitted_relationships.add(relationship)
    table_ids = {node["label"]: node["id"] for node in nodes}
    return DiagramPayload(title="Database Diagram", diagram_type="database_diagram",
                          mermaid="\n".join(mermaid_lines),
                          explanation=(f"This diagram shows the database tables detected in the processed source: {', '.join(node['label'] for node in nodes)}. "
                                       "Columns are included when their declarations were available; relationships are shown only when explicitly detected."),
                          nodes=nodes,
                          edges=[{"source": table_ids[child], "target": table_ids[parent], "label": column}
                                 for child, parent, column in sorted(foreign_keys)
                                 if child in known_tables and parent in known_tables],
                          notes=["Only schema declarations found in processed source were included."])


def render_diagram(payload: DiagramPayload, output_format: str) -> tuple[bytes, str, str]:
    import io

    if not payload.nodes:
        raise ValueError("Cannot render an empty diagram")

    node_by_id = {node["id"]: node for node in payload.nodes}
    endpoint_ids = {node["label"]: node["id"] for node in payload.nodes}
    edges = [
        edge for edge in payload.edges
        if (edge.get("source") in node_by_id or edge.get("source") in endpoint_ids)
        and (edge.get("target") in node_by_id or edge.get("target") in endpoint_ids)
    ]

    def resolve_endpoint(endpoint: str) -> str:
        node_id = endpoint if endpoint in node_by_id else endpoint_ids.get(endpoint)
        if node_id not in node_by_id:
            raise ValueError(f"Diagram edge references unknown node: {endpoint}")
        return node_id

    edges = [
        {**edge, "source": resolve_endpoint(edge["source"]), "target": resolve_endpoint(edge["target"])}
        for edge in edges
        if edge.get("source") != edge.get("target")
    ]

    is_schema = payload.diagram_type.value == "database_diagram" if hasattr(payload.diagram_type, "value") else payload.diagram_type == "database_diagram"
    node_width = 300 if is_schema else 260
    horizontal_gap = 110
    vertical_gap = 38
    margin = 42

    def wrapped_lines(value: str, width: int = 30) -> list[str]:
        lines = textwrap.wrap(str(value), width=width, break_long_words=True, break_on_hyphens=False)
        return lines or [""]

    def node_lines(node: dict) -> list[str]:
        return wrapped_lines(str(node.get("label", "")), 32 if is_schema else 30)

    def node_size(node: dict) -> tuple[int, int]:
        if node.get("columns"):
            return node_width, min(420, 42 + 24 * len(node["columns"]))
        return node_width, max(62, 28 + 20 * len(node_lines(node)))

    sizes = {node["id"]: node_size(node) for node in payload.nodes}
    node_ids = list(node_by_id)
    outgoing = {node_id: [] for node_id in node_ids}
    incoming = {node_id: [] for node_id in node_ids}
    for edge in edges:
        outgoing[edge["source"]].append(edge["target"])
        incoming[edge["target"]].append(edge["source"])

    # Assign columns from the dependency direction. Kahn's algorithm keeps
    # cyclic modules in a stable first column instead of endlessly increasing
    # their rank.
    indegree = {node_id: len(incoming[node_id]) for node_id in node_ids}
    rank = {node_id: 0 for node_id in node_ids}
    queue = [node_id for node_id in node_ids if indegree[node_id] == 0]
    processed: set[str] = set()
    while queue:
        current = queue.pop(0)
        processed.add(current)
        for target in outgoing[current]:
            rank[target] = max(rank[target], rank[current] + 1)
            indegree[target] -= 1
            if indegree[target] == 0:
                queue.append(target)
    for node_id in node_ids:
        if node_id not in processed:
            rank[node_id] = 0

    layers: dict[int, list[str]] = {}
    for node_id in node_ids:
        layers.setdefault(rank[node_id], []).append(node_id)
    for layer in layers.values():
        layer.sort(key=lambda node_id: str(node_by_id[node_id].get("label", "")).lower())

    # Barycentric ordering reduces edge crossings between adjacent columns.
    for _ in range(4):
        order = {node_id: index for layer in layers.values() for index, node_id in enumerate(layer)}
        for layer_number in sorted(layers):
            layer = layers[layer_number]
            if len(layer) < 2:
                continue
            def barycenter(node_id: str) -> tuple[float, str]:
                neighbours = incoming[node_id] + outgoing[node_id]
                positions = [order[neighbour] for neighbour in neighbours if neighbour in order]
                return (sum(positions) / len(positions) if positions else float("inf"), str(node_by_id[node_id].get("label", "")).lower())
            layer.sort(key=barycenter)

    layer_heights = {
        layer_number: sum(sizes[node_id][1] for node_id in layer) + vertical_gap * max(0, len(layer) - 1)
        for layer_number, layer in layers.items()
    }
    canvas_height = max(180, margin * 2 + max(layer_heights.values()))
    positions: dict[str, tuple[float, float, int, int]] = {}
    for layer_number, layer in layers.items():
        x = margin + layer_number * (node_width + horizontal_gap)
        y = margin + (canvas_height - margin * 2 - layer_heights[layer_number]) / 2
        for node_id in layer:
            width, height = sizes[node_id]
            positions[node_id] = (x, y, width, height)
            y += height + vertical_gap
    canvas_width = max(360, margin * 2 + (max(layers) + 1) * node_width + max(0, max(layers)) * horizontal_gap)

    def connector(edge: dict) -> tuple[list[tuple[float, float]], tuple[float, float]]:
        sx, sy, sw, sh = positions[edge["source"]]
        tx, ty, tw, th = positions[edge["target"]]
        if tx > sx:
            start = (sx + sw, sy + sh / 2)
            end = (tx, ty + th / 2)
            bend_x = (start[0] + end[0]) / 2
            points = [start, (bend_x, start[1]), (bend_x, end[1]), end]
        elif tx < sx:
            start = (sx, sy + sh / 2)
            end = (tx + tw, ty + th / 2)
            bend_x = (start[0] + end[0]) / 2
            points = [start, (bend_x, start[1]), (bend_x, end[1]), end]
        else:
            start = (sx + sw / 2, sy + sh)
            end = (tx + tw / 2, ty)
            bend_y = min(start[1], end[1]) - 24
            points = [start, (start[0], bend_y), (end[0], bend_y), end]
        # Remove duplicate points so labels and arrowheads remain stable.
        clean_points = [points[0]]
        for point in points[1:]:
            if point != clean_points[-1]:
                clean_points.append(point)
        midpoint = clean_points[len(clean_points) // 2]
        return clean_points, midpoint

    def arrowhead(points: list[tuple[float, float]], size: float = 9) -> list[tuple[float, float]]:
        x2, y2 = points[-1]
        x1, y1 = points[-2]
        angle = math.atan2(y2 - y1, x2 - x1)
        return [
            (x2, y2),
            (x2 - size * math.cos(angle - math.pi / 6), y2 - size * math.sin(angle - math.pi / 6)),
            (x2 - size * math.cos(angle + math.pi / 6), y2 - size * math.sin(angle + math.pi / 6)),
        ]

    if output_format == "svg":
        parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{canvas_width}" height="{canvas_height}" viewBox="0 0 {canvas_width} {canvas_height}">',
            '<defs><marker id="arrow" markerWidth="9" markerHeight="9" refX="8" refY="4.5" orient="auto"><path d="M0,0 L9,4.5 L0,9 z" fill="#64748b"/></marker></defs>',
            '<style>text{font-family:Arial,sans-serif;fill:#172033;font-size:13px} .edge{fill:none;stroke:#94a3b8;stroke-width:1.8} .card{fill:#f8fafc;stroke:#2563eb;stroke-width:1.6} .header{fill:#e8f0ff} .rule{stroke:#cbd5e1;stroke-width:1}</style>',
        ]
        for edge in edges:
            points, midpoint = connector(edge)
            path_data = " L ".join(f"{x:.1f},{y:.1f}" for x, y in points)
            parts.append(f'<path class="edge" d="M {path_data}" marker-end="url(#arrow)"/>')
            if edge.get("label"):
                label = _svg_escape(str(edge["label"]))
                parts.append(f'<text x="{midpoint[0]:.1f}" y="{midpoint[1] - 5:.1f}" text-anchor="middle" paint-order="stroke" stroke="white" stroke-width="5">{label}</text>')
        for node in payload.nodes:
            x, y, width, height = positions[node["id"]]
            parts.append(f'<rect class="card" x="{x:.1f}" y="{y:.1f}" width="{width}" height="{height}" rx="10"/>')
            if node.get("columns"):
                parts.append(f'<rect class="header" x="{x + 1:.1f}" y="{y + 1:.1f}" width="{width - 2}" height="32" rx="9"/>')
                parts.append(f'<line class="rule" x1="{x:.1f}" y1="{y + 34:.1f}" x2="{x + width:.1f}" y2="{y + 34:.1f}"/>')
                parts.append(f'<text x="{x + 12:.1f}" y="{y + 22:.1f}" font-weight="bold">{_svg_escape(str(node["label"]))}</text>')
                for index, column in enumerate(node["columns"]):
                    value = f'{column.get("type", "") } {column.get("name", "")}'.strip()
                    parts.append(f'<text x="{x + 12:.1f}" y="{y + 54 + index * 22:.1f}">{_svg_escape(value)}</text>')
            else:
                for index, line in enumerate(node_lines(node)):
                    parts.append(f'<text x="{x + width / 2:.1f}" y="{y + 28 + index * 19:.1f}" text-anchor="middle">{_svg_escape(line)}</text>')
        parts.append("</svg>")
        return "".join(parts).encode(), "image/svg+xml", "svg"

    if output_format == "png":
        from PIL import Image, ImageDraw

        image = Image.new("RGB", (int(canvas_width), int(canvas_height)), "white")
        draw = ImageDraw.Draw(image)
        for edge in edges:
            points, midpoint = connector(edge)
            draw.line(points, fill="#94a3b8", width=2, joint="curve")
            draw.polygon(arrowhead(points), fill="#64748b")
            if edge.get("label"):
                draw.text((midpoint[0], midpoint[1] - 16), str(edge["label"]), fill="#334155", anchor="mm")
        for node in payload.nodes:
            x, y, width, height = positions[node["id"]]
            draw.rounded_rectangle((x, y, x + width, y + height), radius=10, fill="#f8fafc", outline="#2563eb", width=2)
            if node.get("columns"):
                draw.rectangle((x + 1, y + 1, x + width - 1, y + 34), fill="#e8f0ff")
                draw.line((x, y + 34, x + width, y + 34), fill="#cbd5e1", width=1)
                draw.text((x + 12, y + 18), str(node["label"]), fill="#172033", anchor="lm")
                for index, column in enumerate(node["columns"]):
                    value = f'{column.get("type", "") } {column.get("name", "")}'.strip()
                    draw.text((x + 12, y + 54 + index * 22), value, fill="#334155", anchor="lm")
            else:
                draw.multiline_text((x + width / 2, y + height / 2), "\n".join(node_lines(node)), fill="#172033", anchor="mm", align="center", spacing=2)
        stream = io.BytesIO()
        image.save(stream, format="PNG")
        return stream.getvalue(), "image/png", "png"

    if output_format == "pdf":
        from reportlab.pdfgen import canvas

        stream = io.BytesIO()
        pdf = canvas.Canvas(stream, pagesize=(canvas_width, canvas_height))

        def pdf_point(point: tuple[float, float]) -> tuple[float, float]:
            return point[0], canvas_height - point[1]

        for edge in edges:
            points, midpoint = connector(edge)
            pdf.setStrokeColorRGB(0.58, 0.65, 0.72)
            pdf.setLineWidth(1.5)
            for start, end in zip(points, points[1:]):
                pdf.line(*pdf_point(start), *pdf_point(end))
            arrow = arrowhead(points)
            pdf.setFillColorRGB(0.39, 0.45, 0.54)
            pdf.setStrokeColorRGB(0.39, 0.45, 0.54)
            pdf.line(*pdf_point(arrow[0]), *pdf_point(arrow[1]))
            pdf.line(*pdf_point(arrow[0]), *pdf_point(arrow[2]))
            if edge.get("label"):
                pdf.setFillColorRGB(0.2, 0.25, 0.32)
                pdf.setFont("Helvetica", 8)
                pdf.drawCentredString(midpoint[0], canvas_height - midpoint[1] + 5, str(edge["label"]))
        for node in payload.nodes:
            x, y, width, height = positions[node["id"]]
            pdf_y = canvas_height - y - height
            pdf.setFillColorRGB(0.97, 0.98, 0.99)
            pdf.setStrokeColorRGB(0.15, 0.39, 0.92)
            pdf.roundRect(x, pdf_y, width, height, 10, fill=1, stroke=1)
            pdf.setFillColorRGB(0.91, 0.94, 1)
            if node.get("columns"):
                pdf.rect(x + 1, pdf_y + height - 34, width - 2, 33, fill=1, stroke=0)
                pdf.setFillColorRGB(0.09, 0.13, 0.2)
                pdf.setFont("Helvetica-Bold", 10)
                pdf.drawString(x + 12, pdf_y + height - 22, str(node["label"])[:42])
                pdf.setFillColorRGB(0.2, 0.25, 0.32)
                pdf.setFont("Helvetica", 8)
                for index, column in enumerate(node["columns"]):
                    value = f'{column.get("type", "") } {column.get("name", "")}'.strip()
                    pdf.drawString(x + 12, pdf_y + height - 54 - index * 22, value[:46])
            else:
                pdf.setFillColorRGB(0.09, 0.13, 0.2)
                pdf.setFont("Helvetica", 9)
                lines = node_lines(node)
                first_y = pdf_y + height / 2 + (len(lines) - 1) * 7
                for index, line in enumerate(lines):
                    pdf.drawCentredString(x + width / 2, first_y - index * 14, line)
        pdf.save()
        return stream.getvalue(), "application/pdf", "pdf"
    raise ValueError("Unsupported diagram format")
