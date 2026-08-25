import json
import math
import re
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
    source_paths = [path for path in files if Path(path).suffix.lower() not in ignored]
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
    for path in selected:
        add_node(path, path)

    # Resolve common internal import forms to actual indexed files.
    def imported_stems(content: str, suffix: str) -> set[str]:
        refs: set[str] = set()
        if suffix == ".py":
            refs.update(re.findall(r"(?:from|import)\s+([A-Za-z_][\w.]*)", content))
        elif suffix in {".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs"}:
            refs.update(re.findall(r"(?:from|require\()\s*[\"']([^\"']+)", content))
        else:
            refs.update(re.findall(r"#include\s*[<\"]([^>\"]+)", content))
        return {ref.replace("\\", "/").rstrip("/").split("/")[-1].split(".")[0].lower() for ref in refs}

    for path in selected:
        source_id = ids[path]
        stems = imported_stems(files[path], Path(path).suffix.lower())
        for target in selected:
            if target == path:
                continue
            target_stem = Path(target).stem.lower()
            if target_stem in stems:
                add_edge(source_id, ids[target], "imports")

    for capability, pattern in capability_patterns.items():
        evidence = [path for path in selected if re.search(pattern, files[path], re.IGNORECASE)]
        if evidence:
            capability_id = add_node(f"capability:{capability}", capability, "capability")
            for path in evidence[:6]:
                add_edge(ids[path], capability_id, "provides")

    for path in source_paths:
        if Path(path).name.lower() not in {"docker-compose.yml", "docker-compose.yaml", "compose.yml", "compose.yaml"}:
            continue
        runtime_id = add_node(f"runtime:{path}", f"Container runtime ({path})", "runtime")
        for target in selected:
            if target != path and (Path(target).suffix.lower() in {".py", ".js", ".jsx", ".ts", ".tsx", ".java", ".go", ".rs"}):
                add_edge(runtime_id, ids[target], "orchestrates")

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
        for column, target_model in re.findall(
            r"^\s{4}(\w+)\s*=\s*models\.ForeignKey\(\s*([A-Za-z_]\w*)",
            body,
            re.MULTILINE,
        ):
            django_foreign_keys.append((table_name, target_model.lower(), column))

        field_pattern = re.compile(
            r"^\s{4}(\w+)\s*=\s*models\.(\w+)\((.*?)(?=^\s{4}\w+\s*=\s*models\.|^\s{4}class\s|\Z)",
            re.MULTILINE | re.DOTALL,
        )
        for column, field_type, declaration in field_pattern.findall(body):
            if field_type == "ForeignKey":
                target = re.search(r"models\.ForeignKey\(\s*([A-Za-z_]\w*)", declaration)
                if target:
                    django_foreign_keys.append((table_name, target.group(1).lower(), column))
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
                    content = json.loads(line).get("content", "")
                except json.JSONDecodeError:
                    continue
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
                    r"CREATE TABLE(?:\s+IF\s+NOT\s+EXISTS)?\s+(\w+)\s*\((?P<body>.*?)(?=\n[ \t]+\)(?:\s|$))",
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

    for child_table, target_model, column in django_foreign_keys:
        parent_table = django_models.get(target_model, target_model)
        if child_table in tables and parent_table in tables:
            foreign_keys.add((child_table, parent_table, column))
    if not tables:
        return DiagramPayload(title="Database Diagram", diagram_type="database_diagram",
                              mermaid='flowchart TD\n    empty["No database schema detected."]',
                              explanation="No database tables or schema relationships were detected in the processed repository.",
                              notes=["No database schema detected."])
    nodes = [{"id": _node_id(name, index), "label": name, "kind": "table",
              "columns": [{"name": col, "type": typ} for col, typ in columns]}
             for index, (name, columns) in enumerate(tables.items())]
    mermaid_lines = ["erDiagram"]
    for node in nodes:
        mermaid_lines.append(f"    {node['label']} {{")
        for column in node["columns"]:
            mermaid_lines.append(f"        {column['type']} {column['name']}")
        mermaid_lines.append("    }")
    known_tables = {node["label"] for node in nodes}
    emitted_relationships: set[tuple[str, str]] = set()
    for child, parent, _column in sorted(foreign_keys):
        if child in known_tables and parent in known_tables:
            relationship = (parent, child)
            if relationship not in emitted_relationships:
                mermaid_lines.append(f"    {parent} ||--o{{ {child} : contains")
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
    positions = {
        node["id"]: (40 + (index % 3) * 290, 40 + (index // 3) * 100)
        for index, node in enumerate(payload.nodes)
    }
    endpoint_ids = {node["label"]: node["id"] for node in payload.nodes}

    def position_for(endpoint: str):
        node_id = endpoint if endpoint in positions else endpoint_ids.get(endpoint)
        if node_id not in positions:
            raise ValueError(f"Diagram edge references unknown node: {endpoint}")
        return positions[node_id]

    def edge_points(edge):
        """Clip connectors to node boundaries so they never run through boxes."""
        sx, sy = position_for(edge["source"])
        tx, ty = position_for(edge["target"])
        sx, sy, tx, ty = sx + 110, sy + 25, tx + 110, ty + 25
        dx, dy = tx - sx, ty - sy
        if dx == 0 and dy == 0:
            return sx, sy, tx, ty
        scale = min(110 / abs(dx) if dx else float("inf"), 25 / abs(dy) if dy else float("inf"))
        return sx + dx * scale, sy + dy * scale, tx - dx * scale, ty - dy * scale

    if output_format == "svg":
        width, height = 900, max(180, 100 + len(payload.nodes) * 70)
        parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
                 '<defs><marker id="arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 z" fill="#555"/></marker></defs>',
                 '<style>text{font-family:Arial,sans-serif;font-size:14px}rect{fill:#eef4ff;stroke:#315d9b;stroke-width:2}line{stroke:#555;stroke-width:2}</style>']
        for index, node in enumerate(payload.nodes):
            x, y = positions[node["id"]]
            parts.append(f'<rect x="{x}" y="{y}" width="220" height="50" rx="8"/><text x="{x + 110}" y="{y + 30}" text-anchor="middle">{_svg_escape(node["label"])}</text>')
        for edge in payload.edges:
            x1, y1, x2, y2 = edge_points(edge)
            label = _svg_escape(str(edge.get("label", "")))
            parts.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" marker-end="url(#arrow)"/>')
            if label:
                lx, ly = (x1 + x2) / 2, (y1 + y2) / 2 - 4
                parts.append(f'<text x="{lx:.1f}" y="{ly:.1f}" text-anchor="middle" paint-order="stroke" stroke="white" stroke-width="5">{label}</text>')
        parts.append("</svg>")
        return "".join(parts).encode(), "image/svg+xml", "svg"
    if output_format == "png":
        from PIL import Image, ImageDraw
        image = Image.new("RGB", (900, max(180, 100 + len(payload.nodes) * 70)), "white")
        draw = ImageDraw.Draw(image)
        for edge in payload.edges:
            x1, y1, x2, y2 = edge_points(edge)
            draw.line((x1, y1, x2, y2), fill="#555555", width=3)
            angle = math.atan2(y2 - y1, x2 - x1)
            size = 9
            draw.polygon([
                (x2, y2),
                (x2 - size * math.cos(angle - math.pi / 6), y2 - size * math.sin(angle - math.pi / 6)),
                (x2 - size * math.cos(angle + math.pi / 6), y2 - size * math.sin(angle + math.pi / 6)),
            ], fill="#555555")
            if edge.get("label"):
                draw.text(((x1 + x2) / 2, (y1 + y2) / 2 - 14), str(edge["label"]), fill="#333333")
        for index, node in enumerate(payload.nodes):
            x, y = positions[node["id"]]
            draw.rounded_rectangle((x, y, x + 220, y + 50), radius=8, fill="#eef4ff", outline="#315d9b", width=2)
            draw.text((x + 10, y + 18), node["label"], fill="black")
        import io
        stream = io.BytesIO(); image.save(stream, format="PNG")
        return stream.getvalue(), "image/png", "png"
    if output_format == "pdf":
        from reportlab.pdfgen import canvas
        import io
        stream = io.BytesIO(); pdf = canvas.Canvas(stream, pagesize=(900, max(180, 100 + len(payload.nodes) * 70)))
        for edge in payload.edges:
            x1, y1, x2, y2 = edge_points(edge)
            pdf.setStrokeColorRGB(0.33, 0.33, 0.33)
            pdf.setLineWidth(2)
            pdf.line(x1, y1, x2, y2)
            angle = math.atan2(y2 - y1, x2 - x1)
            size = 8
            pdf.setFillColorRGB(0.33, 0.33, 0.33)
            pdf.wedge(x2 - size, y2 - size, x2 + size, y2 + size,
                      math.degrees(angle) - 30, 60)
        for index, node in enumerate(payload.nodes):
            x, y = positions[node["id"]]
            pdf.rect(x, y, 220, 50); pdf.drawString(x + 10, y + 22, node["label"])
        pdf.save(); return stream.getvalue(), "application/pdf", "pdf"
    raise ValueError("Unsupported diagram format")
