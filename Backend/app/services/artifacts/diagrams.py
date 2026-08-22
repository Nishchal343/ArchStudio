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
    sources = [node["label"] for node in nodes if node.get("kind") not in {"output", "note"}]
    if not sources:
        return "No architecture relationships were supported by the indexed repository evidence."
    labels = {node["id"]: node["label"] for node in nodes}
    relationships = "; ".join(
        f"{labels.get(edge['source'], edge['source'])} {edge.get('label', 'connects to')} "
        f"{labels.get(edge['target'], edge['target'])}" for edge in edges
    ) or "No explicit relationships were detected in the retrieved source snippets."
    return ("This diagram summarizes the repository components identified from the processed source files. "
            f"The source files represented are {', '.join(sources)}. "
            f"The detected relationships are: {relationships}. "
            "Relationships are included only when supported by file paths or source references; this is not a complete runtime trace.")


async def enrich_diagram_explanation(payload: DiagramPayload) -> DiagramPayload:
    """Generate a grounded walkthrough from the diagram structure itself."""
    graph = json.dumps({"nodes": payload.nodes, "edges": payload.edges}, ensure_ascii=False)
    prompt = f"""Explain this {payload.diagram_type.value} diagram for a developer.
Return JSON only: {{"explanation": "..."}}.

Explain every box/node: what it represents and what role it plays. Explain every arrow: what moves or depends on what, and in which direction. Then describe the complete flow from input or entry point to the final application output. For a database schema, explain every table and its columns, how records would be written/read through the application, and explain relationships only when they are explicitly present in the supplied graph. Use only the supplied diagram structure. Do not invent functions, business rules, database relationships, or runtime behavior that the graph does not establish. If a role cannot be determined from a file name or node kind, say that clearly. Write several useful paragraphs, not a generic disclaimer.

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
    if path.is_file():
        with path.open("r", encoding="utf-8") as source:
            for line in source:
                try:
                    content = json.loads(line).get("content", "")
                except json.JSONDecodeError:
                    continue
                for table in re.findall(r"__tablename__\s*=\s*[\"']([^\"']+)", content):
                    tables.setdefault(table, [])
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
    return DiagramPayload(title="Database Diagram", diagram_type="database_diagram",
                          mermaid="\n".join(mermaid_lines),
                          explanation=(f"This diagram shows the database tables detected in the processed source: {', '.join(node['label'] for node in nodes)}. "
                                       "Columns are included when their declarations were available; relationships are shown only when explicitly detected."),
                          nodes=nodes,
                          edges=[{"source": child, "target": parent, "label": column}
                                 for child, parent, column in sorted(foreign_keys)
                                 if child in known_tables and parent in known_tables],
                          notes=["Only schema declarations found in processed source were included."])


def render_diagram(payload: DiagramPayload, output_format: str) -> tuple[bytes, str, str]:
    positions = {
        node["id"]: (40 + (index % 3) * 290, 40 + (index // 3) * 100)
        for index, node in enumerate(payload.nodes)
    }

    def edge_points(edge):
        x1, y1 = positions[edge["source"]]
        x2, y2 = positions[edge["target"]]
        return x1 + 110, y1 + 50, x2 + 110, y2

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
            parts.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" marker-end="url(#arrow)"/>')
            if label:
                parts.append(f'<text x="{(x1 + x2) / 2}" y="{(y1 + y2) / 2 - 4}" text-anchor="middle">{label}</text>')
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
