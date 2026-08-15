import json
import re
from pathlib import Path
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from ...config import get_settings
from ...schemas import DiagramPayload
from ..rag.retrieval import search_project
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


async def architecture_diagram(db: AsyncSession, project_id: UUID) -> DiagramPayload:
    results = await search_project(db, project_id,
                                   "main entry point architecture data flow libraries runtime output",
                                   top_k=6, min_similarity=get_settings().analysis_min_similarity)
    nodes: list[dict] = []
    edges: list[dict] = []
    seen: set[str] = set()
    for result in results:
        path = result["file_path"]
        if path not in seen:
            node = {"id": _node_id(path, len(nodes)), "label": path, "kind": "source"}
            nodes.append(node)
            seen.add(path)
        source_id = next(node["id"] for node in nodes if node["label"] == path)
        content = result.get("content", "")
        for include in re.findall(r"#include\s*[<\"]([^>\"]+)", content):
            label = include.split("/")[-1]
            if label not in seen:
                lib = {"id": _node_id(label, len(nodes)), "label": label, "kind": "library"}
                nodes.append(lib)
                seen.add(label)
            target_id = next(node["id"] for node in nodes if node["label"] == label)
            if not any(edge["source"] == source_id and edge["target"] == target_id for edge in edges):
                edges.append({"source": source_id, "target": target_id, "label": "includes"})
        if "glutMainLoop" in content or "glut" in content.lower():
            label = "GLUT runtime"
            if label not in seen:
                nodes.append({"id": _node_id(label, len(nodes)), "label": label, "kind": "runtime"})
                seen.add(label)
            runtime_id = next(node["id"] for node in nodes if node["label"] == label)
            if not any(edge["source"] == source_id and edge["target"] == runtime_id for edge in edges):
                edges.append({"source": source_id, "target": runtime_id, "label": "runs"})
    if not nodes:
        nodes = [{"id": "n0_empty", "label": "No architecture evidence detected", "kind": "note"}]
    if any(node["kind"] == "source" for node in nodes) and any("gl" in node["label"].lower() for node in nodes):
        output = {"id": "n_output", "label": "Rendered application output", "kind": "output"}
        nodes.append(output)
        source = next(node for node in nodes if node["kind"] == "source")
        edges.append({"source": source["id"], "target": output["id"], "label": "renders"})
    return DiagramPayload(title="Architecture and Data Flow", diagram_type="architecture_diagram",
                          mermaid=mermaid_from_graph(nodes, edges), nodes=nodes, edges=edges)


def database_diagram(project_id: UUID) -> DiagramPayload:
    path = project_directory(str(project_id)) / "processed" / "chunks" / "chunks.jsonl"
    tables: dict[str, list[tuple[str, str]]] = {}
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
                for table, column, typ in re.findall(r"CREATE TABLE\s+(\w+).*?\b(\w+)\s+([A-Za-z]+)", content, re.IGNORECASE | re.DOTALL):
                    tables.setdefault(table, []).append((column, typ))
    if not tables:
        return DiagramPayload(title="Database Diagram", diagram_type="database_diagram",
                              mermaid='flowchart TD\n    empty["No database schema detected."]',
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
    return DiagramPayload(title="Database Diagram", diagram_type="database_diagram",
                          mermaid="\n".join(mermaid_lines), nodes=nodes,
                          notes=["Only schema declarations found in processed source were included."])


def render_diagram(payload: DiagramPayload, output_format: str) -> tuple[bytes, str, str]:
    if output_format == "svg":
        width, height = 900, max(180, 100 + len(payload.nodes) * 70)
        parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
                 '<style>text{font-family:Arial,sans-serif;font-size:14px}rect{fill:#eef4ff;stroke:#315d9b;stroke-width:2}line{stroke:#555;stroke-width:2}</style>']
        positions = {}
        for index, node in enumerate(payload.nodes):
            x, y = 40 + (index % 3) * 290, 40 + (index // 3) * 100
            positions[node["id"]] = (x, y)
            parts.append(f'<rect x="{x}" y="{y}" width="220" height="50" rx="8"/><text x="{x + 110}" y="{y + 30}" text-anchor="middle">{_svg_escape(node["label"])}</text>')
        for edge in payload.edges:
            x1, y1 = positions[edge["source"]]; x2, y2 = positions[edge["target"]]
            parts.append(f'<line x1="{x1 + 110}" y1="{y1 + 50}" x2="{x2 + 110}" y2="{y2}"/>')
        parts.append("</svg>")
        return "".join(parts).encode(), "image/svg+xml", "svg"
    if output_format == "png":
        from PIL import Image, ImageDraw
        image = Image.new("RGB", (900, max(180, 100 + len(payload.nodes) * 70)), "white")
        draw = ImageDraw.Draw(image)
        for index, node in enumerate(payload.nodes):
            x, y = 40 + (index % 3) * 290, 40 + (index // 3) * 100
            draw.rounded_rectangle((x, y, x + 220, y + 50), radius=8, fill="#eef4ff", outline="#315d9b", width=2)
            draw.text((x + 10, y + 18), node["label"], fill="black")
        import io
        stream = io.BytesIO(); image.save(stream, format="PNG")
        return stream.getvalue(), "image/png", "png"
    if output_format == "pdf":
        from reportlab.pdfgen import canvas
        import io
        stream = io.BytesIO(); pdf = canvas.Canvas(stream, pagesize=(900, max(180, 100 + len(payload.nodes) * 70)))
        for index, node in enumerate(payload.nodes):
            x, y = 40 + (index % 3) * 290, 40 + (index // 3) * 100
            pdf.rect(x, y, 220, 50); pdf.drawString(x + 10, y + 22, node["label"])
        pdf.save(); return stream.getvalue(), "application/pdf", "pdf"
    raise ValueError("Unsupported diagram format")
