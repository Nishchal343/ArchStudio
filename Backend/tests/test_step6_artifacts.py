import io
import asyncio
from uuid import uuid4

import pytest

from app.schemas import DiagramPayload, EditableDocument, ReportSection
from app.services.artifacts.diagrams import database_diagram, render_diagram
from app.services.artifacts.export import export_document
from app.services.artifacts.reports import _clean_model_section, _finalize_report_sections
from app.services.chat.service import answer_project_question
from app.services.artifacts.export import _section_break_indices


def document():
    return EditableDocument(title="Test report", document_type="report",
                            sections=[ReportSection(heading="Overview", content="Actual repository evidence.")])


def test_report_pagination_does_not_insert_artificial_section_breaks():
    assert _section_break_indices(12, 10) == set()


def test_report_rejects_raw_retrieval_excerpts():
    fallback = "The architecture separates the application into coordinated backend and persistence layers."
    generated = "The architecture is organized around services. Evidence detail from the indexed repository: FILE: Backend/app/services/llm/base.py:1-11 class LLMProvider: pass"
    assert _clean_model_section(generated, fallback) == fallback


def test_report_keeps_user_research_sections_as_placeholders():
    sections = [("Literature Review", "Unrelated repository source code."), ("System Architecture", "Relevant architecture prose.")]
    fallback = {
        "literature review": "USER INPUT REQUIRED — Enter the literature review here.",
        "system architecture": "Fallback architecture prose.",
    }
    finalized = _finalize_report_sections(sections, fallback)
    assert finalized[0][1].startswith("USER INPUT REQUIRED")
    assert finalized[1][1] == "Relevant architecture prose."


def test_document_exports_are_real_files():
    pdf, pdf_type, _ = export_document(document(), "pdf")
    docx, docx_type, _ = export_document(document(), "docx")
    assert pdf.startswith(b"%PDF")
    assert pdf_type == "application/pdf"
    assert docx[:2] == b"PK"
    assert "wordprocessingml" in docx_type


def test_database_diagram_reports_absence_without_schema():
    payload = database_diagram(uuid4())
    assert "No database schema detected" in payload.notes[0]
    assert "No database schema detected" in payload.mermaid


def test_diagram_renders_locally():
    payload = DiagramPayload(title="Architecture", diagram_type="architecture_diagram", mermaid="flowchart TD",
                             nodes=[{"id": "a", "label": "app"}], edges=[])
    svg, svg_type, _ = render_diagram(payload, "svg")
    png, png_type, _ = render_diagram(payload, "png")
    pdf, pdf_type, _ = render_diagram(payload, "pdf")
    assert svg.startswith(b"<svg") and svg_type == "image/svg+xml"
    assert png.startswith(b"\x89PNG") and png_type == "image/png"
    assert pdf.startswith(b"%PDF") and pdf_type == "application/pdf"


class FakeProvider:
    async def generate(self, prompt):
        assert "FILE: app/auth.py:10-20" in prompt
        return '{"answer":"Authentication is implemented in the cited file.","evidence":[{"file_path":"app/auth.py"}]}'


def test_chat_is_grounded_and_returns_file_evidence(monkeypatch):
    async def fake_search(*args, **kwargs):
        return [{"file_path": "app/auth.py", "start_line": 10, "end_line": 20,
                 "similarity": 0.8, "content": "def authenticate(): pass"}]
    monkeypatch.setattr("app.services.chat.service.search_project", fake_search)
    result = asyncio.run(answer_project_question(None, uuid4(), "Where is auth?", 5, FakeProvider()))
    assert result.answer.startswith("Authentication")
    assert result.evidence[0].file_path == "app/auth.py"


def test_chat_retries_vague_insufficient_evidence_answer(monkeypatch):
    async def fake_search(*args, **kwargs):
        return [{"file_path": "frontend/src/LoadingOverlay.jsx", "start_line": 10, "end_line": 20,
                 "similarity": 0.8, "content": "if (loading) return null"}]

    class RetryProvider:
        def __init__(self):
            self.calls = 0

        async def generate(self, prompt):
            self.calls += 1
            if self.calls == 1:
                return '{"answer":"Insufficient evidence","evidence":[]}'
            return '{"answer":"The loading state should remain visible while data is being fetched.","evidence":[{"file_path":"frontend/src/LoadingOverlay.jsx"}]}'

    monkeypatch.setattr("app.services.chat.service.search_project", fake_search)
    provider = RetryProvider()
    result = asyncio.run(answer_project_question(None, uuid4(), "What improvements can be made?", 5, provider))
    assert provider.calls == 2
    assert "loading state" in result.answer
