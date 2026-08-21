import io
import json
from typing import Any

import httpx
from pypdf import PdfReader
from sqlalchemy import desc, select

from .llm.service import get_llm_provider
from .rag.retrieval import search_project
from ..config import get_settings
from ..database import SessionLocal
from ..models import ProjectAnalysis
from ..schemas import AnalysisResult

MAX_PAPER_TEXT_CHARS = 6000
MAX_PROJECT_EVIDENCE_CHARS = 4000


def _json_object(text: str) -> dict:
    """Accept strict JSON, fenced JSON, or JSON followed by a short explanation."""
    candidate = (text or "").strip()
    if candidate.startswith("```"):
        candidate = candidate[3:]
        if candidate.lstrip().startswith("json"):
            candidate = candidate.lstrip()[4:]
        candidate = candidate.strip().rstrip("`").strip()
    try:
        value = json.loads(candidate)
    except json.JSONDecodeError:
        start = candidate.find("{")
        if start < 0:
            raise ValueError("LLM returned invalid structured analysis")
        value, _ = json.JSONDecoder().raw_decode(candidate[start:])
    if not isinstance(value, dict):
        raise ValueError("LLM returned invalid structured analysis")
    return value


async def _extract_file(filename: str, content: bytes) -> str:
    if filename.lower().endswith(".pdf"):
        reader = PdfReader(io.BytesIO(content))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    raise ValueError(f"Unsupported reference file: {filename}. Upload PDF files.")


async def _extract_url(url: str) -> str:
    async with httpx.AsyncClient(follow_redirects=True, timeout=30) as client:
        response = await client.get(url)
        response.raise_for_status()
    content_type = response.headers.get("content-type", "").lower()
    if "pdf" in content_type or url.lower().split("?")[0].endswith(".pdf"):
        return "\n".join(page.extract_text() or "" for page in PdfReader(io.BytesIO(response.content)).pages)
    # Public HTML papers are accepted; strip the common markup before analysis.
    from html import unescape
    import re
    return unescape(re.sub(r"<[^>]+>", " ", response.text))


async def generate_literature_review(db, project_id, urls: list[str], files: list[Any]) -> dict:
    papers = []
    for index, url in enumerate(urls, 1):
        try:
            text = await _extract_url(url.strip())
            if not text.strip():
                raise ValueError("No extractable text")
            papers.append({"label": f"Reference Paper {index}", "source": url, "content": text[:MAX_PAPER_TEXT_CHARS]})
        except Exception as error:
            papers.append({"label": f"Reference Paper {index}", "source": url, "error": f"Paper could not be accessed or extracted: {error}"})
    offset = len(papers)
    for index, upload in enumerate(files, 1):
        try:
            text = await _extract_file(upload.filename or f"paper-{index}.pdf", await upload.read())
            if not text.strip():
                raise ValueError("No extractable text")
            papers.append({"label": f"Reference Paper {offset + index}", "source": upload.filename, "content": text[:MAX_PAPER_TEXT_CHARS]})
        except Exception as error:
            papers.append({"label": f"Reference Paper {offset + index}", "source": upload.filename, "error": f"Paper could not be extracted: {error}"})
    if not papers:
        raise ValueError("Provide at least one public paper URL or PDF file.")
    if not any(paper.get("content") for paper in papers):
        raise ValueError(
            "No reference paper text could be extracted. IEEE Xplore pages often require access beyond a public URL; "
            "download the paper as a PDF and upload it instead."
        )

    # Reuse the completed analysis saved during project setup. Calling the
    # analysis model again here unnecessarily consumes another Groq request
    # before the actual literature-review request is made.
    analysis_row = (await db.execute(
        select(ProjectAnalysis)
        .where(ProjectAnalysis.project_id == project_id, ProjectAnalysis.status == "completed")
        .order_by(desc(ProjectAnalysis.created_at))
        .limit(1)
    )).scalars().first()
    if analysis_row and analysis_row.result:
        try:
            analysis = AnalysisResult.model_validate(analysis_row.result)
        except Exception:
            analysis = AnalysisResult(
                project_id=project_id,
                overview="The saved project analysis was unavailable; use only the retrieved repository evidence below.",
            )
    else:
        analysis = AnalysisResult(
            project_id=project_id,
            overview="No saved project analysis was available; use only the retrieved repository evidence below.",
        )
    settings = get_settings()
    async with SessionLocal() as evidence_db:
        hits = await search_project(evidence_db, project_id, "architecture methodology algorithms technologies results limitations", top_k=12,
                                    min_similarity=settings.analysis_min_similarity)
    evidence_parts = []
    evidence_size = 0
    for hit in hits:
        part = f"FILE: {hit['file_path']}\n{hit['content']}"
        remaining = MAX_PROJECT_EVIDENCE_CHARS - evidence_size
        if remaining <= 0:
            break
        evidence_parts.append(part[:remaining])
        evidence_size += min(len(part), remaining)
    project_evidence = "\n\n".join(evidence_parts)
    paper_evidence = "\n\n".join(json.dumps(paper, ensure_ascii=False) for paper in papers)
    analysis_summary = json.dumps({
        "overview": analysis.overview,
        "architecture": analysis.architecture,
        "technology_stack": analysis.technology_stack,
        "summary": analysis.summary,
        "recommendations": analysis.recommendations,
    }, ensure_ascii=False)
    prompt = f"""Analyze the supplied reference papers and compare them with the selected project's extracted repository code.
Distinguish each paper. Never treat unavailable or failed extraction as evidence. Never claim the project implements a paper technique unless the project evidence supports it.
Return JSON with exactly these keys: literature_review, research_gap_motivation, comparison_existing_system, paper_analyses.
The first three values must be polished, copy-ready academic prose. comparison_existing_system may include a Markdown table. paper_analyses must preserve one entry per supplied paper with problem, motivation, objectives, methodology, approach, architecture, algorithms, technologies, results, contributions, limitations, future_work, and access_status.

REFERENCE PAPERS:
{paper_evidence}

SELECTED PROJECT ANALYSIS:
        {analysis_summary}

EXTRACTED PROJECT CODE EVIDENCE:
{project_evidence or '[No additional retrieved evidence available.]'}"""
    provider = get_llm_provider()
    # Groq's on-demand tier has an 8k tokens-per-minute request limit. Keep
    # the combined prompt plus completion reservation comfortably below it.
    raw = await provider.generate(prompt, max_tokens=3500)
    try:
        result = _json_object(raw)
    except ValueError:
        # Ask once more with a smaller, stricter contract. Some providers
        # return useful prose or fenced JSON despite the JSON response mode.
        retry_prompt = f"""Return only one valid JSON object with exactly these keys:
literature_review, research_gap_motivation, comparison_existing_system, paper_analyses.
Write concise evidence-based content for the supplied reference papers and project. Do not use Markdown fences or commentary.
REFERENCE PAPERS:
{paper_evidence}
PROJECT EVIDENCE:
{project_evidence or '[No project evidence available.]'}"""
        try:
            result = _json_object(await provider.generate(retry_prompt, max_tokens=2500))
        except Exception:
            result = {
                "literature_review": "The supplied reference paper was accessed, but the AI response could not be parsed into a structured review. Re-run the analysis to generate the full literature synthesis.",
                "research_gap_motivation": "The research gap must be established by comparing the supplied paper with verified project evidence; no unsupported gap is asserted.",
                "comparison_existing_system": "The comparison could not be structured from the current AI response. Re-run the analysis to generate the evidence-based comparison.",
                "paper_analyses": [{
                    "problem": "See the supplied reference paper.",
                    "motivation": "See the supplied reference paper.",
                    "objectives": "See the supplied reference paper.",
                    "methodology": "Unavailable from the parsed response.",
                    "approach": "Unavailable from the parsed response.",
                    "architecture": "Unavailable from the parsed response.",
                    "algorithms": "Unavailable from the parsed response.",
                    "technologies": "Unavailable from the parsed response.",
                    "results": "Unavailable from the parsed response.",
                    "contributions": "Unavailable from the parsed response.",
                    "limitations": "Unavailable from the parsed response.",
                    "future_work": "Unavailable from the parsed response.",
                    "access_status": paper.get("error") or "available",
                } for paper in papers],
            }
    result.setdefault("literature_review", "")
    result.setdefault("research_gap_motivation", "")
    result.setdefault("comparison_existing_system", "")
    result.setdefault("paper_analyses", [])
    result["sources"] = [{"label": p["label"], "source": p["source"], "access_status": "available" if p.get("content") else p.get("error")} for p in papers]
    return result
