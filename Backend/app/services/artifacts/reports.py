import asyncio
from uuid import UUID
import json
import re

from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession

from ...models import ProjectAnalysis
from ...schemas import AnalysisResult, EditableDocument, ReportSection
from ...database import SessionLocal
from ..analysis.service import analyze_project
from ..storage_service import project_directory
from ..rag.retrieval import search_project
from ..llm.service import get_llm_provider
from ...config import get_settings
from .templates import predefined_template

USER_INPUT_SECTIONS = {
    "Literature Review": "USER INPUT REQUIRED — Enter the literature review here.",
    "Research Gap & Motivation": "USER INPUT REQUIRED — Enter the research gap and motivation here.",
    "Comparison with Existing System": "USER INPUT REQUIRED — Enter the comparison with the existing system here.",
    "References": "USER INPUT REQUIRED — Enter references here.",
}


async def _analysis_for_project(db: AsyncSession, project_id: UUID) -> AnalysisResult:
    result = await db.execute(select(ProjectAnalysis).where(
        ProjectAnalysis.project_id == project_id,
        ProjectAnalysis.status == "completed",
    ).order_by(desc(ProjectAnalysis.created_at)).limit(1))
    row = result.scalar_one_or_none()
    if row and row.result and any(row.result.get(field) for field in ("overview", "summary", "architecture", "components")):
        return AnalysisResult.model_validate(row.result)
    return await analyze_project(db, project_id)


def _text(value) -> str:
    if isinstance(value, list):
        return "\n".join(f"- {item}" for item in value)
    if isinstance(value, dict):
        return "\n".join(f"{key}: {val}" for key, val in value.items())
    return str(value or "Evidence unavailable.")


def _items(items: list[dict]) -> str:
    if not items:
        return "Evidence unavailable."
    entries = []
    for item in items:
        label = item.get("title") or item.get("name")
        detail = item.get("description") or item.get("purpose") or item.get("relationship")
        if not detail:
            detail = "; ".join(f"{key.replace('_', ' ')}: {value}" for key, value in item.items() if value)
        if detail:
            entries.append(f"{label}: {detail}" if label and label != detail else str(detail))
    return " ".join(entries) if entries else "Evidence unavailable."


def _prose(value, subject: str = "The project") -> str:
    """Turn structured analysis values into report prose rather than raw lists."""
    if isinstance(value, str):
        return value.strip() or "Evidence unavailable."
    if isinstance(value, list):
        text = _items(value) if value and isinstance(value[0], dict) else ", ".join(str(item) for item in value if item)
        return f"{subject} includes {text}." if text and not text.endswith(".") else text
    if isinstance(value, dict):
        return "; ".join(f"{key.replace('_', ' ')} is {val}" for key, val in value.items() if val) + "."
    return str(value or "Evidence unavailable.")


async def _template_sections(db: AsyncSession, project_id: UUID, template: dict, analysis: AnalysisResult) -> list[tuple[str, str]]:
    settings = get_settings()
    fallback = {heading.lower(): _text(content) for heading, content in _default_sections(analysis)}
    semaphore = asyncio.Semaphore(4)

    async def render_section(item: dict) -> tuple[str, str]:
        heading = str(item.get("heading", "")).strip()
        if not heading:
            return "", ""
        try:
            async with semaphore:
                # Each concurrent section gets its own DB session. AsyncSession is
                # not safe to share across concurrent retrievals.
                async with SessionLocal() as section_db:
                    results = await search_project(section_db, project_id, heading, top_k=4,
                                                   min_similarity=settings.analysis_min_similarity)
                context = "\n\n".join(f"FILE: {r['file_path']}:{r.get('start_line')}-{r.get('end_line')}\n{r['content']}" for r in results)
                prompt = ("Write one report section using only the project evidence below. The predefined application format supplies "
                          "headings and layout only; never copy template facts, authors, citations, metrics, or claims. "
                          "If evidence is insufficient, state that clearly. Return JSON only: {\"content\": \"...\"}.\n\n"
                          f"SECTION: {heading}\nPROJECT EVIDENCE:\n{context or '[No supporting evidence found.]'}")
                raw = await get_llm_provider().generate(prompt)
                parsed = json.loads(raw)
                content = parsed.get("content") if isinstance(parsed, dict) else None
                if not isinstance(content, str) or not content.strip():
                    raise ValueError("empty section")
        except Exception:
            content = fallback.get(heading.lower(), "This section is not supported by the available project evidence.")
        return heading, content

    rendered = await asyncio.gather(*(render_section(item) for item in template.get("sections", [])))
    return [(heading, content) for heading, content in rendered if heading]


def _default_sections(analysis: AnalysisResult) -> list[tuple[str, object]]:
    return [("Project Overview", analysis.overview), ("Abstract", analysis.summary),
            ("Technology Stack", analysis.technology_stack), ("Project Structure", analysis.project_structure),
            ("System Architecture", analysis.architecture), ("Modules and Components", analysis.components),
            ("Implementation Details", analysis.dependencies), ("Security Observations", analysis.security_findings),
            ("Future Improvements", analysis.recommendations), ("Conclusion", analysis.summary)]


def _normal_report_sections(analysis: AnalysisResult) -> list[tuple[str, object]]:
    """Populate the supplied project-report structure from the full analysis.

    The report front matter is structural, while the technical chapters should
    remain useful even when a heading-specific vector search returns no hit.
    No author, supervisor, citation, result, or institutional detail is
    invented here.
    """
    technology = _prose(analysis.technology_stack, "The technology foundation")
    components = _prose(analysis.components, "The implementation")
    relationships = _prose(analysis.component_relationships, "The components")
    dependencies = _prose(analysis.dependencies, "The project depends on")
    quality = _prose(analysis.code_quality + analysis.maintainability_findings + analysis.code_smells, "The implementation quality review identified")
    security = _prose(analysis.security_findings, "The security review identified")
    performance = _prose(analysis.performance_findings, "The performance review identified")
    recommendations = _prose(analysis.recommendations, "The evidence-based improvement plan includes")
    headings = [
        ("Abstract", analysis.summary + "\n\nProject context: " + analysis.overview),
        ("Introduction", analysis.overview + "\n\nProject structure and implementation context:\n" + analysis.project_structure),
        ("Literature Review", USER_INPUT_SECTIONS["Literature Review"]),
        ("Research Gap & Motivation", USER_INPUT_SECTIONS["Research Gap & Motivation"]),
        ("Proposed Methodology", "The proposed methodology is grounded in the observed project architecture and implementation:\n\n" + analysis.architecture),
        ("System Architecture", analysis.architecture + "\n\nComponents:\n" + components + "\n\nRelationships:\n" + relationships),
        ("Implementation Technology", technology + "\n\nDependencies and supporting elements:\n" + dependencies),
        ("Comparison with Existing System", USER_INPUT_SECTIONS["Comparison with Existing System"]),
        ("Results and Discussion", "The analysis identified the following implementation results and observations:\n\n" + quality + "\n\nSecurity observations:\n" + security + "\n\nPerformance observations:\n" + performance),
        ("Future Scope", recommendations),
        ("Conclusion", analysis.summary + "\n\nOverall architecture:\n" + analysis.architecture),
        ("References", USER_INPUT_SECTIONS["References"]),
    ]
    return headings


def _ieee_report_sections(analysis: AnalysisResult) -> list[tuple[str, object]]:
    technology = _prose(analysis.technology_stack, "The implementation uses")
    components = _prose(analysis.components, "The principal implementation components are")
    relationships = _prose(analysis.component_relationships, "The component relationships indicate")
    dependencies = _prose(analysis.dependencies, "The supporting dependencies are")
    quality = _prose(analysis.code_quality + analysis.maintainability_findings + analysis.code_smells, "The code review identified")
    security = _prose(analysis.security_findings, "The security review identified")
    performance = _prose(analysis.performance_findings, "The performance review identified")
    recommendations = _prose(analysis.recommendations, "The recommended improvements are")
    return [
        ("Abstract", analysis.summary + " " + analysis.overview + " The discussion is limited to behavior and structure supported by the analyzed repository. Where the source does not provide measurable experimental results, the report does not present invented metrics."),
        ("Keywords", technology + " These terms describe the technologies and implementation concerns that are actually represented in the project evidence."),
        ("Introduction", "This paper presents a grounded technical description of the project and the engineering decisions visible in its source repository. " + analysis.overview + " The purpose of this section is to establish the project context before examining its structure, implementation approach, and limitations."),
        ("Ease of Use", "The available evidence describes the implementation rather than a formal usability study. The observable user-facing behavior is therefore discussed only where it is represented in the source. A complete usability assessment would require documented user flows, test participants, and measured outcomes, none of which are asserted here."),
        ("Selecting a Template", "This section is interpreted as the document-format guidance from the IEEE template, not as a project feature. The generated paper uses the predefined IEEE structure and applies the project evidence to each heading. Template examples, authors, citations, and instructional claims are not copied into the generated content."),
        ("Maintaining the Integrity of the Specifications", "The report preserves the required IEEE section order and uses the supplied format as a structural guide. Technical claims remain bounded by repository evidence. In particular, missing measurements, external references, author information, and experimental validation are identified as unavailable rather than being filled with plausible-looking values."),
        ("Prepare Your Paper Before Styling", "The project evidence was analyzed before the document was assembled. This separates content generation from document formatting and makes it possible to explain the implementation independently of the template styling. The resulting sections describe the project first and apply the IEEE presentation structure second."),
        ("Abbreviations and Acronyms", "The abbreviations relevant to the analyzed project are retained only when they are evident from the technology or source context. " + technology + " No additional acronym definitions are introduced without supporting evidence."),
        ("Units", "The repository analysis does not provide a validated measurement study for the project. Consequently, this paper does not invent units, benchmark values, timing measurements, or physical quantities. Any numerical behavior mentioned elsewhere must be read as source-level behavior only when directly supported by the analyzed code."),
        ("Equations", "No independently validated mathematical model or experiment was established by the available project analysis. The implementation may contain calculations used for control flow, rendering, timing, or data transformation, but this report does not elevate those expressions into formal research equations without sufficient source context and validation."),
        ("Some Common Mistakes", "The main risks identified in the implementation are described in the code-quality and maintainability review. " + quality + " These observations concern maintainability and engineering practice; they are not presented as performance benchmarks or security certifications."),
        ("Using the Template", "The IEEE format is used to organize the project evidence into a consistent technical narrative. " + components + " The format improves readability, but it does not change the underlying implementation or add capabilities that are not present in the repository."),
        ("Authors and Affiliations", "Author names, affiliations, email addresses, and ORCID identifiers were not available in the project evidence. The section is retained because it belongs to the supplied IEEE format, but no personal or institutional information is fabricated."),
        ("Identify the Headings", "The headings in this paper are taken from the predefined IEEE structure. Each heading is populated with project-specific discussion where evidence supports it, and limitations are stated directly where the repository does not support a stronger conclusion."),
        ("Figures and Tables", "No verified figures or tables were available as part of the analyzed project evidence. The architecture, components, and relationships are therefore described in prose. Figures or measurements should be added only when they can be generated from verified project data."),
        ("Acknowledgment", "This paper acknowledges the project repository and the available source evidence used for the analysis. No external contributor, sponsor, institution, or funding agency is named because that information was not available in the project evidence."),
        ("References", "No external references were available in the analyzed project evidence. Repository files serve as primary implementation evidence for this generated paper, while external citations remain intentionally absent rather than fabricated."),
    ]


async def generate_report(db: AsyncSession, project_id: UUID, document_type: str) -> EditableDocument:
    analysis = await _analysis_for_project(db, project_id)

    template = predefined_template(document_type)
    fallback = {
        "abstract": analysis.summary or analysis.overview,
        "introduction": analysis.overview or analysis.project_structure,
        "literature review": "The available repository evidence does not contain external literature; this section requires the supplied reference papers for a literature-grounded discussion.",
        "research gap & motivation": "The repository evidence describes the implemented system but does not establish a validated research gap; the gap should be refined against the reference literature.",
        "proposed methodology": analysis.overview + "\n\n" + analysis.architecture,
        "system architecture": analysis.architecture + "\n\n" + _prose(analysis.components, "Components"),
        "implementation technology": _prose(analysis.technology_stack, "The implementation uses"),
        "comparison with existing system": "A comparison requires the supplied reference papers. The repository evidence alone supports describing this implementation, not claiming superiority over an unobserved system.",
        "results and discussion": "Verified implementation observations:\n" + _prose(analysis.performance_findings, "Performance evidence"),
        "future scope": _prose(analysis.recommendations, "Evidence-based future improvements"),
        "conclusion": analysis.summary or analysis.overview,
        "references": "The analyzed repository is the available primary source. No external citation is added without supplied reference material.",
    }
    settings = get_settings()
    headings = [item["heading"] for item in template["sections"]]

    async def retrieve_for_heading(heading: str) -> tuple[str, str]:
        async with SessionLocal() as section_db:
            results = await search_project(section_db, project_id, heading, top_k=6,
                                           min_similarity=settings.analysis_min_similarity)
        context = "\n\n".join(
            f"FILE: {r['file_path']}:{r.get('start_line')}-{r.get('end_line')}\n{r['content']}"
            for r in results
        )
        return heading, context or "[No heading-specific evidence found; use the complete analysis.]"

    # Retrieval can remain heading-focused, but the LLM receives every heading
    # and all evidence in one request so it can write a coherent paper with
    # consistent depth and transitions instead of isolated one-line sections.
    retrieved = dict(await asyncio.gather(*(retrieve_for_heading(heading) for heading in headings)))
    evidence = "\n\n".join(f"## {heading}\n{retrieved[heading]}" for heading in headings)
    analysis_json = json.dumps(analysis.model_dump(mode="json"), ensure_ascii=False)
    heading_json = json.dumps(headings, ensure_ascii=False)
    prompt = f"""Write the complete {document_type} as one coherent grounded paper.
Return exactly one valid JSON object with this shape:
{{"sections": [{{"heading": "exact heading from the list", "content": "substantial section prose"}}]}}

Use every heading exactly once, preserve the exact order below, and do not add, remove, rename, or reorder headings.
HEADINGS: {heading_json}

Target layout: {template['layout']['columns']} column(s).
Target length: {template['layout']['page_target']} and approximately {template['layout']['target_words']} words of actual content. Write enough connected prose to reach that target; do not pad with blank lines or repeat the same point.
Writing style: {template['layout']['depth']}. Make the sections flow as one paper, with useful paragraphs rather than one-line summaries.

Use only the supplied repository analysis and evidence. Do not invent metrics, citations, technologies, features, external papers, authors, or experimental results. For Literature Review, Research Gap & Motivation, Comparison with Existing System, and References, explicitly explain when external evidence is unavailable while still providing relevant repository-grounded discussion.

COMPLETE PROJECT ANALYSIS:
{analysis_json}

HEADING-RELATED REPOSITORY EVIDENCE:
{evidence}
"""
    try:
        raw = await get_llm_provider().generate(prompt, max_tokens=template["layout"]["output_tokens"])
        parsed = json.loads(raw)
        generated = parsed.get("sections") if isinstance(parsed, dict) else None
        by_heading = {
            str(item.get("heading", "")).strip().lower(): item.get("content")
            for item in (generated or []) if isinstance(item, dict)
        }
    except Exception:
        by_heading = {}

    first_pass_sections = [
        (heading, by_heading.get(heading.lower()) if isinstance(by_heading.get(heading.lower()), str)
         and by_heading[heading.lower()].strip() else _text(fallback.get(heading.lower(), "Evidence unavailable.")))
        for heading in headings
    ]

    # A second pass expands only the content produced by the first pass. It
    # does not receive repository evidence or any new instructions about the
    # project, so it can improve continuity and paragraph length without
    # introducing unsupported facts.
    first_pass_json = json.dumps(
        [{"heading": heading, "content": content} for heading, content in first_pass_sections],
        ensure_ascii=False,
    )
    expansion_prompt = f"""Rewrite the supplied report content for presentation.
Return exactly one valid JSON object with this shape:
{{"sections": [{{"heading": "same heading", "content": "exactly two paragraphs"}}]}}

Use every supplied heading exactly once, in the same order. For every heading, write exactly two substantial paragraphs separated by one blank line. Preserve the meaning and facts of the supplied content. Expand wording and improve transitions only; do not add facts, citations, metrics, technologies, headings, placeholders, or information from outside the supplied content. Return no title, commentary, markdown fences, or content outside the JSON object.

FIRST-PASS CONTENT ONLY:
{first_pass_json}
"""
    sections = first_pass_sections
    try:
        expanded_raw = await get_llm_provider().generate(
            expansion_prompt, max_tokens=template["layout"]["output_tokens"]
        )
        expanded_payload = json.loads(expanded_raw)
        expanded_items = expanded_payload.get("sections") if isinstance(expanded_payload, dict) else None
        expanded_by_heading = {
            str(item.get("heading", "")).strip().lower(): item.get("content")
            for item in (expanded_items or []) if isinstance(item, dict)
        }
        expanded_sections = []
        for heading, original_content in first_pass_sections:
            content = expanded_by_heading.get(heading.lower())
            if isinstance(content, str) and content.strip():
                paragraphs = [part.strip() for part in re.split(r"\n\s*\n", content) if part.strip()]
                if len(paragraphs) == 2:
                    expanded_sections.append((heading, "\n\n".join(paragraphs)))
                else:
                    expanded_sections.append((heading, original_content))
            else:
                expanded_sections.append((heading, original_content))
        if len(expanded_sections) == len(first_pass_sections):
            sections = expanded_sections
    except Exception:
        pass
    return EditableDocument(title="USER INPUT REQUIRED — Enter paper title", document_type=document_type,
                            authors=[{"name": "USER INPUT REQUIRED — Enter author name", "institution": ""}],
                            date="",
                            sections=[ReportSection(heading=h, content=_text(c)) for h, c in sections],
                            layout=template["layout"])
