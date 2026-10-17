import asyncio
import re
from collections.abc import Awaitable, Callable
from uuid import UUID
import json
import logging

from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession

from ...models import ProjectAnalysis
from ...schemas import AnalysisResult, EditableDocument, ReportSection
from ...database import SessionLocal
from ..analysis.service import analyze_project
from ..storage_service import project_directory
from ..rag.retrieval import search_project
from ..llm.base import LLMGeneration, LLMProviderError
from ..llm.service import get_llm_provider
from ...config import get_settings
from .templates import predefined_template

logger = logging.getLogger(__name__)

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


def _bounded_prompt_text(value: str, limit: int) -> str:
    text = str(value or "").strip()
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + " [additional detail omitted from prompt]"


def _analysis_prompt_payload(analysis: AnalysisResult) -> dict:
    """Keep each section request below provider token limits."""
    payload = analysis.model_dump(mode="json")
    for key, value in list(payload.items()):
        if isinstance(value, str):
            payload[key] = _bounded_prompt_text(value, 2800)
        elif isinstance(value, list):
            # The complete analysis remains available to the fallback report;
            # the LLM only needs representative structured findings.
            payload[key] = value[:6]
    return payload


_USER_INPUT_HEADING_KEYS = {heading.lower() for heading in USER_INPUT_SECTIONS}


def _looks_like_source_dump(content: str) -> bool:
    """Detect source excerpts that must never become report prose."""
    if re.search(r"\bFILE:\s*[^\s]+", content, flags=re.IGNORECASE):
        return True
    if "className=" in content or "<div" in content or "</" in content:
        return True
    code_markers = (
        r"\bfrom\s+[\w.]+\s+import\s+",
        r"\bdef\s+\w+\s*\(",
        r"\bclass\s+\w+\s*[:({]",
        r"\bfunction\s+\w+\s*\(",
        r"\b(?:const|let|var)\s+\w+\s*=",
        r"@(?:abstractmethod|app\.|router\.)",
    )
    return sum(bool(re.search(marker, content)) for marker in code_markers) >= 2


def _clean_model_section(content: str, fallback: str) -> str:
    """Keep generated prose while rejecting copied evidence/code blocks.

    Repository retrieval is grounding input, not report content. In
    particular, a model must not paste FILE blocks, Python, JSX, or HTML into
    a section just because that section was short.
    """
    text = str(content or "").strip()
    if not text:
        return fallback

    text = re.sub(r"```(?:\w+)?\s*.*?```", "", text, flags=re.IGNORECASE | re.DOTALL).strip()
    evidence_marker = re.search(
        r"\b(?:Evidence detail from the indexed repository|PROJECT EVIDENCE|"
        r"HEADING-RELATED REPOSITORY EVIDENCE)\s*:",
        text,
        flags=re.IGNORECASE,
    )
    if evidence_marker:
        text = text[:evidence_marker.start()].strip()
        # Do not keep a short lead-in after removing the copied evidence; use
        # the complete fallback for that heading instead.
        if len(text.split()) < 20:
            return fallback

    if _looks_like_source_dump(text):
        return fallback
    return text or fallback


def _finalize_report_sections(
    sections: list[tuple[str, str]],
    fallback: dict[str, str],
) -> list[tuple[str, str]]:
    """Apply section-specific fallback and protect user-editable sections."""
    finalized: list[tuple[str, str]] = []
    for heading, content in sections:
        key = heading.lower()
        fallback_content = fallback.get(key, "Evidence unavailable.")
        # These sections require user-provided literature/comparison material.
        # Repository code is not a valid substitute and often produces
        # unrelated paragraphs when the model tries to fill the space.
        if key in _USER_INPUT_HEADING_KEYS:
            finalized.append((heading, fallback_content))
            continue
        finalized.append((heading, _clean_model_section(content, fallback_content)))
    return finalized


class ReportGenerationError(LLMProviderError):
    """A report section failed after bounded retries, with partial output kept."""

    def __init__(self, heading: str, message: str, partial_document: EditableDocument):
        super().__init__(f"Report section '{heading}' could not be generated: {message}")
        self.heading = heading
        self.partial_document = partial_document


def _section_word_target(document_type: str, heading: str) -> tuple[int, int]:
    """Return target/minimum words for one generated section."""
    if document_type == "ieee_report":
        targets = {"Abstract": (160, 90), "Conclusion": (220, 130)}
        return targets.get(heading, (220, 130))
    targets = {"Abstract": (220, 120), "Conclusion": (320, 200)}
    return targets.get(heading, (500, 300))


def _section_output_tokens(document_type: str, heading: str, target_words: int) -> int:
    """Leave room for JSON syntax while bounding one request's completion."""
    maximum = 1200 if document_type == "ieee_report" else 1600
    return min(maximum, max(500, int(target_words * 1.8)))


def _section_analysis_context(analysis: AnalysisResult, heading: str) -> str:
    fields = {
        "Abstract": ("summary", "overview"),
        "Introduction": ("overview", "project_structure"),
        "Proposed Methodology": ("architecture", "project_structure", "component_relationships"),
        "System Architecture": ("architecture", "components", "component_relationships"),
        "Implementation Technology": ("technology_stack", "dependencies"),
        "Results and Discussion": ("code_quality", "security_findings", "performance_findings", "maintainability_findings"),
        "Future Scope": ("recommendations", "maintainability_findings"),
        "Conclusion": ("summary", "architecture", "recommendations"),
    }.get(heading, ("overview", "architecture"))
    payload = _analysis_prompt_payload(analysis)
    return json.dumps({field: payload.get(field) for field in fields}, ensure_ascii=False)


def _previous_report_context(sections: list[tuple[str, str]]) -> str:
    if not sections:
        return "[No previous generated sections.]"
    return "\n\n".join(
        f"{heading}: {_bounded_prompt_text(content, 650)}"
        for heading, content in sections[-2:]
    )


def _section_failure_document(
    document_type: str,
    layout: dict,
    headings: list[str],
    sections: list[tuple[str, str]],
    failed_heading: str,
    failure_message: str,
    fallback: dict[str, str],
) -> EditableDocument:
    completed = {heading.lower(): content for heading, content in sections}
    partial_sections = []
    for heading in headings:
        if heading.lower() in completed:
            content = completed[heading.lower()]
        elif heading == failed_heading:
            content = f"Section generation failed: {failure_message}"
        elif heading.lower() in _USER_INPUT_HEADING_KEYS:
            content = fallback.get(heading.lower(), USER_INPUT_SECTIONS[heading])
        else:
            content = "Not generated because an earlier report section failed."
        partial_sections.append(ReportSection(heading=heading, content=_text(content)))
    return EditableDocument(
        title="USER INPUT REQUIRED â€” Enter paper title",
        document_type=document_type,
        authors=[{"name": "USER INPUT REQUIRED â€” Enter author name", "institution": ""}],
        date="",
        sections=partial_sections,
        layout=layout,
    )


async def _retrieve_section_evidence(
    project_id: UUID,
    heading: str,
    settings,
) -> str:
    async with SessionLocal() as section_db:
        results = await search_project(
            section_db,
            project_id,
            heading,
            top_k=4,
            min_similarity=settings.analysis_min_similarity,
        )
    evidence = []
    for result in results[:4]:
        evidence.append(
            f"FILE: {result['file_path']}:{result.get('start_line')}-{result.get('end_line')}\n"
            f"{_bounded_prompt_text(result.get('content', ''), 1400)}"
        )
    return "\n\n".join(evidence) or "[No heading-specific evidence found.]"


def _section_prompt(
    document_type: str,
    heading: str,
    target_words: int,
    minimum_words: int,
    analysis_context: str,
    evidence: str,
    previous_context: str,
) -> str:
    return f"""Generate only the report section named '{heading}' for a {document_type}.
Return exactly one valid JSON object and no Markdown:
{{"content": "section prose only"}}

The section target is approximately {target_words} words, with at least {minimum_words} meaningful words unless the supplied evidence genuinely cannot support that length. Use connected academic prose with paragraphs. Explain the section's topic, relevant implementation details, repository-grounded evidence, limitations, and examples where supported. Avoid repetition and filler.

Use only the supplied project analysis and evidence. Do not invent metrics, citations, external papers, authors, features, or implementation details. Repository evidence is grounding input only: never copy FILE blocks, file paths with source excerpts, Python, JavaScript, JSX, HTML, configuration, JSON, or fenced code. Do not discuss topics belonging to another heading. Do not generate the user's reserved research sections; those are handled by the application.

PROJECT ANALYSIS RELEVANT TO THIS SECTION:
{analysis_context}

SECTION-SPECIFIC REPOSITORY EVIDENCE:
{evidence}

LIMITED PREVIOUS REPORT CONTEXT:
{previous_context}
"""


def _parse_section_generation(
    generation: LLMGeneration,
    heading: str,
    minimum_words: int,
) -> str:
    if generation.truncated:
        raise ValueError(f"provider truncated the response ({generation.finish_reason})")
    try:
        parsed = json.loads(generation.content)
    except (TypeError, json.JSONDecodeError) as error:
        raise ValueError("provider returned invalid JSON") from error
    content = parsed.get("content") if isinstance(parsed, dict) else None
    if not isinstance(content, str) or not content.strip():
        raise ValueError("provider returned empty section content")
    content = content.strip()
    if _looks_like_source_dump(content):
        raise ValueError("provider returned copied source evidence instead of prose")
    content = _clean_model_section(content, "")
    if len(content.split()) < minimum_words:
        raise ValueError(
            f"provider returned {len(content.split())} words; at least {minimum_words} were required"
        )
    return content


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


async def generate_report(
    db: AsyncSession,
    project_id: UUID,
    document_type: str,
    progress_callback: Callable[[str], Awaitable[None]] | None = None,
) -> EditableDocument:
    analysis = await _analysis_for_project(db, project_id)

    template = predefined_template(document_type)
    # Use the complete grounded report structure as the fallback. The previous
    # fallback reduced most headings to one or two sentences, which made a
    # report appear complete in the editor but export as only a few pages when
    # the model response was short or unavailable.
    fallback = {heading.lower(): _text(content) for heading, content in _normal_report_sections(analysis)}
    settings = get_settings()
    headings = [item["heading"] for item in template["sections"]]
    generated_sections: list[tuple[str, str]] = []
    provider = get_llm_provider()
    for heading in headings:
        if heading.lower() in _USER_INPUT_HEADING_KEYS:
            generated_sections.append((heading, fallback.get(heading.lower(), USER_INPUT_SECTIONS[heading])))
            continue

        if progress_callback:
            await progress_callback(heading)
        target_words, minimum_words = _section_word_target(document_type, heading)
        output_tokens = _section_output_tokens(document_type, heading, target_words)
        try:
            evidence = await _retrieve_section_evidence(project_id, heading, settings)
            prompt = _section_prompt(
                document_type,
                heading,
                target_words,
                minimum_words,
                _section_analysis_context(analysis, heading),
                evidence,
                _previous_report_context(generated_sections),
            )
            last_error = None
            content = None
            for attempt in range(2):
                try:
                    generation = await provider.generate_with_metadata(prompt, max_tokens=output_tokens)
                    content = _parse_section_generation(generation, heading, minimum_words)
                    break
                except Exception as error:
                    last_error = error
                    if attempt == 0:
                        prompt += (
                            f"\n\nRETRY FOR {heading}: The previous response was incomplete or invalid. "
                            f"Return at least {minimum_words} words of prose in the required JSON object."
                        )
                        await asyncio.sleep(0.2)
            if content is None:
                raise ValueError(str(last_error or "empty section response"))
            generated_sections.append((heading, content))
            logger.info(
                "Report section generated project_id=%s document_type=%s heading=%s words=%s "
                "finish_reason=%s prompt_tokens=%s completion_tokens=%s total_tokens=%s",
                project_id, document_type, heading, len(content.split()),
                generation.finish_reason, generation.prompt_tokens,
                generation.completion_tokens, generation.total_tokens,
            )
        except Exception as error:
            message = str(error)[:300] or "unknown generation error"
            logger.warning(
                "Report section failed project_id=%s document_type=%s heading=%s error=%s",
                project_id, document_type, heading, message,
            )
            partial = _section_failure_document(
                document_type,
                template["layout"],
                headings,
                generated_sections,
                heading,
                message,
                fallback,
            )
            raise ReportGenerationError(heading, message, partial) from error
    return EditableDocument(title="USER INPUT REQUIRED — Enter paper title", document_type=document_type,
                            authors=[{"name": "USER INPUT REQUIRED — Enter author name", "institution": ""}],
                            date="",
                            sections=[ReportSection(heading=heading, content=_text(content)) for heading, content in generated_sections],
                            layout=template["layout"])
