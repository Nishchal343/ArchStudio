import json
import logging
from typing import Any
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from ...config import get_settings
from ...schemas import AnalysisResult
from ..llm.base import LLMProvider
from ..llm.service import get_llm_provider
from ..rag.retrieval import search_project

logger = logging.getLogger(__name__)

ANALYSIS_RETRIEVAL_TASKS = (
    "project overview structure architecture entry points and module relationships",
    "technology stack programming languages frameworks libraries dependencies and their purposes",
    "code quality maintainability code smells duplication complexity and improvement opportunities",
    "security authentication authorization secrets input validation unsafe operations and vulnerabilities",
    "performance bottlenecks resource usage file I/O concurrency algorithmic concerns and scalability",
)


class AnalysisServiceError(RuntimeError):
    pass


async def analyze_project(db: AsyncSession, project_id: UUID, llm: LLMProvider | None = None) -> AnalysisResult:
    settings = get_settings()
    llm = llm or get_llm_provider()
    context: list[dict[str, Any]] = []
    seen: set[str] = set()
    for task in ANALYSIS_RETRIEVAL_TASKS:
        results = await search_project(db, project_id, task, top_k=6,
                                       min_similarity=settings.analysis_min_similarity)
        task_added = 0
        for result in results:
            if result["chunk_id"] in seen:
                continue
            seen.add(result["chunk_id"])
            context.append(result)
            task_added += 1
            if task_added >= 1 or len(context) >= settings.analysis_max_context_chunks:
                break
        if len(context) >= settings.analysis_max_context_chunks:
            break

    evidence_parts: list[str] = []
    total_chars = 0
    for item in context:
        evidence = (
            f"FILE: {item['file_path']}\n"
            f"LINES: {item.get('start_line')}–{item.get('end_line')}\n"
            f"LANGUAGE: {item.get('language') or 'unknown'}\n"
            f"CONTENT:\n{item.get('content', '')}\n"
        )
        if total_chars + len(evidence) > settings.analysis_max_context_chars:
            break
        evidence_parts.append(evidence)
        total_chars += len(evidence)

    prompt = build_analysis_prompt(project_id, "\n---\n".join(evidence_parts))
    try:
        generated = await llm.generate(prompt)
        return parse_analysis_result(generated, project_id)
    except AnalysisServiceError:
        raise
    except Exception as error:
        raise AnalysisServiceError("Project analysis could not be generated") from error


def build_analysis_prompt(project_id: UUID, context: str) -> str:
    return f"""Analyze project {project_id} using only the retrieved evidence below from its actual repository.
Do not invent files, functions, dependencies, technologies, or behavior. Say when evidence is insufficient.
Separate observed facts from recommendations. Cite the supplied FILE and LINES in findings. Return one valid JSON object only, never Markdown.
Always include these keys: overview, project_structure, architecture, technology_stack, components, component_relationships, dependencies, code_quality, security_findings, performance_findings, maintainability_findings, code_smells, recommendations, summary.
Use strings for overview/project_structure/architecture/summary, an array of technology names for technology_stack, and arrays of concise objects for the other list fields. Use [] only when the evidence does not support a finding.

RETRIEVED REPOSITORY EVIDENCE:
{context or '[No relevant repository evidence was retrieved.]'}
"""


def parse_analysis_result(text: str, project_id: UUID) -> AnalysisResult:
    candidate = text.strip()
    if candidate.startswith("```"):
        candidate = candidate.strip("`").removeprefix("json").strip()
    try:
        payload = json.loads(candidate)
    except json.JSONDecodeError:
        start, end = candidate.find("{"), candidate.rfind("}")
        if start < 0 or end <= start:
            raise AnalysisServiceError("LLM returned invalid structured analysis")
        try:
            payload = json.loads(candidate[start:end + 1])
        except json.JSONDecodeError as error:
            raise AnalysisServiceError("LLM returned invalid structured analysis") from error
    if not isinstance(payload, dict):
        raise AnalysisServiceError("LLM returned invalid structured analysis")
    core_fields = ("overview", "project_structure", "architecture", "summary", "technology_stack",
                   "components", "dependencies", "security_findings", "recommendations")
    if not any(payload.get(field) for field in core_fields):
        raise AnalysisServiceError("LLM returned empty structured analysis")
    payload["project_id"] = project_id
    # The prompt describes these fields as prose, but models sometimes return
    # a useful list/object representation (for example, a list of repository
    # paths for project_structure). Coerce those values before schema
    # validation so a shape variation does not discard the whole analysis.
    for field in ("overview", "project_structure", "architecture", "summary"):
        value = payload.get(field, "")
        if isinstance(value, list):
            payload[field] = "\n".join(f"- {item}" for item in value)
        elif isinstance(value, dict):
            payload[field] = "\n".join(f"{key}: {value}" for key, value in value.items())
        elif value is None:
            payload[field] = ""
        elif not isinstance(value, str):
            payload[field] = str(value)
    technology_stack = payload.get("technology_stack", [])
    if isinstance(technology_stack, dict):
        payload["technology_stack"] = [f"{name}: {description}" for name, description in technology_stack.items()]
    elif isinstance(technology_stack, list):
        payload["technology_stack"] = [item if isinstance(item, str) else str(item) for item in technology_stack]
    else:
        payload["technology_stack"] = []
    list_fields = ("components", "component_relationships", "dependencies",
                   "code_quality", "security_findings", "performance_findings",
                   "maintainability_findings", "code_smells", "recommendations")
    for field in list_fields:
        value = payload.get(field, [])
        if not isinstance(value, list):
            payload[field] = []
        else:
            payload[field] = [item if isinstance(item, dict) else {"description": str(item)} for item in value]
    try:
        return AnalysisResult.model_validate(payload)
    except Exception as error:
        raise AnalysisServiceError("LLM returned invalid structured analysis") from error
