from uuid import UUID

from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession

from ...models import ProjectAnalysis
from ...schemas import AnalysisResult, EditableDocument, ReportSection
from ..analysis.service import analyze_project


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
    return "\n".join(
        f"- {item.get('title') or item.get('name') or item.get('description', 'Finding')}: "
        f"{item.get('description') or item.get('purpose') or item.get('relationship', '')}"
        for item in items
    )


async def generate_report(db: AsyncSession, project_id: UUID, document_type: str) -> EditableDocument:
    analysis = await _analysis_for_project(db, project_id)
    if document_type == "report":
        sections = [
            ("Project Overview", analysis.overview),
            ("Abstract", analysis.summary),
            ("Objectives", "Objectives explicitly supported by the retrieved repository evidence are summarized below.\n" + analysis.overview),
            ("Technology Stack", _text(analysis.technology_stack)),
            ("Project Structure", analysis.project_structure),
            ("System Architecture", analysis.architecture),
            ("Modules and Components", _items(analysis.components)),
            ("Component Relationships", _items(analysis.component_relationships)),
            ("Implementation Details", _items(analysis.dependencies)),
            ("Database Information", "No database information was identified unless explicitly described in the analysis evidence."),
            ("Security Observations", _items(analysis.security_findings)),
            ("Performance Observations", _items(analysis.performance_findings)),
            ("Limitations", "Only retrieved repository evidence was used; unavailable evidence is not inferred."),
            ("Future Improvements", _items(analysis.recommendations)),
            ("Conclusion", analysis.summary),
        ]
        title = "Project Report"
    elif document_type == "ieee_report":
        sections = [
            ("Abstract", analysis.summary), ("Keywords", _text(analysis.technology_stack)),
            ("I. Introduction", analysis.overview), ("II. Problem Statement", "Not explicitly established by repository evidence."),
            ("III. Objectives", analysis.overview), ("IV. Methodology", analysis.architecture),
            ("V. System Architecture", analysis.project_structure + "\n" + analysis.architecture),
            ("VI. Implementation", _items(analysis.components)), ("VII. Results", "No experimental results were invented; see the retrieved evidence."),
            ("VIII. Discussion", _items(analysis.code_quality)), ("IX. Limitations", "Evidence is limited to retrieved project sources."),
            ("X. Future Scope", _items(analysis.recommendations)), ("XI. Conclusion", analysis.summary),
            ("References", "No external references were identified in the repository evidence."),
        ]
        title = "IEEE-Style Project Report"
    else:
        sections = [
            ("Abstract", analysis.summary), ("Keywords", _text(analysis.technology_stack)),
            ("1. Introduction", analysis.overview), ("2. Motivation", "Motivation is reported only where supported by repository evidence."),
            ("3. Problem", "A project problem statement was not explicitly available in the retrieved evidence."),
            ("4. Existing or Related Approach", "No related-work claims were inferred from the repository."),
            ("5. Proposed Methodology", analysis.architecture), ("6. System Architecture", analysis.project_structure),
            ("7. Implementation", _items(analysis.components)), ("8. Results", "No unsupported research results are reported."),
            ("9. Discussion", _items(analysis.code_quality)), ("10. Limitations", "Only repository evidence was considered."),
            ("11. Future Work", _items(analysis.recommendations)), ("12. Conclusion", analysis.summary),
            ("References", "No citations or references were fabricated."),
        ]
        title = "Research Paper"
    return EditableDocument(title=title, document_type=document_type,
                            sections=[ReportSection(heading=heading, content=_text(content)) for heading, content in sections])
