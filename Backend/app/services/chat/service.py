import json
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from ...config import get_settings
from ...schemas import ChatEvidence, ChatResponse
from ..llm.base import LLMProvider, LLMProviderError
from ..llm.service import get_llm_provider
from ..rag.retrieval import search_project


def _parse_answer(raw: str) -> tuple[str, list[dict]]:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        return raw.strip(), []
    if not isinstance(value, dict) or not isinstance(value.get("answer"), str) or not value["answer"].strip():
        raise LLMProviderError("Local LLM returned an invalid chat response")
    evidence = value.get("evidence", [])
    return value["answer"].strip(), evidence if isinstance(evidence, list) else []


async def answer_project_question(db: AsyncSession, project_id: UUID, query: str, top_k: int,
                                 provider: LLMProvider | None = None) -> ChatResponse:
    settings = get_settings()
    results = await search_project(db, project_id, query, top_k, settings.analysis_min_similarity)
    if not results:
        return ChatResponse(project_id=project_id, query=query,
                            answer="I could not find supporting evidence in this project.")
    context_parts: list[str] = []
    total = 0
    for result in results:
        block = f"FILE: {result['file_path']}:{result.get('start_line')}-{result.get('end_line')}\n{result['content']}"
        if total + len(block) > settings.analysis_max_context_chars:
            break
        context_parts.append(block)
        total += len(block)
    prompt = (
        "Answer the user's question using only the repository evidence below. Do not invent code, "
        "files, behavior, citations, or facts. If evidence is insufficient, say so. Return JSON only "
        "with keys answer and evidence; evidence is an array of file_path, start_line, end_line, similarity.\n\n"
        f"QUESTION:\n{query}\n\nEVIDENCE:\n{chr(10).join(context_parts)}"
    )
    answer, cited = _parse_answer(await (provider or get_llm_provider()).generate(prompt))
    evidence = []
    for item in cited:
        if not isinstance(item, dict):
            continue
        matching = next((r for r in results if r["file_path"] == item.get("file_path")), None)
        if matching:
            evidence.append(ChatEvidence(file_path=matching["file_path"], start_line=matching.get("start_line"),
                                         end_line=matching.get("end_line"), similarity=matching["similarity"]))
    if not evidence:
        evidence = [ChatEvidence(file_path=r["file_path"], start_line=r.get("start_line"),
                                 end_line=r.get("end_line"), similarity=r["similarity"]) for r in results]
    return ChatResponse(project_id=project_id, query=query, answer=answer, evidence=evidence)
