import asyncio
import json
from types import SimpleNamespace
from uuid import UUID

import pytest
from pydantic import ValidationError

from app.config import get_settings
from app.schemas import AnalysisResult
from app.services.analysis import service as analysis_service
from app.services.analysis.service import AnalysisServiceError, build_analysis_prompt, parse_analysis_result
from app.services.llm.base import LLMProviderError
from app.services.llm.groq import GroqLLMProvider
from app.services.llm.service import get_llm_provider


def run(coro):
    return asyncio.run(coro)


class FakeResponse:
    def __init__(self, status_code, body):
        self.status_code = status_code
        self.body = body

    def json(self):
        return self.body


class FakeClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    async def post(self, url, **kwargs):
        self.calls.append((url, kwargs))
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


PROJECT_ID = UUID("27ce6a06-6e88-43e5-8dba-6a6cc165cda0")


def test_groq_provider_uses_structured_chat_completions():
    client = FakeClient([FakeResponse(200, {"choices": [{"message": {"content": '{"overview":"grounded"}'}}]})])
    provider = GroqLLMProvider(api_key="test-key", client=client, max_retries=0)

    assert run(provider.generate("prompt")) == '{"overview":"grounded"}'
    assert client.calls[0][0] == "https://api.groq.com/openai/v1/chat/completions"
    headers = client.calls[0][1]["headers"]
    payload = client.calls[0][1]["json"]
    assert headers["Authorization"] == "Bearer test-key"
    assert payload["model"] == "qwen/qwen3.6-27b"
    assert payload["response_format"] == {"type": "json_object"}

    long_result = GroqLLMProvider(api_key="test-key", client=FakeClient([
        FakeResponse(200, {"choices": [{"message": {"content": "{}"}}]})
    ]), max_retries=0)
    run(long_result.generate("prompt", max_tokens=5000))
    assert long_result.client.calls[0][1]["json"]["max_tokens"] == 5000


def test_groq_provider_reports_safe_api_errors():
    invalid_key = GroqLLMProvider(api_key="test-key", client=FakeClient([FakeResponse(401, {})]), max_retries=0)
    with pytest.raises(LLMProviderError, match="key was rejected"):
        run(invalid_key.generate("prompt"))

    malformed = GroqLLMProvider(api_key="test-key", client=FakeClient([FakeResponse(200, {})]), max_retries=0)
    with pytest.raises(LLMProviderError, match="malformed"):
        run(malformed.generate("prompt"))


def test_provider_factory_uses_groq_by_default(monkeypatch):
    monkeypatch.setattr(get_settings(), "groq_api_key", "test-key")
    assert isinstance(get_llm_provider(), GroqLLMProvider)


def test_prompt_requires_grounded_evidence_and_no_hallucination():
    prompt = build_analysis_prompt(PROJECT_ID, "FILE: animation.c\nLINES: 1–5\nCONTENT: OpenGL")
    assert "actual repository" in prompt
    assert "Do not invent files" in prompt
    assert "FILE: animation.c" in prompt
    assert "Return one valid JSON object only" in prompt


def test_analysis_result_parser_handles_json_fences_and_rejects_invalid_json():
    result = parse_analysis_result("```json\n{\"overview\":\"facts\",\"technology_stack\":[\"C\"]}\n```", PROJECT_ID)
    assert isinstance(result, AnalysisResult)
    assert result.project_id == PROJECT_ID
    assert result.technology_stack == ["C"]
    with pytest.raises(AnalysisServiceError):
        parse_analysis_result("not json", PROJECT_ID)
    with pytest.raises(AnalysisServiceError):
        parse_analysis_result("{}", PROJECT_ID)


def test_analysis_result_parser_coerces_list_project_structure():
    result = parse_analysis_result(json.dumps({
        "overview": "Grounded overview",
        "project_structure": ["frontend/index.html", "backend/app/main.py"],
        "technology_stack": ["Python"],
    }), PROJECT_ID)

    assert result.project_structure == "- frontend/index.html\n- backend/app/main.py"


def test_analysis_retrieves_focused_context_before_llm(monkeypatch):
    calls = []

    async def fake_search(db, project_id, query, top_k, min_similarity):
        calls.append((query, top_k, min_similarity))
        return [{"chunk_id": "animation.c:0", "file_path": "animation.c", "language": "c",
                 "start_line": 1, "end_line": 5, "similarity": 0.9, "content": "OpenGL animation"}]

    class FakeLLM:
        def __init__(self):
            self.prompt = None

        async def generate(self, prompt):
            self.prompt = prompt
            return json.dumps({"overview": "OpenGL animation", "technology_stack": ["C", "OpenGL"],
                               "summary": "Evidence-based summary"})

    llm = FakeLLM()
    monkeypatch.setattr(analysis_service, "search_project", fake_search)
    result = run(analysis_service.analyze_project(None, PROJECT_ID, llm))
    assert len(calls) == 5
    assert all(call[1] == 6 for call in calls)
    assert all(call[2] == 0.0 for call in calls)
    assert result.summary == "Evidence-based summary"
    assert "animation.c" in llm.prompt
    assert "OpenGL animation" in llm.prompt
