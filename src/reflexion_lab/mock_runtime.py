from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Generic, TypeVar
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .prompts import ACTOR_SYSTEM, EVALUATOR_SYSTEM, REFLECTOR_SYSTEM
from .schemas import JudgeResult, QAExample, ReflectionEntry
from .utils import normalize_answer

try:
    from dotenv import load_dotenv

    load_dotenv(override=True)
except Exception:
    pass


T = TypeVar("T")
FAILURE_MODE_BY_QID: dict[str, str] = {}


@dataclass
class RuntimeStep(Generic[T]):
    value: T
    token_estimate: int
    prompt_tokens: int
    completion_tokens: int
    token_source: str
    latency_ms: int


@dataclass(frozen=True)
class LLMSettings:
    provider: str
    base_url: str
    api_key: str
    model: str
    timeout_s: float
    temperature: float


def _settings() -> LLMSettings:
    provider = os.getenv("LLM_PROVIDER", "openai-compatible").strip().lower()
    base_url = (
        os.getenv("BASE_URL")
        or os.getenv("OPENAI_BASE_URL")
        or os.getenv("OLLAMA_BASE_URL")
        or "https://api.openai.com/v1"
    )
    api_key = os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY") or os.getenv("GEMINI_API_KEY") or ""
    model = os.getenv("MODEL_NAME") or os.getenv("OPENAI_MODEL") or os.getenv("OLLAMA_MODEL") or "gpt-4o-mini"
    timeout_s = float(os.getenv("LLM_TIMEOUT", "60"))
    temperature = float(os.getenv("LLM_TEMPERATURE", "0"))
    return LLMSettings(
        provider=provider,
        base_url=base_url.strip(),
        api_key=api_key.strip(),
        model=model.strip(),
        timeout_s=timeout_s,
        temperature=temperature,
    )


def _rough_token_count(text: str) -> int:
    return max(1, len(re.findall(r"\w+|[^\s\w]", text or "")))


def _usage_from_openai(payload: dict[str, Any], messages: list[dict[str, str]], content: str) -> tuple[int, int, int, str]:
    usage = payload.get("usage") or {}
    prompt_tokens = int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
    completion_tokens = int(usage.get("completion_tokens") or usage.get("output_tokens") or 0)
    total_tokens = int(usage.get("total_tokens") or 0)
    if total_tokens > 0:
        if prompt_tokens == 0 and completion_tokens == 0:
            completion_tokens = _rough_token_count(content)
            prompt_tokens = max(0, total_tokens - completion_tokens)
        return total_tokens, prompt_tokens, completion_tokens, "api_usage"

    prompt_text = "\n".join(message.get("content", "") for message in messages)
    prompt_tokens = _rough_token_count(prompt_text)
    completion_tokens = _rough_token_count(content)
    return prompt_tokens + completion_tokens, prompt_tokens, completion_tokens, "local_estimate"


def _openai_candidate_urls(base_url: str) -> list[str]:
    clean = base_url.rstrip("/")
    if clean.endswith("/chat/completions"):
        return [clean]
    if clean.endswith("/v1"):
        return [f"{clean}/chat/completions"]
    return [f"{clean}/v1/chat/completions", f"{clean}/chat/completions"]


def _post_json(url: str, body: dict[str, Any], api_key: str, timeout_s: float) -> dict[str, Any]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = Request(url, data=json.dumps(body).encode("utf-8"), headers=headers, method="POST")
    with urlopen(request, timeout=timeout_s) as response:
        raw = response.read().decode("utf-8")
    return json.loads(raw)


def _call_openai_compatible(messages: list[dict[str, str]], max_tokens: int) -> RuntimeStep[str]:
    settings = _settings()
    body = {
        "model": settings.model,
        "messages": messages,
        "temperature": settings.temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }
    errors: list[str] = []
    started = time.perf_counter()
    for url in _openai_candidate_urls(settings.base_url):
        try:
            payload = _post_json(url, body, settings.api_key, settings.timeout_s)
            latency_ms = int((time.perf_counter() - started) * 1000)
            choice = (payload.get("choices") or [{}])[0]
            message = choice.get("message") or {}
            content = str(message.get("content") or choice.get("text") or "").strip()
            total, prompt, completion, source = _usage_from_openai(payload, messages, content)
            return RuntimeStep(content, total, prompt, completion, source, latency_ms)
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:300]
            errors.append(f"{url}: HTTP {exc.code} {detail}")
        except (URLError, TimeoutError, json.JSONDecodeError) as exc:
            errors.append(f"{url}: {exc}")

    raise RuntimeError(
        "LLM API call failed. Check BASE_URL, API_KEY, MODEL_NAME, and network access. "
        + " | ".join(errors)
    )


def _call_ollama(messages: list[dict[str, str]], max_tokens: int) -> RuntimeStep[str]:
    settings = _settings()
    url = settings.base_url.rstrip("/") + "/api/chat"
    body = {
        "model": settings.model,
        "messages": messages,
        "stream": False,
        "options": {"temperature": settings.temperature, "num_predict": max_tokens},
    }
    started = time.perf_counter()
    payload = _post_json(url, body, "", settings.timeout_s)
    latency_ms = int((time.perf_counter() - started) * 1000)
    content = str((payload.get("message") or {}).get("content") or "").strip()
    prompt_tokens = int(payload.get("prompt_eval_count") or 0)
    completion_tokens = int(payload.get("eval_count") or 0)
    if prompt_tokens or completion_tokens:
        return RuntimeStep(content, prompt_tokens + completion_tokens, prompt_tokens, completion_tokens, "api_usage", latency_ms)

    prompt_text = "\n".join(message.get("content", "") for message in messages)
    prompt_tokens = _rough_token_count(prompt_text)
    completion_tokens = _rough_token_count(content)
    return RuntimeStep(content, prompt_tokens + completion_tokens, prompt_tokens, completion_tokens, "local_estimate", latency_ms)


def chat_completion(messages: list[dict[str, str]], max_tokens: int = 256) -> RuntimeStep[str]:
    settings = _settings()
    if settings.provider == "ollama" or "11434" in settings.base_url:
        return _call_ollama(messages, max_tokens=max_tokens)
    return _call_openai_compatible(messages, max_tokens=max_tokens)


def _context_text(example: QAExample) -> str:
    return "\n\n".join(f"[{chunk.title}]\n{chunk.text}" for chunk in example.context)


def _reflection_text(reflection_memory: list[str]) -> str:
    if not reflection_memory:
        return "No previous reflections."
    return "\n".join(f"- {item}" for item in reflection_memory[-3:])


def _json_object(text: str) -> Any:
    stripped = text.strip()
    stripped = re.sub(r"^```(?:json)?", "", stripped, flags=re.IGNORECASE).strip()
    stripped = re.sub(r"```$", "", stripped).strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def _clean_answer(text: str) -> str:
    answer = text.strip()
    answer = re.sub(r"^```(?:text)?", "", answer, flags=re.IGNORECASE).strip()
    answer = re.sub(r"```$", "", answer).strip()
    for marker in ("final answer:", "answer:"):
        index = answer.lower().find(marker)
        if index >= 0:
            answer = answer[index + len(marker) :].strip()
    for line in answer.splitlines():
        line = line.strip(" \t-")
        if line:
            answer = line
            break
    return answer.strip().strip('"').strip("'").strip()


def actor_answer(
    example: QAExample,
    attempt_id: int,
    agent_type: str,
    reflection_memory: list[str],
) -> RuntimeStep[str]:
    user_prompt = f"""Question:
{example.question}

Context:
{_context_text(example)}

Reflection memory:
{_reflection_text(reflection_memory)}

Attempt: {attempt_id}
Agent type: {agent_type}

Return only the final short answer."""
    step = chat_completion(
        [
            {"role": "system", "content": ACTOR_SYSTEM},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=80,
    )
    step.value = _clean_answer(step.value)
    return step


def evaluator(example: QAExample, answer: str) -> RuntimeStep[JudgeResult]:
    user_prompt = f"""Question: {example.question}
Gold answer: {example.gold_answer}
Predicted answer: {answer}

Return the JSON object only."""
    step = chat_completion(
        [
            {"role": "system", "content": EVALUATOR_SYSTEM},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=180,
    )
    exact_score = int(normalize_answer(example.gold_answer) == normalize_answer(answer))
    try:
        raw = _json_object(step.value)
    except Exception:
        raw = {}
    if not isinstance(raw, dict):
        raw = {}

    reason = str(raw.get("reason") or "")
    if not reason:
        reason = (
            "Predicted answer matches the normalized gold answer."
            if exact_score
            else "Predicted answer does not match the normalized gold answer."
        )

    missing = raw.get("missing_evidence") if isinstance(raw.get("missing_evidence"), list) else []
    spurious = raw.get("spurious_claims") if isinstance(raw.get("spurious_claims"), list) else []
    if not exact_score and answer and not spurious:
        spurious = [answer]

    step.value = JudgeResult(
        score=exact_score,
        reason=reason,
        missing_evidence=[str(item) for item in missing],
        spurious_claims=[str(item) for item in spurious],
    )
    return step


def reflector(example: QAExample, attempt_id: int, judge: JudgeResult, answer: str) -> RuntimeStep[ReflectionEntry]:
    user_prompt = f"""Question:
{example.question}

Context:
{_context_text(example)}

Failed attempt: {attempt_id}
Predicted answer: {answer}
Gold answer: {example.gold_answer}
Evaluator reason: {judge.reason}
Missing evidence: {judge.missing_evidence}
Spurious claims: {judge.spurious_claims}

Return the JSON object only."""
    step = chat_completion(
        [
            {"role": "system", "content": REFLECTOR_SYSTEM},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=220,
    )
    try:
        raw = _json_object(step.value)
    except Exception:
        raw = {}
    if not isinstance(raw, dict):
        raw = {}

    lesson = str(raw.get("lesson") or judge.reason)
    next_strategy = str(
        raw.get("next_strategy")
        or "Re-read the supporting context, complete both reasoning hops, and answer with only the final entity."
    )
    step.value = ReflectionEntry(
        attempt_id=attempt_id,
        failure_reason=judge.reason,
        lesson=lesson,
        next_strategy=next_strategy,
    )
    return step


def classify_failure(example: QAExample, answer: str, judge: JudgeResult, attempts: int) -> str:
    if judge.score == 1:
        return "none"
    if attempts > 1 and judge.spurious_claims and normalize_answer(answer) in {
        normalize_answer(item) for item in judge.spurious_claims
    }:
        return "reflection_overfit"
    normalized_answer = normalize_answer(answer)
    context_titles = {normalize_answer(chunk.title) for chunk in example.context}
    if normalized_answer in context_titles:
        return "incomplete_multi_hop"
    if judge.spurious_claims:
        return "entity_drift"
    return "wrong_final_answer"
