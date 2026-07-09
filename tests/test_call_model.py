"""call_model retry behavior + threaded runner end-to-end, all against a fake transport."""

import json

import pytest
import requests

import run_openrouter
import throttle


class FakeResponse:
    def __init__(self, status_code=200, body=None, headers=None, text=None):
        self.status_code = status_code
        self._body = body
        self.headers = headers or {}
        self.text = text if text is not None else (json.dumps(body) if body is not None else "")

    def json(self):
        if self._body is None:
            raise ValueError("no body")
        return self._body


class ScriptedSession:
    """Returns (or raises) the scripted outcomes in order; records every request."""

    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.requests_made = []

    def post(self, url, headers=None, data=None, timeout=None, stream=False):
        self.requests_made.append(json.loads(data))
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def success_body(content="FINAL ANSWER: e2e4", reasoning=None):
    message = {"role": "assistant", "content": content}
    if reasoning:
        message["reasoning"] = reasoning
        message["reasoning_details"] = [{"type": "reasoning.text", "text": reasoning, "format": "anthropic-claude-v1"}]
    return {
        "id": "cmpl-1",
        "model": "anthropic/claude-haiku-4.5",
        "choices": [{"message": message}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30, "cost": 0.001},
    }


@pytest.fixture()
def inferencer(monkeypatch):
    monkeypatch.setenv("AI_GATEWAY_API_KEY", "test-key")
    return run_openrouter.OpenrouterInferencer("anthropic/claude-haiku-4.5", max_retries=4, backend="vercel-gateway")


@pytest.fixture()
def no_sleep(monkeypatch):
    sleeps = []
    monkeypatch.setattr(run_openrouter.time, "sleep", lambda seconds: sleeps.append(seconds))
    return sleeps


def test_success_first_attempt(inferencer):
    session = ScriptedSession([FakeResponse(200, success_body(reasoning="thinking hard"))])
    call = inferencer.call_model("prompt", session=session)
    assert call.ok and call.content == "FINAL ANSWER: e2e4"
    assert (call.thinking_content, call.thinking_source) == ("thinking hard", "full_text")
    assert call.usage["cost"] == 0.001
    assert call.raw_message["content"] == "FINAL ANSWER: e2e4"
    assert len(call.attempts) == 1 and call.attempts[0]["outcome"] == "success"
    assert call.latency_ms is not None


def test_429_honors_retry_after_and_penalizes_limiter(inferencer, no_sleep, monkeypatch):
    session = ScriptedSession([
        FakeResponse(429, headers={"Retry-After": "7"}, text="rate limited"),
        FakeResponse(200, success_body()),
    ])
    penalties = []
    limiter = throttle.RateLimiter(1000, 1000)
    monkeypatch.setattr(limiter, "penalize", lambda seconds: penalties.append(seconds))
    call = inferencer.call_model("prompt", limiter=limiter, session=session)
    assert call.ok
    assert len(call.attempts) == 2
    first = call.attempts[0]
    assert first["error_class"] == "http_429" and first["outcome"] == "retried"
    assert first["retry_after_s"] == 7.0 and first["wait_s"] == 7.0
    assert penalties == [7.0], "429 must penalize the shared limiter with Retry-After"
    assert no_sleep == [], "429 path must not double-sleep locally when a limiter exists"


def test_fatal_400_single_attempt(inferencer, no_sleep):
    session = ScriptedSession([FakeResponse(400, text='{"error": "bad request"}')])
    call = inferencer.call_model("prompt", session=session)
    assert not call.ok and call.content.startswith("ERROR: HTTP 400")
    assert len(call.attempts) == 1
    assert call.attempts[0]["outcome"] == "fatal"
    assert len(session.requests_made) == 1, "permanent 4xx must never retry"


def test_timeout_capped_at_expensive_max_attempts(inferencer, no_sleep):
    session = ScriptedSession([requests.Timeout("read timed out")] * 10)
    call = inferencer.call_model("prompt", session=session)
    assert not call.ok
    assert len(session.requests_made) == throttle.EXPENSIVE_MAX_ATTEMPTS
    assert all(attempt["billed_risk"] == 1 for attempt in call.attempts)
    assert call.attempts[-1]["outcome"] == "gave_up"


def test_cheap_503_retries_to_success(inferencer, no_sleep):
    session = ScriptedSession([
        FakeResponse(503, text="unavailable"),
        requests.ConnectionError("reset"),
        FakeResponse(200, success_body()),
    ])
    call = inferencer.call_model("prompt", session=session)
    assert call.ok and len(call.attempts) == 3
    assert [attempt["error_class"] for attempt in call.attempts] == ["http_5xx", "connection", "ok"]
    assert len(no_sleep) == 2, "cheap retries sleep locally"


def test_parse_error_on_2xx_never_regenerates(inferencer, no_sleep):
    session = ScriptedSession([FakeResponse(200, {"unexpected": "shape"})])
    call = inferencer.call_model("prompt", session=session)
    assert not call.ok and len(session.requests_made) == 1
    assert call.attempts[0]["error_class"] == "parse"
    assert call.attempts[0]["billed_risk"] == 1


def test_gateway_payload_shape(inferencer):
    session = ScriptedSession([FakeResponse(200, success_body())])
    inferencer.enable_thinking = True
    inferencer.call_model("prompt", session=session)
    payload = session.requests_made[0]
    assert payload["reasoning"] == {"effort": "medium"}
    assert "usage" not in payload and "provider" not in payload


# --- threaded runner end-to-end -------------------------------------------------------------


def make_task(index):
    return {
        "task_id": f"synthetic_task_{index:04d}",
        "task_type": "synthetic",
        "task_category": "Synthetic",
        "input": "8/8/8/8/8/8/8/8 w - - 0 1",
        "question": "CONTEXT_PLACEHOLDERPick e2e4.\nFINAL ANSWER: <answer>\nFORMAT_EXAMPLE_PLACEHOLDER",
        "format_examples": ["e2e4", "d2d4"],
        "correct_answer": "e2e4",
        "answer_type": "single",
        "metadata": None,
    }


def test_runner_end_to_end_threads(inferencer, monkeypatch, tmp_path):
    tasks = [make_task(index) for index in range(30)]
    session = ScriptedSession([FakeResponse(200, success_body())] * 30)
    lock_free_session = session  # ScriptedSession.pop(0) is GIL-atomic enough for identical outcomes
    monkeypatch.setattr(run_openrouter, "_get_thread_session", lambda: lock_free_session)

    limiter = throttle.RateLimiter(requests_per_second=10_000, burst=10_000)
    results = inferencer.run_inference(
        tasks, num_workers=8, output_path=str(tmp_path), save_interval=10, limiter=limiter
    )

    assert len(results) == 30
    assert [result["task_id"] for result in results] == [task["task_id"] for task in tasks], "task order restored"
    assert all(result["inference"]["is_correct"] for result in results)
    assert all(result["inference"]["attempts"][0]["outcome"] == "success" for result in results)

    # Incremental JSONL exists, has all rows, every line valid JSON
    jsonl_path = tmp_path / "anthropic_claude-haiku-4.5.jsonl"
    lines = [json.loads(line) for line in jsonl_path.read_text().splitlines() if line.strip()]
    assert len(lines) == 30

    # Resume: every task complete -> zero incomplete
    existing = run_openrouter.load_existing_results(jsonl_path)
    incomplete, complete = run_openrouter.filter_incomplete_tasks(tasks, existing)
    assert incomplete == [] and len(complete) == 30
