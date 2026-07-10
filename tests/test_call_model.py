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
        self.calls = []

    def post(self, url, headers=None, data=None, timeout=None, stream=False):
        payload = json.loads(data)
        self.requests_made.append(payload)
        self.calls.append({"url": url, "headers": headers or {}, "payload": payload})
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
    call = session.calls[0]
    assert call["url"] == "https://ai-gateway.vercel.sh/v1/chat/completions"
    assert call["payload"]["reasoning"] == {"effort": "medium"}
    assert "usage" not in call["payload"] and "provider" not in call["payload"]
    assert "thinking" not in call["payload"], "classic-thinking models stay on chat completions"


# --- Claude 5 adaptive thinking: native-endpoint routing -------------------------------------


def test_adaptive_thinking_version_detection():
    assert run_openrouter.anthropic_adaptive_thinking("anthropic/claude-sonnet-5") is True
    assert run_openrouter.anthropic_adaptive_thinking("anthropic/claude-opus-5.1") is True
    assert run_openrouter.anthropic_adaptive_thinking("anthropic/claude-haiku-4.5") is False
    assert run_openrouter.anthropic_adaptive_thinking("anthropic/claude-opus-4.8") is False
    assert run_openrouter.anthropic_adaptive_thinking("anthropic/claude-3.5-haiku") is False
    assert run_openrouter.anthropic_adaptive_thinking("openai/gpt-5.5") is False


def native_body(thinking_text="Summarized reasoning about the position.", include_thinking=True):
    content = []
    if include_thinking:
        content.append({"type": "thinking", "thinking": thinking_text, "signature": "sig-abc"})
    content.append({"type": "text", "text": "FINAL ANSWER: e2e4"})
    return {
        "id": "msg-1",
        "model": "claude-sonnet-5",
        "role": "assistant",
        "stop_reason": "end_turn",
        "content": content,
        "usage": {"input_tokens": 100, "output_tokens": 900},
    }


def test_claude5_thinking_routes_to_native_endpoint(monkeypatch):
    monkeypatch.setenv("AI_GATEWAY_API_KEY", "test-key")
    inferencer = run_openrouter.OpenrouterInferencer(
        "anthropic/claude-sonnet-5", enable_thinking=True, backend="vercel-gateway"
    )
    session = ScriptedSession([FakeResponse(200, native_body())])
    call = inferencer.call_model("prompt", session=session)

    request = session.calls[0]
    assert request["url"] == "https://ai-gateway.vercel.sh/v1/messages"
    assert request["headers"]["anthropic-version"] == "2023-06-01"
    assert request["payload"]["thinking"] == {"type": "adaptive", "display": "summarized"}
    assert "reasoning" not in request["payload"]

    assert call.ok and call.content == "FINAL ANSWER: e2e4"
    assert call.thinking_content == "Summarized reasoning about the position."
    assert call.thinking_source == "summary"
    assert call.usage["prompt_tokens"] == 100 and call.usage["completion_tokens"] == 900
    assert call.usage["total_tokens"] == 1000
    # Raw native message (thinking blocks + signature) preserved for storage
    assert call.raw_message["content"][0]["signature"] == "sig-abc"


def test_claude5_without_thinking_stays_on_chat_completions(monkeypatch):
    monkeypatch.setenv("AI_GATEWAY_API_KEY", "test-key")
    inferencer = run_openrouter.OpenrouterInferencer(
        "anthropic/claude-sonnet-5", enable_thinking=False, backend="vercel-gateway"
    )
    session = ScriptedSession([FakeResponse(200, success_body())])
    call = inferencer.call_model("prompt", session=session)
    assert session.calls[0]["url"] == "https://ai-gateway.vercel.sh/v1/chat/completions"
    assert call.ok


def test_native_parse_thinking_withheld_and_absent():
    # Thinking block present but text withheld (empty) -> encrypted_only
    withheld = native_body(thinking_text="", include_thinking=True)
    content, thinking, source, usage = run_openrouter.parse_native_anthropic_message(withheld)
    assert (content, thinking, source) == ("FINAL ANSWER: e2e4", "", "encrypted_only")
    # No thinking block at all (adaptive skipped thinking) -> none
    skipped = native_body(include_thinking=False)
    content, thinking, source, usage = run_openrouter.parse_native_anthropic_message(skipped)
    assert (thinking, source) == ("", "none")
    assert usage["completion_tokens"] == 900


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
    import storage

    tasks = [make_task(index) for index in range(30)]
    session = ScriptedSession([FakeResponse(200, success_body())] * 30)
    lock_free_session = session  # ScriptedSession.pop(0) is GIL-atomic enough for identical outcomes
    monkeypatch.setattr(run_openrouter, "_get_thread_session", lambda: lock_free_session)

    # Live DB mirroring via the record hook (main() wires this identically)
    conn = storage.connect(tmp_path / "test.sqlite3")
    run_id = storage.get_or_create_run(conn, "e2e-run", {"model": "anthropic/claude-haiku-4.5"})

    limiter = throttle.RateLimiter(requests_per_second=10_000, burst=10_000)
    results = inferencer.run_inference(
        tasks,
        num_workers=8,
        output_path=str(tmp_path),
        save_interval=10,
        limiter=limiter,
        record_hook=lambda result: storage.upsert_result(conn, run_id, result),
    )
    conn.commit()
    db_rows = conn.execute("SELECT COUNT(*) AS n, SUM(is_correct) AS correct FROM results").fetchone()
    assert db_rows["n"] == 30 and db_rows["correct"] == 30
    conn.close()

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
