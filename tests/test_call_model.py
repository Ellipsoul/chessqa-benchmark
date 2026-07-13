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

    def close(self):
        pass


class FakeStreamResponse:
    """A 2xx SSE response: iter_lines yields the scripted lines (or raises a scripted
    exception mid-stream, simulating a transport drop)."""

    def __init__(self, lines, status_code=200):
        self.status_code = status_code
        self._lines = list(lines)
        self.headers = {}
        self.text = ""
        self.closed = False

    def iter_lines(self):
        for item in self._lines:
            if isinstance(item, Exception):
                raise item
            yield item

    def close(self):
        self.closed = True


class ScriptedSession:
    """Returns (or raises) the scripted outcomes in order; records every request."""

    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.requests_made = []
        self.calls = []

    def post(self, url, headers=None, data=None, timeout=None, stream=False):
        payload = json.loads(data)
        self.requests_made.append(payload)
        self.calls.append({"url": url, "headers": headers or {}, "payload": payload, "stream": stream})
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


def sse_lines_from_body(body, piece_len=4):
    """Chop a stored non-streamed chat body into the SSE lines a streamed request delivers.

    The parity fixture: reasoning streams as delta.reasoning plus indexed typed
    reasoning_details fragments, content as delta.content pieces, usage in a final chunk
    with empty choices (stream_options.include_usage behavior), plus a comment keepalive
    and [DONE] terminator for grammar coverage.
    """
    message = body["choices"][0]["message"]
    base = {key: body[key] for key in ("id", "model", "provider", "created") if key in body}

    def event(payload):
        return ("data: " + json.dumps(payload)).encode()

    lines = [b": KEEPALIVE", event({**base, "choices": [{"delta": {"role": "assistant", "content": ""}}]})]
    reasoning = message.get("reasoning") or ""
    for start in range(0, len(reasoning), piece_len):
        piece = reasoning[start : start + piece_len]
        delta = {
            "reasoning": piece,
            "reasoning_details": [
                {"type": "reasoning.text", "text": piece, "format": "anthropic-claude-v1", "index": 0}
            ],
        }
        lines.append(event({**base, "choices": [{"delta": delta}]}))
    content = message.get("content") or ""
    for start in range(0, len(content), piece_len):
        lines.append(event({**base, "choices": [{"delta": {"content": content[start : start + piece_len]}}]}))
    lines.append(event({**base, "choices": [], "usage": body.get("usage")}))
    lines.append(b"data: [DONE]")
    return lines


def streamed_success(content="FINAL ANSWER: e2e4", reasoning=None):
    return FakeStreamResponse(sse_lines_from_body(success_body(content, reasoning)))


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
    session = ScriptedSession([streamed_success(reasoning="thinking hard")])
    call = inferencer.call_model("prompt", session=session)
    assert call.ok and call.content == "FINAL ANSWER: e2e4"
    assert (call.thinking_content, call.thinking_source) == ("thinking hard", "full_text")
    assert call.usage["cost"] == 0.001
    assert call.raw_message["content"] == "FINAL ANSWER: e2e4"
    assert len(call.attempts) == 1 and call.attempts[0]["outcome"] == "success"
    assert call.latency_ms is not None
    assert call.ttft_ms is not None and call.attempts[0]["ttft_ms"] == call.ttft_ms
    assert call.attempts[0]["stream_chunks"] > 0
    # Chat-completions requests are streamed (the 340s-wall fix)
    assert session.calls[0]["stream"] is True
    assert session.calls[0]["payload"]["stream"] is True
    assert session.calls[0]["payload"]["stream_options"] == {"include_usage": True}


def test_429_honors_retry_after_and_penalizes_limiter(inferencer, no_sleep, monkeypatch):
    session = ScriptedSession([
        FakeResponse(429, headers={"Retry-After": "7"}, text="rate limited"),
        streamed_success(),
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
        streamed_success(),
    ])
    call = inferencer.call_model("prompt", session=session)
    assert call.ok and len(call.attempts) == 3
    assert [attempt["error_class"] for attempt in call.attempts] == ["http_5xx", "connection", "ok"]
    assert len(no_sleep) == 2, "cheap retries sleep locally"


def test_parse_error_on_2xx_never_regenerates(inferencer, no_sleep):
    # A 2xx whose body is not an SSE stream (e.g. a plain JSON error blob): received and
    # billed, so never regenerate — same taxonomy as the pre-streaming parse case.
    session = ScriptedSession([FakeStreamResponse([b'data: {"unexpected": "shape"}', b"data: [DONE]"])])
    call = inferencer.call_model("prompt", session=session)
    assert not call.ok and len(session.requests_made) == 1
    assert call.attempts[0]["error_class"] == "parse"
    assert call.attempts[0]["billed_risk"] == 1


def test_gateway_payload_shape(inferencer):
    session = ScriptedSession([streamed_success()])
    inferencer.enable_thinking = True
    inferencer.call_model("prompt", session=session)
    call = session.calls[0]
    assert call["url"] == "https://ai-gateway.vercel.sh/v1/chat/completions"
    assert call["payload"]["reasoning"] == {"effort": "medium"}
    assert "usage" not in call["payload"] and "provider" not in call["payload"]
    assert "thinking" not in call["payload"], "classic-thinking models stay on chat completions"


# --- streaming transport (the 340s-wall fix) --------------------------------------------------


def test_stream_parity_with_nonstreamed_body(inferencer):
    """A stored non-streamed response vs the same content reassembled from SSE chunks:
    identical extracted answer, usage, thinking content, and thinking_source."""
    body = success_body(content="I think.\nFINAL ANSWER: e2e4", reasoning="Long reasoning about the position.")
    session = ScriptedSession([FakeStreamResponse(sse_lines_from_body(body, piece_len=3))])
    call = inferencer.call_model("prompt", session=session)

    reference_message = body["choices"][0]["message"]
    expected_thinking, expected_source = run_openrouter.extract_thinking(reference_message)
    expected_answer, expected_extracted_ok = run_openrouter.extract_answer(reference_message["content"])

    assert call.ok
    assert call.content == reference_message["content"]
    assert call.usage == body["usage"]
    assert (call.thinking_content, call.thinking_source) == (expected_thinking, expected_source)
    streamed_answer, streamed_ok = run_openrouter.extract_answer(call.content)
    assert (streamed_answer, streamed_ok) == (expected_answer, expected_extracted_ok) == ("e2e4", True)
    # The reassembled message is byte-identical where downstream code looks
    assert call.raw_message["content"] == reference_message["content"]
    assert call.raw_message["reasoning"] == reference_message["reasoning"]
    assert call.raw_message["reasoning_details"] == reference_message["reasoning_details"]


def test_midstream_drop_is_billed_risk_capped(inferencer, no_sleep):
    """A drop AFTER tokens streamed = the provider generated (and billed) output we lost:
    retry under the billed-risk cap, with partial counters recorded per attempt."""
    def dropped():
        lines = sse_lines_from_body(success_body(reasoning="deep thought"))[:4]
        lines.append(requests.ConnectionError("connection reset mid-stream"))
        return FakeStreamResponse(lines)

    session = ScriptedSession([dropped() for _ in range(10)])
    call = inferencer.call_model("prompt", session=session)
    assert not call.ok
    assert len(session.requests_made) == throttle.EXPENSIVE_MAX_ATTEMPTS
    for attempt in call.attempts:
        assert attempt["error_class"] == "stream_drop"
        assert attempt["billed_risk"] == 1
        assert attempt["stream_chunks"] > 0
        assert attempt["reasoning_chars"] > 0
        assert attempt["ttft_ms"] is not None
    assert call.attempts[-1]["outcome"] == "gave_up"
    assert "streamed chunks" in call.content


def test_drop_before_any_token_stays_cheap(inferencer, no_sleep):
    """A drop before the first token = nothing generated: the old cheap connection retry."""
    session = ScriptedSession(
        [FakeStreamResponse([requests.ConnectionError("reset before first byte")]) for _ in range(10)]
    )
    call = inferencer.call_model("prompt", session=session)
    assert not call.ok
    assert len(session.requests_made) == inferencer.max_retries, "cheap retries use the full budget"
    assert all(attempt["error_class"] == "connection" for attempt in call.attempts)
    assert all(attempt["billed_risk"] == 0 for attempt in call.attempts)
    assert all(attempt["stream_chunks"] == 0 for attempt in call.attempts)


def test_midstream_drop_then_success(inferencer, no_sleep):
    lines = sse_lines_from_body(success_body(reasoning="deep thought"))[:4]
    lines.append(requests.ConnectionError("blip"))
    session = ScriptedSession([FakeStreamResponse(lines), streamed_success(reasoning="deep thought")])
    call = inferencer.call_model("prompt", session=session)
    assert call.ok and call.content == "FINAL ANSWER: e2e4"
    assert [attempt["error_class"] for attempt in call.attempts] == ["stream_drop", "ok"]


def test_clean_eof_without_terminator_is_stream_drop(inferencer, no_sleep):
    """Observed live (qwen3.7-max, 2026-07-12): the gateway cuts streams at a hard ~785s
    total-duration ceiling with a CLEAN close — no exception, no [DONE], no finish_reason,
    no usage. That must be a billed-risk stream_drop retry, never a silent empty success."""
    def truncated():
        lines = sse_lines_from_body(success_body(reasoning="deep thought"))[:4]  # cut before content/usage/[DONE]
        return FakeStreamResponse(lines)

    session = ScriptedSession([truncated() for _ in range(10)])
    call = inferencer.call_model("prompt", session=session)
    assert not call.ok
    assert len(session.requests_made) == throttle.EXPENSIVE_MAX_ATTEMPTS
    assert all(attempt["error_class"] == "stream_drop" for attempt in call.attempts)
    assert all(attempt["billed_risk"] == 1 for attempt in call.attempts)
    assert "truncated upstream" in call.content


def test_forged_done_without_completion_evidence_is_stream_drop(inferencer, no_sleep):
    """Observed live (qwen3.7-max, 2026-07-12, second wave): when the gateway kills a
    stream at its ~785s duration ceiling it sends a graceful [DONE] on the way out, so
    terminator-presence checks pass the corpse as success. Completion requires POSITIVE
    evidence — finish_reason or usage — not just [DONE]."""
    def forged():
        lines = sse_lines_from_body(success_body(reasoning="deep thought"))[:4]  # deltas only
        lines.append(b"data: [DONE]")
        return FakeStreamResponse(lines)

    session = ScriptedSession([forged() for _ in range(10)])
    call = inferencer.call_model("prompt", session=session)
    assert not call.ok
    assert len(session.requests_made) == throttle.EXPENSIVE_MAX_ATTEMPTS
    assert all(attempt["error_class"] == "stream_drop" for attempt in call.attempts)
    assert all(attempt["saw_done"] is True for attempt in call.attempts), "forensics: [DONE] was forged"
    assert "truncated upstream" in call.content


def test_in_stream_error_event_is_stream_drop(inferencer, no_sleep):
    lines = sse_lines_from_body(success_body(reasoning="deep thought"))[:4]
    lines.append(b'data: {"error": {"code": 502, "message": "provider disconnected"}}')
    session = ScriptedSession([FakeStreamResponse(lines), streamed_success()])
    call = inferencer.call_model("prompt", session=session)
    assert call.ok, "error event retried, second attempt succeeded"
    assert call.attempts[0]["error_class"] == "stream_drop"
    assert call.attempts[0]["stream_chunks"] > 0


def test_usage_without_done_is_accepted(inferencer):
    """A stream that delivers usage (or finish_reason) but no [DONE] line is complete —
    only a stream with no terminator of any kind counts as truncated."""
    body = success_body()
    lines = [line for line in sse_lines_from_body(body) if not line.startswith(b"data: [DONE]")]
    session = ScriptedSession([FakeStreamResponse(lines)])
    call = inferencer.call_model("prompt", session=session)
    assert call.ok and call.content == "FINAL ANSWER: e2e4"
    assert call.usage == body["usage"]


def test_merge_reasoning_fragments_without_index():
    """Index-less fragments continue the most recent entry of the same type; a type
    switch starts a new entry (so extract_thinking's per-entry newline join stays valid)."""
    entries, indexed = [], {}
    for fragment in [
        {"type": "reasoning.text", "text": "Hel"},
        {"type": "reasoning.text", "text": "lo"},
        {"type": "reasoning.summary", "summary": "short"},
        {"type": "reasoning.text", "text": "again", "signature": "sig-1"},
    ]:
        run_openrouter._merge_reasoning_fragment(entries, indexed, fragment)
    assert entries == [
        {"type": "reasoning.text", "text": "Hello"},
        {"type": "reasoning.summary", "summary": "short"},
        {"type": "reasoning.text", "text": "again", "signature": "sig-1"},
    ]


# --- Claude 5 adaptive thinking: native-endpoint routing -------------------------------------


def test_adaptive_thinking_version_detection():
    assert run_openrouter.anthropic_adaptive_thinking("anthropic/claude-sonnet-5") is True
    assert run_openrouter.anthropic_adaptive_thinking("anthropic/claude-opus-5.1") is True
    assert run_openrouter.anthropic_adaptive_thinking("anthropic/claude-fable-5") is True
    # Probe-verified 2026-07-10: the adaptive interface starts at 4.7, not 5 — opus-4.7/4.8
    # reject thinking.type.enabled, while opus-4.6/sonnet-4.6/haiku-4.5 accept it.
    assert run_openrouter.anthropic_adaptive_thinking("anthropic/claude-opus-4.8") is True
    assert run_openrouter.anthropic_adaptive_thinking("anthropic/claude-opus-4.7") is True
    assert run_openrouter.anthropic_adaptive_thinking("anthropic/claude-opus-4.6") is False
    assert run_openrouter.anthropic_adaptive_thinking("anthropic/claude-sonnet-4.6") is False
    assert run_openrouter.anthropic_adaptive_thinking("anthropic/claude-haiku-4.5") is False
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
    session = ScriptedSession([streamed_success()])
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
    session = ScriptedSession([streamed_success() for _ in range(30)])
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


def test_resume_keeps_capped_rows_unless_retry_capped():
    """max_token_reached rows are recorded experimental outcomes: resume keeps them by
    default and only re-runs them under --retry-capped (upstream's original behavior)."""
    tasks = [make_task(0), make_task(1), make_task(2)]

    def result_for(task, error_type, response="FINAL ANSWER: e2e4"):
        row = dict(task)
        row["inference"] = {"response": response, "error_type": error_type}
        return row

    existing = {
        tasks[0]["task_id"]: result_for(tasks[0], "correct"),
        tasks[1]["task_id"]: result_for(tasks[1], "max_token_reached"),
        tasks[2]["task_id"]: result_for(tasks[2], "wrong_answer", response="ERROR: gave up"),
    }

    incomplete, complete = run_openrouter.filter_incomplete_tasks(tasks, existing)
    assert [task["task_id"] for task in incomplete] == [tasks[2]["task_id"]], "only the ERROR row re-runs"
    assert len(complete) == 2, "the capped row is kept as a recorded outcome"

    incomplete, complete = run_openrouter.filter_incomplete_tasks(tasks, existing, retry_capped=True)
    assert {task["task_id"] for task in incomplete} == {tasks[1]["task_id"], tasks[2]["task_id"]}
    assert len(complete) == 1
