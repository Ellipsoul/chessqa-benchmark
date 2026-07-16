"""ChessQA evaluation runner: benchmark JSONL -> LLM gateway -> scored results.

Descended from the single-file harness used for all of the paper's 23 runs. All inference
is routed through the Vercel AI Gateway's OpenAI-compatible endpoint (plus its
Anthropic-native /v1/messages endpoint for adaptive-thinking Claude models). Auth via the
``AI_GATEWAY_API_KEY`` env var (fallback: ``VERCEL_OIDC_TOKEN``). Verified live: the
gateway returns per-call dollar cost in ``usage`` (cost/gateway_cost/market_cost), so
cost columns populate; the dashboard adds request-level logs.

End-to-end flow:

1. Load every ``*.jsonl`` under ``--dataset-root`` (optionally subsampled per task type).
2. Resume: match tasks against the existing results file by ``task_id``; only unfinished
   tasks are re-run (``max_token_reached`` rows are kept as recorded outcomes unless
   ``--retry-capped`` is passed). ``--no-resume`` skips this.
3. For each task, ``format_prompt`` resolves the placeholders baked into the question
   (CONTEXT_PLACEHOLDER, FORMAT_EXAMPLE_PLACEHOLDER) according to ``--add-context`` and
   ``--use-format-example-group``.
4. Fan out over a ``ThreadPoolExecutor`` (``--workers``) paced by a shared rate limiter
   (``--rps``/``--burst``; see eval/throttle.py for the retry/backoff policy); each worker
   POSTs to the gateway as a STREAMING (SSE) request — reassembled by
   ``consume_chat_sse`` (chat completions) or ``consume_anthropic_sse`` (the
   Anthropic-native /v1/messages path for adaptive-thinking models) into the
   non-streaming message shape — so long generations keep bytes moving and survive the
   gateway's ~340s wire-idle kill (Incident 2,
   docs/model-trials/2026-07-12-smoke-campaign.md). Every attempt is recorded, with
   ``ttft_ms`` and partial stream counters; thinking traces come from the reassembled
   ``reasoning``/``reasoning_details`` fields, tagged with a ``thinking_source`` fidelity
   label (full_text / summary / encrypted_only / ...) per task. A monitor thread shows
   live tokens/s and cumulative recorded cost in the tqdm postfix.
5. ``extract_answer`` takes the last ``FINAL ANSWER:`` line (with ``\\boxed{}`` fallback);
   ``evaluate_answer_with_error_type`` scores it (exact match for "single", set match for
   "multi") and classifies failures. Note this classifies *answers only* — nothing inspects
   the reasoning trace (that gap is this project's Phase 3).

Outputs, all under ``--output-dir`` and all named ``<model with / and : -> _>`` plus
variant suffixes ``-thinking`` / ``-piecearr`` / ``-fmt2`` (flags must match for resume
and ``--eval-only`` to find the file):
- ``<name>.jsonl``  one result per line: the full task + an ``inference`` block (prompt,
  response, thinking_content, thinking_source, extracted answer, correctness, error_type,
  usage, raw_message, provider_meta, attempts, latency_ms). This file is CANONICAL.
- ``<name>_stats.json``  aggregate accuracy/cost/error-type/per-category stats.
- ``chessqa.sqlite3``  derived, rebuildable cross-run index (see eval/storage.py); skip
  with ``--no-db``, rebuild anytime with ``python eval/storage.py ingest results/*.jsonl``.

``--eval-only`` re-runs steps 5+ on an existing results file without any API calls —
useful after changing extraction/scoring logic.
"""

import argparse
import contextlib
import datetime
import json
import os
import random
import re
import sqlite3
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import chess
import requests
from tqdm import tqdm

import storage
import throttle

GATEWAY_URL = "https://ai-gateway.vercel.sh/v1/chat/completions"

# One requests.Session per worker thread: connection pooling without sharing a Session
# across threads (not documented thread-safe).
_thread_local = threading.local()


def _get_thread_session() -> requests.Session:
    session = getattr(_thread_local, "session", None)
    if session is None:
        session = requests.Session()
        _thread_local.session = session
    return session


ANTHROPIC_NATIVE_URL = "https://ai-gateway.vercel.sh/v1/messages"


def anthropic_adaptive_thinking(model: str) -> bool:
    """True for Anthropic models on the *adaptive thinking* interface.

    Detected by the trailing version in the slug (``anthropic/claude-sonnet-5`` -> 5.0,
    ``anthropic/claude-haiku-4.5`` -> 4.5). Adaptive models reject the classic
    ``thinking.type: enabled`` budget interface (and every chat-completions ``reasoning``
    shape silently no-ops), so they must be routed to the Anthropic-native endpoint with
    ``thinking.type: adaptive``.

    The boundary is version >= 4.7, not >= 5: probe-verified 2026-07-10 against the
    gateway's /v1/messages endpoint — claude-opus-4.7 and claude-opus-4.8 reject
    ``thinking.type.enabled`` ("Use thinking.type.adaptive"), while claude-opus-4.6,
    claude-sonnet-4.6, and claude-haiku-4.5 accept it and return full raw thinking blocks.
    Adaptive models also cap ``thinking.display`` at ``summarized`` (``full`` rejected),
    so 4.6-and-earlier models are the only Anthropic source of full_text traces.
    """
    if not model.startswith("anthropic/claude"):
        return False
    version_token = model.rsplit("-", 1)[-1]
    try:
        return float(version_token) >= 4.7
    except ValueError:
        return False


def build_reasoning_payload(model: str) -> dict[str, Any]:
    """Return the thinking config sent when --enable-thinking is set (probe-verified 2026-07-10).

    Two regimes, selected per model:

    - Classic-thinking models (e.g. claude-haiku-4.5, and non-Anthropic reasoning models):
      ``reasoning: {"effort": "medium"}`` on the chat-completions endpoint —
      the paper's setting; yields FULL-TEXT traces (``thinking_source: full_text``).
    - Adaptive-thinking models — Claude 5 family and Opus 4.7+ (``anthropic_adaptive_thinking``): full raw
      CoT is withheld by Anthropic (``thinking.display`` accepts only
      ``"omitted" | "summarized"``), and the OpenAI-compatible endpoint cannot express
      adaptive thinking at all (every ``reasoning`` shape no-ops; ``display`` is not passed
      through). The runner therefore routes these calls to the gateway's Anthropic-native
      /v1/messages endpoint with ``{"type": "adaptive", "display": "summarized"}``, which
      returns a REAL summarized trace (``thinking_source: summary``) — usable for coarse
      analysis, but not the verbatim CoT that Phase 3 line-verification needs.

    The returned dict is also recorded per run in ``runs.reasoning_config`` provenance.
    """
    if anthropic_adaptive_thinking(model):
        return {"type": "adaptive", "display": "summarized"}
    return {"effort": "medium"}


def parse_native_anthropic_message(result: dict[str, Any]) -> tuple[str, str, str, dict[str, Any]]:
    """Normalize an Anthropic-native /v1/messages response to the harness's shape.

    Returns ``(content, thinking_content, thinking_source, usage)`` where usage carries the
    chat-completions key names the rest of the pipeline expects (prompt_tokens /
    completion_tokens / total_tokens) alongside the original native fields. thinking_source:
    ``summary`` when a thinking block carries text under display=summarized;
    ``encrypted_only`` when a thinking block exists but its text was withheld (signature
    only); ``none`` when the model chose not to think (adaptive skips easy prompts).

    Known trade-off: the native endpoint's usage has no per-call ``cost`` field (verified
    live), so stats cost columns are zero for native-routed runs — reconcile spend from the
    gateway dashboard or compute from token prices.
    """
    text_parts = []
    thinking_parts = []
    saw_thinking_block = False
    for block in result.get("content", []) or []:
        block_type = block.get("type")
        if block_type == "text" and block.get("text"):
            text_parts.append(block["text"])
        elif block_type in ("thinking", "redacted_thinking"):
            saw_thinking_block = True
            if block.get("thinking"):
                thinking_parts.append(block["thinking"])

    if thinking_parts:
        thinking_content, thinking_source = "\n".join(thinking_parts), "summary"
    elif saw_thinking_block:
        thinking_content, thinking_source = "", "encrypted_only"
    else:
        thinking_content, thinking_source = "", "none"

    native_usage = result.get("usage", {}) or {}
    input_tokens = native_usage.get("input_tokens", 0)
    output_tokens = native_usage.get("output_tokens", 0)
    usage = {
        "prompt_tokens": input_tokens,
        "completion_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        **native_usage,
    }
    return "\n".join(text_parts).strip(), thinking_content, thinking_source, usage


@dataclass
class ModelCall:
    """Everything one call_model invocation produced, including its retry history."""

    content: str
    thinking_content: str = ""
    thinking_source: str = "none"
    usage: dict[str, Any] = field(default_factory=dict)
    raw_message: dict[str, Any] | None = None
    provider_meta: dict[str, Any] | None = None
    attempts: list[dict[str, Any]] = field(default_factory=list)
    latency_ms: int | None = None
    ttft_ms: int | None = None
    ok: bool = False
    error: str | None = None


def load_env_file(env_path: Path | None = None) -> None:
    """Load KEY=VALUE lines from the repo-root ``.env`` into os.environ.

    Stdlib-only stand-in for python-dotenv, so credentials can live in a gitignored
    file instead of the shell profile. Real environment variables always win — values
    from the file are applied only for keys not already set. Blank lines, ``#`` comments,
    an optional ``export `` prefix, and single/double quotes around values are tolerated.
    Called once at the top of main(); worker processes inherit the parent's environment.
    """
    if env_path is None:
        env_path = Path(__file__).parent.parent / ".env"
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export ") :]
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip("'\"")
        if key and key not in os.environ:
            os.environ[key] = value


def resolve_api_key() -> str:
    """Resolve the Vercel AI Gateway API key, failing fast with a setup hint.

    ``AI_GATEWAY_API_KEY`` env var, falling back to ``VERCEL_OIDC_TOKEN`` (the short-lived
    JWT written to .env.local by ``vercel env pull`` — only useful if the caller exported
    it into the environment). Either variable may come from the shell or from the
    repo-root ``.env`` file (see ``load_env_file``).
    """
    api_key = os.environ.get("AI_GATEWAY_API_KEY") or os.environ.get("VERCEL_OIDC_TOKEN")
    if not api_key:
        raise SystemExit(
            "No Vercel AI Gateway credential found. Put AI_GATEWAY_API_KEY=... in the "
            "repo-root .env file (cp .env.example .env), or export AI_GATEWAY_API_KEY / "
            "VERCEL_OIDC_TOKEN in your shell. Keys are created in the Vercel dashboard "
            "under AI Gateway."
        )
    return api_key


def extract_thinking(message: dict[str, Any]) -> tuple[str, str]:
    """Extract the thinking trace and classify its fidelity.

    The Vercel AI Gateway returns a typed
    ``reasoning_details`` array whose block types make trace fidelity machine-legible —
    the signal that gates Phase 3 depth:
    - ``reasoning.text``: the actual chain-of-thought (Anthropic-style, may be signed).
    - ``reasoning.summary``: a condensed rendering (OpenAI-style) — NOT the real trace.
    - ``reasoning.encrypted``: redacted/protected content, unusable for analysis.

    Returns ``(thinking_content, thinking_source)`` where source is one of:
    ``full_text`` | ``summary`` | ``untyped`` (legacy dicts with only a "text" key, or
    bare strings — fidelity unknown) | ``plain`` (only the flat ``message.reasoning``
    string was present) | ``encrypted_only`` | ``none``.
    """
    details = message.get("reasoning_details") or []
    text_parts = [
        detail.get("text")
        for detail in details
        if isinstance(detail, dict) and detail.get("type") == "reasoning.text" and detail.get("text")
    ]
    summary_parts = [
        detail.get("summary")
        for detail in details
        if isinstance(detail, dict) and detail.get("type") == "reasoning.summary" and detail.get("summary")
    ]
    untyped_parts = [
        detail.get("text") for detail in details if isinstance(detail, dict) and "type" not in detail and detail.get("text")
    ]
    untyped_parts += [detail for detail in details if isinstance(detail, str)]

    if text_parts:
        return "\n".join(text_parts), "full_text"
    if summary_parts:
        return "\n".join(summary_parts), "summary"
    if untyped_parts:
        return "\n".join(untyped_parts), "untyped"
    if message.get("reasoning"):
        return str(message["reasoning"]), "plain"
    if any(isinstance(detail, dict) and detail.get("type") == "reasoning.encrypted" for detail in details):
        return "", "encrypted_only"
    return "", "none"


class StreamDrop(Exception):
    """Transport failure while consuming a streamed (SSE) chat-completions response.

    Carries how much had already arrived so retry classification can distinguish a
    mid-generation drop (tokens streamed -> the provider generated, and billed, output;
    retrying regenerates it, so it falls under the billed-risk attempt cap) from a drop
    before any token arrived (the old cheap pre-generation connection failure).
    """

    def __init__(self, cause: Exception, stream_info: dict[str, Any]):
        super().__init__(f"{type(cause).__name__}: {cause}")
        self.cause = cause
        self.stream_info = stream_info


def _merge_reasoning_fragment(entries: list[Any], indexed: dict[Any, dict[str, Any]], fragment: Any) -> None:
    """Fold one streamed reasoning_details fragment into the accumulated entries.

    Providers stream reasoning_details as many small fragments of what the non-streaming
    response returns as one block per type. Fragments carrying an ``index`` merge into
    the entry first seen with that index; index-less fragments continue the most recent
    entry of the same type. String payload fields (text/summary/data) concatenate;
    everything else (format, signature, id) last-write-wins. The ``index`` key itself is
    dropped so the reassembled message matches the non-streaming shape.
    """
    if not isinstance(fragment, dict):
        # extract_thinking tolerates bare strings inside reasoning_details; keep them.
        if fragment:
            entries.append(fragment)
        return
    key = fragment.get("index")
    entry = None
    if key is not None:
        entry = indexed.get(key)
    elif entries and isinstance(entries[-1], dict) and entries[-1].get("type") == fragment.get("type"):
        entry = entries[-1]
    if entry is None:
        entry = {name: value for name, value in fragment.items() if name != "index"}
        entries.append(entry)
        if key is not None:
            indexed[key] = entry
        return
    for name, value in fragment.items():
        if name == "index" or value is None:
            continue
        if name in ("text", "summary", "data") and isinstance(value, str) and isinstance(entry.get(name), str):
            entry[name] += value
        else:
            entry[name] = value


def consume_chat_sse(
    response: requests.Response, started_at_monotonic: float, on_delta=None
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Drain a streamed chat-completions response and reassemble the non-streaming body.

    This is the core of the 340s-wall fix: with ``stream: true`` the gateway forwards
    bytes as the provider generates them, so the wire is never silent long enough for the
    gateway's ~340s idle timer to kill the socket (Incident 2 in
    docs/model-trials/2026-07-12-smoke-campaign.md). The reassembled dict is shaped
    exactly like the non-streaming response — ``choices[0].message`` with content /
    reasoning / typed reasoning_details, ``usage`` from the final chunk (requested via
    ``stream_options.include_usage``) — so extract_thinking, thinking_source tagging,
    cost capture, and storage see no difference.

    Returns ``(result, stream_info)``. stream_info records ``ttft_ms`` (first token,
    measured from ``started_at_monotonic``), ``stream_chunks`` (delta events — the best
    available mid-flight proxy for completion tokens; the exact count only exists in the
    final usage chunk), and content/reasoning character counts. ``on_delta`` (if given)
    is called once per delta event for live progress reporting.

    Failure modes: any exception while iterating the stream raises StreamDrop carrying
    the partial stream_info (transport truncation is overwhelmingly more likely than a
    provider emitting malformed JSON mid-stream); a 2xx that yields no SSE data events,
    or events with neither deltas nor usage, raises ValueError — classified "parse"
    downstream: billed and received, never regenerate.
    """
    content_parts: list[str] = []
    reasoning_parts: list[str] = []
    detail_entries: list[Any] = []
    indexed_details: dict[Any, dict[str, Any]] = {}
    meta: dict[str, Any] = {}
    usage: dict[str, Any] | None = None
    finish_reason = None
    role = None
    data_events = 0
    delta_events = 0
    saw_done = False
    first_token_ms: int | None = None

    def stream_info() -> dict[str, Any]:
        return {
            "ttft_ms": first_token_ms,
            "stream_chunks": delta_events,
            "content_chars": sum(len(part) for part in content_parts),
            "reasoning_chars": sum(len(part) for part in reasoning_parts),
            "saw_done": saw_done,
        }

    try:
        for raw_line in response.iter_lines():
            if not raw_line:
                continue
            line = raw_line.decode("utf-8", errors="replace") if isinstance(raw_line, bytes) else raw_line
            if line.startswith(":"):  # SSE comment keepalive
                continue
            if not line.startswith("data:"):
                continue
            payload = line[len("data:") :].strip()
            if payload == "[DONE]":
                saw_done = True
                break
            chunk = json.loads(payload)
            data_events += 1
            if chunk.get("error"):
                # Mid-stream error event (gateway convention): the provider or
                # gateway aborted the generation. Billed-risk if tokens already flowed.
                raise StreamDrop(
                    RuntimeError(f"in-stream error event: {json.dumps(chunk['error'])[:300]}"), stream_info()
                )
            for key in ("id", "model", "provider", "created"):
                if chunk.get(key) is not None and key not in meta:
                    meta[key] = chunk[key]
            if chunk.get("usage"):
                usage = chunk["usage"]
            choices = chunk.get("choices") or []
            if not choices:
                continue
            choice = choices[0]
            if choice.get("finish_reason"):
                finish_reason = choice["finish_reason"]
            delta = choice.get("delta") or {}
            if delta.get("role"):
                role = delta["role"]
            got_tokens = False
            if delta.get("content"):
                content_parts.append(delta["content"])
                got_tokens = True
            if delta.get("reasoning"):
                reasoning_parts.append(delta["reasoning"])
                got_tokens = True
            for fragment in delta.get("reasoning_details") or []:
                _merge_reasoning_fragment(detail_entries, indexed_details, fragment)
                got_tokens = True
            if got_tokens:
                delta_events += 1
                if first_token_ms is None:
                    first_token_ms = int((time.monotonic() - started_at_monotonic) * 1000)
                if on_delta is not None:
                    on_delta()
    except StreamDrop:
        raise
    except Exception as stream_error:
        raise StreamDrop(stream_error, stream_info()) from stream_error

    if data_events == 0:
        raise ValueError("2xx response produced no SSE data events (streaming not honored?)")
    if delta_events == 0 and usage is None:
        raise ValueError(f"SSE stream had {data_events} data events but no message deltas or usage")
    if usage is None and finish_reason is None:
        # No POSITIVE completion evidence. A complete gateway stream carries finish_reason
        # and (with stream_options.include_usage) a usage chunk; their joint absence means
        # the generation was cut, EVEN IF a [DONE] line arrived — observed live 2026-07-12:
        # the gateway kills streams at a hard ~785s total-duration ceiling (distinct from
        # the fixed 340s idle wall) and forges a graceful [DONE] on the way out, so
        # absence-of-terminator checks pass the corpse as success (six qwen3.7-max streams,
        # all cut at exactly 785.1s, single attempt marked "ok", empty content). Treat as a
        # dropped stream (billed-risk retry), never as a successful empty response.
        raise StreamDrop(
            RuntimeError(
                "SSE stream ended without finish_reason or usage "
                f"(saw_done={saw_done}) — truncated upstream"
            ),
            stream_info(),
        )

    message: dict[str, Any] = {"role": role or "assistant", "content": "".join(content_parts)}
    if reasoning_parts:
        message["reasoning"] = "".join(reasoning_parts)
    if detail_entries:
        message["reasoning_details"] = detail_entries
    result = dict(meta)
    result["choices"] = [{"message": message, "finish_reason": finish_reason}]
    if usage is not None:
        result["usage"] = usage
    return result, stream_info()


def consume_anthropic_sse(
    response: requests.Response, started_at_monotonic: float, on_delta=None
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Drain a streamed Anthropic-native /v1/messages response into the non-streaming shape.

    The native endpoint has its own SSE grammar — typed events instead of the chat path's
    uniform chunks: ``message_start`` (message skeleton + input usage),
    ``content_block_start``/``content_block_delta`` (``thinking_delta`` / ``text_delta`` /
    ``signature_delta`` fragments per indexed block), ``message_delta`` (stop_reason +
    output usage), ``message_stop``, plus ``ping`` keepalives and ``error`` events.
    The reassembled dict matches what ``parse_native_anthropic_message`` already parses.

    Streaming this path became necessary on 2026-07-13: adaptive-thinking Claude models
    were assumed to answer quickly (probe-verified on short prompts), but on hard chess
    tasks with a 32K budget sonnet-5 generates for 380-400s — past the gateway's ~340s
    idle wall for NON-streaming requests. Each wall kill was billed upstream (a 200 on
    the gateway dashboard) while the client saw a connection error and re-requested:
    70+ billed calls for a 49-task smoke.

    Completion requires positive evidence — a ``message_stop`` event or a ``stop_reason``
    — mirroring the chat path's forged-terminator defense. Transport failures and
    in-stream error events raise StreamDrop with partial counters.
    """
    message: dict[str, Any] = {}
    blocks: dict[int, dict[str, Any]] = {}
    usage: dict[str, Any] = {}
    stop_reason = None
    saw_message_stop = False
    data_events = 0
    delta_events = 0
    first_token_ms: int | None = None

    def stream_info() -> dict[str, Any]:
        return {
            "ttft_ms": first_token_ms,
            "stream_chunks": delta_events,
            "content_chars": sum(len(block.get("text") or "") for block in blocks.values()),
            "reasoning_chars": sum(len(block.get("thinking") or "") for block in blocks.values()),
            "saw_done": saw_message_stop,
        }

    try:
        for raw_line in response.iter_lines():
            if not raw_line:
                continue
            line = raw_line.decode("utf-8", errors="replace") if isinstance(raw_line, bytes) else raw_line
            if line.startswith(":") or not line.startswith("data:"):
                continue  # comments and "event: <type>" lines; the type is repeated in the data
            chunk = json.loads(line[len("data:") :].strip())
            data_events += 1
            chunk_type = chunk.get("type")
            if chunk_type == "error" or chunk.get("error"):
                raise StreamDrop(
                    RuntimeError(f"in-stream error event: {json.dumps(chunk.get('error') or chunk)[:300]}"),
                    stream_info(),
                )
            if chunk_type == "ping":
                continue
            if chunk_type == "message_start":
                start = chunk.get("message") or {}
                message = {key: start[key] for key in ("id", "model", "role") if start.get(key) is not None}
                usage.update(start.get("usage") or {})
            elif chunk_type == "content_block_start":
                blocks[chunk.get("index", 0)] = dict(chunk.get("content_block") or {})
            elif chunk_type == "content_block_delta":
                block = blocks.setdefault(chunk.get("index", 0), {})
                delta = chunk.get("delta") or {}
                delta_type = delta.get("type")
                got_tokens = False
                if delta_type == "thinking_delta" and delta.get("thinking"):
                    block.setdefault("type", "thinking")
                    block["thinking"] = (block.get("thinking") or "") + delta["thinking"]
                    got_tokens = True
                elif delta_type == "text_delta" and delta.get("text"):
                    block.setdefault("type", "text")
                    block["text"] = (block.get("text") or "") + delta["text"]
                    got_tokens = True
                elif delta_type == "signature_delta" and delta.get("signature"):
                    block["signature"] = (block.get("signature") or "") + delta["signature"]
                if got_tokens:
                    delta_events += 1
                    if first_token_ms is None:
                        first_token_ms = int((time.monotonic() - started_at_monotonic) * 1000)
                    if on_delta is not None:
                        on_delta()
            elif chunk_type == "message_delta":
                if (chunk.get("delta") or {}).get("stop_reason"):
                    stop_reason = chunk["delta"]["stop_reason"]
                usage.update(chunk.get("usage") or {})
            elif chunk_type == "message_stop":
                saw_message_stop = True
                break
    except StreamDrop:
        raise
    except Exception as stream_error:
        raise StreamDrop(stream_error, stream_info()) from stream_error

    if data_events == 0:
        raise ValueError("2xx native response produced no SSE data events (streaming not honored?)")
    if not saw_message_stop and stop_reason is None:
        raise StreamDrop(
            RuntimeError("native SSE stream ended without message_stop or stop_reason — truncated upstream"),
            stream_info(),
        )

    result = dict(message)
    result["content"] = [blocks[index] for index in sorted(blocks)]
    if stop_reason is not None:
        result["stop_reason"] = stop_reason
    if usage:
        result["usage"] = usage
    return result, stream_info()


class RunProgress:
    """Thread-safe live counters behind the tqdm postfix (forward progress + burn rate).

    Workers tick ``on_stream_delta`` once per streamed SSE delta event (~one token) and
    ``on_task_done`` with each finished task's usage; a monitor thread snapshots these to
    display live tokens/s and cumulative recorded cost while long generations are still
    in flight — previously the bar only moved on task completion, which for multi-minute
    generations meant zero visible progress while money was being spent.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._stream_chunks = 0
        self._cost = 0.0

    def on_stream_delta(self) -> None:
        with self._lock:
            self._stream_chunks += 1

    def on_task_done(self, usage: dict[str, Any]) -> None:
        with self._lock:
            self._cost += usage.get("cost", 0.0) or 0.0

    def snapshot(self) -> tuple[int, float]:
        with self._lock:
            return self._stream_chunks, self._cost


def _progress_monitor(pbar: tqdm, progress: RunProgress, stop: threading.Event, interval_s: float = 2.0) -> None:
    """Refresh the tqdm postfix with live tokens/s and cumulative run cost.

    tokens/s is approximated from streamed delta events (~one token each; exact counts
    only exist in each task's final usage chunk). Cost sums recorded ``usage.cost`` —
    native-routed Anthropic runs report none (see parse_native_anthropic_message), so
    their burn rate must still come from the gateway credits meter.
    """
    last_chunks, last_time = progress.snapshot()[0], time.monotonic()
    while not stop.wait(interval_s):
        chunks, cost = progress.snapshot()
        now = time.monotonic()
        rate = (chunks - last_chunks) / max(now - last_time, 1e-6)
        last_chunks, last_time = chunks, now
        with tqdm.get_lock():
            pbar.set_postfix({"tok/s": f"{rate:.0f}", "cost": f"${cost:.2f}"}, refresh=True)


def load_tasks(
    dataset_root: Path, max_tasks: int | None = None, n_samples_per_task: int | None = None
) -> list[dict[str, Any]]:
    """Load tasks from all JSONL files in dataset root directory with optional sampling per task type.

    ``n_samples_per_task`` shuffles within each task_type and keeps the first N;
    ``max_tasks`` then truncates the combined list (applied after sampling).
    """
    tasks = []

    # Find all JSONL files in the dataset root
    jsonl_files = list(dataset_root.glob("*.jsonl"))
    if not jsonl_files:
        raise ValueError(f"No JSONL files found in {dataset_root}")

    print(f"Found {len(jsonl_files)} JSONL files:")
    for file in sorted(jsonl_files):
        print(f"  - {file.name}")

    # Load tasks from all JSONL files
    for file_path in sorted(jsonl_files):
        print(f"Loading {file_path.name}...")
        file_tasks = []
        with open(file_path, encoding="utf-8") as task_file:
            for line in task_file:
                if line.strip():
                    file_tasks.append(json.loads(line.strip()))
        print(f"  Loaded {len(file_tasks)} tasks")
        tasks.extend(file_tasks)

    print(f"Total tasks loaded: {len(tasks)}")

    if n_samples_per_task:
        # Group tasks by task_type
        tasks_by_type = {}
        for task in tasks:
            task_type = task.get("task_type", "unknown")
            if task_type not in tasks_by_type:
                tasks_by_type[task_type] = []
            tasks_by_type[task_type].append(task)

        # Sample n_samples_per_task from each type.
        # Fixed relative to upstream: the shuffle now uses a dedicated fixed-seed RNG, so
        # repeated --N-samples-per-task runs draw the same subset (upstream claimed a fixed
        # seed in a comment but used the unseeded global RNG). This also makes resume
        # coherent for subsampled runs — previously each invocation sampled a different
        # subset, so resumed runs silently evaluated a mix of subsets.
        subsample_random_generator = random.Random(42)
        sampled_tasks = []
        for task_type, type_tasks in tasks_by_type.items():
            if len(type_tasks) > n_samples_per_task:
                subsample_random_generator.shuffle(type_tasks)
                sampled = type_tasks[:n_samples_per_task]
            else:
                sampled = type_tasks
            sampled_tasks.extend(sampled)
            print(f"Task type '{task_type}': {len(sampled)}/{len(type_tasks)} tasks selected")

        tasks = sampled_tasks

    if max_tasks and len(tasks) > max_tasks:
        tasks = tasks[:max_tasks]

    return tasks


def get_context(fen: str) -> str:
    """Build the --add-context injection string: piece arrangement + legal moves for a FEN.

    This is the paper's key intervention — handing the model an explicit board state to
    bypass FEN-parsing/board-hallucination failures. The arrangement uses the same canonical
    ordering as piece_arrangement *answers* (dataset/utils.get_piece_arrangement): White then
    Black, King/Queen/Rook/Bishop/Knight/Pawn, squares alphabetical. (Fixed relative to
    upstream, which emitted board-scan a1..h8 order — an inconsistency that made the injected
    context clash with the answer format piece_arrangement tasks demand. Paper --add-context
    runs used the old ordering, so piecearr variants are not byte-comparable to upstream.)

    Some dataset entries append move hints after a pipe ("|") like:
    "<FEN> | e2e4 e7e5". Strip that part before parsing. Returns "" if the FEN is unparseable.
    """
    fen_clean = fen.split("|", 1)[0].strip()
    try:
        board = chess.Board(fen_clean)
    except Exception:
        # Fallback: try to coerce whitespace and retry; otherwise return minimal context
        try:
            board = chess.Board(" ".join(fen_clean.split()))
        except Exception:
            return ""

    # Piece arrangement
    pieces = {}
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            color = "White" if piece.color == chess.WHITE else "Black"
            names = {1: "Pawn", 2: "Knight", 3: "Bishop", 4: "Rook", 5: "Queen", 6: "King"}
            piece_key = f"{color} {names[piece.piece_type]}"
            if piece_key not in pieces:
                pieces[piece_key] = []
            pieces[piece_key].append(chess.square_name(square))

    # Canonical presentation order, mirroring dataset/utils.get_piece_arrangement.
    for squares in pieces.values():
        squares.sort()
    piece_order = ["King", "Queen", "Rook", "Bishop", "Knight", "Pawn"]
    ordered_keys = [f"{color} {piece_type}" for color in ("White", "Black") for piece_type in piece_order]
    arrangement = ", ".join(f"{piece_key}: {pieces[piece_key]}" for piece_key in ordered_keys if piece_key in pieces)
    legal_moves = ", ".join(sorted(move.uci() for move in board.legal_moves))

    return f"Piece arrangement: {arrangement}\nLegal moves: {legal_moves}\n\n"


def format_prompt(task: dict[str, Any], add_context: bool = False, format_example_group: int = 1) -> str:
    """Resolve a task's question template into the final prompt string.

    CONTEXT_PLACEHOLDER becomes the get_context block (or "" without --add-context);
    FORMAT_EXAMPLE_PLACEHOLDER becomes format_examples[0] or [1] depending on
    ``format_example_group`` — the mechanism behind the paper's format-sensitivity variant.
    """
    question = task["question"]

    if add_context and "input" in task:
        context = get_context(task["input"])
        question = question.replace("CONTEXT_PLACEHOLDER", context)
    else:
        question = question.replace("CONTEXT_PLACEHOLDER", "")

    if "format_examples" in task and task["format_examples"]:
        examples_list = task["format_examples"]
        if len(examples_list) >= 2:
            # Select specific example based on group (1 or 2)
            example = examples_list[1] if format_example_group == 2 and len(examples_list) >= 2 else examples_list[0]
        else:
            # Fallback to first/only example
            example = examples_list[0] if examples_list else ""
        question = question.replace("FORMAT_EXAMPLE_PLACEHOLDER", example)

    return question


def extract_answer(response: str) -> tuple[str, bool]:
    """Pull the model's answer out of its response text.

    Primary: the *last* ``FINAL ANSWER: ...`` line (case-insensitive; last wins because
    models sometimes restate the header while reasoning). Fallback: the last
    ``The final answer is \\boxed{...}`` (some models default to this despite instructions).

    Returns ``(answer, True)`` or ``("", False)``; the False case is scored as
    ``format_error`` downstream, so extraction robustness directly shapes the paper's
    "format following rate" metric.
    """
    # Look for the last occurrence of FINAL ANSWER: in the response
    matches = list(re.finditer(r"FINAL ANSWER:\s*(.+?)(?:\n|$)", response, re.IGNORECASE | re.DOTALL))
    if matches:
        # Take the last match and extract only the answer part (group 1)
        answer = matches[-1].group(1).strip()
        # Remove any leading "FINAL ANSWER:" if it got captured
        answer = re.sub(r"^FINAL ANSWER:\s*", "", answer, flags=re.IGNORECASE).strip()
        # Strip markdown formatting (**, *, etc.)
        answer = re.sub(r"^\*+|\*+$", "", answer).strip()
        # Strip any remaining whitespace including newlines
        answer = answer.strip()
        return answer, True

    # Try to extract from "The final answer is $\boxed{...}$" format
    boxed_matches = list(re.finditer(r"[Tt]he\s+final\s+answer\s+is\s+\$?\\boxed\{([^}]+)\}\$?", response))
    if boxed_matches:
        # Take the last match and extract the content inside \boxed{}
        answer = boxed_matches[-1].group(1).strip()
        return answer, True

    return "", False


def calculate_total_usage(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate cost/token/extraction stats across results (a result with no usage dict
    counts as an API error). Costs come from the gateway's per-call usage accounting."""
    total_cost = 0.0
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_tokens = 0
    error_count = 0
    extraction_success_count = 0

    for result in results:
        usage = result.get("inference", {}).get("usage", {})
        if usage:
            total_cost += usage.get("cost", 0.0)
            total_prompt_tokens += usage.get("prompt_tokens", 0)
            total_completion_tokens += usage.get("completion_tokens", 0)
            total_tokens += usage.get("total_tokens", 0)
        else:
            error_count += 1

        # Track extraction success
        if result.get("inference", {}).get("extraction_successful", False):
            extraction_success_count += 1

    return {
        "total_cost": total_cost,
        "total_prompt_tokens": total_prompt_tokens,
        "total_completion_tokens": total_completion_tokens,
        "total_tokens": total_tokens,
        "error_count": error_count,
        "extraction_success_count": extraction_success_count,
        "extraction_success_rate": extraction_success_count / len(results) if results else 0.0,
        "avg_cost_per_task": total_cost / len(results) if results else 0.0,
        "avg_tokens_per_task": total_tokens / len(results) if results else 0.0,
        "cost_per_1k_tokens": (total_cost / (total_tokens / 1000)) if total_tokens > 0 else 0.0,
        "tokens_per_dollar": (total_tokens / total_cost) if total_cost > 0 else 0.0,
        "avg_prompt_tokens": total_prompt_tokens / len(results) if results else 0.0,
        "avg_completion_tokens": total_completion_tokens / len(results) if results else 0.0,
        "completion_ratio": (total_completion_tokens / total_prompt_tokens) if total_prompt_tokens > 0 else 0.0,
    }


def evaluate_answer_with_error_type(
    extracted: str,
    correct_answer: str,
    answer_type: str,
    extraction_successful: bool,
    usage: dict[str, Any],
    max_tokens: int,
) -> tuple[bool, str]:
    """
    Evaluate extracted answer and return (is_correct, error_type).

    Error types:
    - "correct": Answer is correct
    - "max_token_reached": Response was truncated due to max token limit
    - "format_error": Could not extract answer from response (format mismatch)
    - "wrong_answer": Answer was extracted but incorrect
    - "multi_extra_items": Multi-answer has extra incorrect items
    - "multi_missing_items": Multi-answer is missing required items
    - "multi_false_items": Multi-answer has false items instead of correct ones

    Comparison is case-insensitive after whitespace stripping; "multi" answers are
    comma-split and compared as sets, so order never matters. Precedence:
    max_token_reached > format_error > correctness — a truncated response is blamed on the
    token budget even if an answer happened to be extracted.
    """

    # Check for max token reached first (highest priority)
    completion_tokens = usage.get("completion_tokens", 0)
    if completion_tokens >= max_tokens * 0.98:  # 98% threshold to account for slight variations
        return False, "max_token_reached"

    # Check for format error
    if not extraction_successful:
        return False, "format_error"

    # Evaluate answer correctness
    if answer_type == "single":
        is_correct = extracted.lower().strip() == correct_answer.lower().strip()
        if is_correct:
            return True, "correct"
        else:
            return False, "wrong_answer"

    elif answer_type == "multi":
        # Parse comma-separated values
        extracted_set = set(item.strip().lower() for item in extracted.split(",") if item.strip())
        correct_set = set(item.strip().lower() for item in correct_answer.split(",") if item.strip())

        if extracted_set == correct_set:
            return True, "correct"

        # Determine specific multi-answer error type
        extra_items = extracted_set - correct_set
        missing_items = correct_set - extracted_set

        if extra_items and not missing_items:
            return False, "multi_extra_items"
        elif missing_items and not extra_items:
            return False, "multi_missing_items"
        else:
            # Has both extra and missing items, or completely wrong items
            return False, "multi_false_items"

    else:
        # Fallback for unknown answer types
        is_correct = extracted.lower().strip() == correct_answer.lower().strip()
        if is_correct:
            return True, "correct"
        else:
            return False, "wrong_answer"


def run_one_task(
    task: dict[str, Any],
    inferencer: "GatewayInferencer",
    format_example_group: int,
    limiter: "throttle.RateLimiter | None",
    progress: "RunProgress | None" = None,
) -> dict[str, Any]:
    """Worker entry point: run one task end-to-end (format -> call -> extract -> score).

    Runs inside a ThreadPoolExecutor worker — the shared inferencer is read-only here and
    HTTP state is per-thread (see _get_thread_session). Returns the original task dict +
    an ``inference`` block — the record that becomes one line of the results JSONL.
    """
    prompt = format_prompt(task, inferencer.add_context, format_example_group)
    call = inferencer.call_model(prompt, limiter=limiter, session=_get_thread_session(), progress=progress)
    extracted, extraction_successful = extract_answer(call.content)

    # Use answer_type-aware evaluation with error type classification
    answer_type = task.get("answer_type", "single")
    correct, error_type = evaluate_answer_with_error_type(
        extracted, task["correct_answer"], answer_type, extraction_successful, call.usage, inferencer.max_tokens
    )

    # Include all original task fields plus inference information
    result = dict(task)  # Copy all original fields
    result["inference"] = {
        "prompt": prompt,
        "response": call.content,
        "thinking_content": call.thinking_content,
        "thinking_source": call.thinking_source,
        "extracted": extracted,
        "extraction_successful": extraction_successful,
        "is_correct": correct,
        "error_type": error_type,
        "usage": call.usage,
        "raw_message": call.raw_message,
        "provider_meta": call.provider_meta,
        "attempts": call.attempts,
        "latency_ms": call.latency_ms,
        "ttft_ms": call.ttft_ms,
    }
    return result


def build_run_key(model: str, enable_thinking: bool, variant_suffix: str) -> str:
    """The results-filename stem (and runs.run_key): model-safe name + variant suffixes."""
    run_key = model.replace("/", "_").replace(":", "_")
    if enable_thinking:
        run_key += "-thinking"
    return run_key + variant_suffix


def _build_variant_suffix(add_context: bool, format_example_group: int) -> str:
    """Return a short suffix for filenames to distinguish experiment variants."""
    parts = []
    if add_context:
        parts.append("piecearr")
    if format_example_group == 2:
        parts.append("fmt2")
    return ("-" + "-".join(parts)) if parts else ""


class GatewayInferencer:
    """Thin client around the Vercel AI Gateway plus result-file management.

    Holds the run configuration (model, retries, token budget, thinking mode, filename
    suffix) and owns three concerns: calling the API with retries (``call_model``),
    orchestrating sequential/parallel inference (``run_inference``), and incremental
    result persistence (``_save_*``). Credentials come from ``resolve_api_key`` — see
    the module docstring.
    """

    def __init__(
        self,
        model: str,
        add_context: bool = False,
        max_retries: int = 5,
        timeout: int = 60,
        max_tokens: int = 2048,
        enable_thinking: bool = False,
        filename_suffix: str = "",
    ):
        self.model = model
        self.add_context = add_context
        self.max_retries = max_retries
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.enable_thinking = enable_thinking
        # Additional filename suffix to differentiate experiment variants in outputs
        self.filename_suffix = filename_suffix
        self.api_key = resolve_api_key()
        self.url = GATEWAY_URL

    def call_model(
        self,
        prompt: str,
        limiter: "throttle.RateLimiter | None" = None,
        session: requests.Session | None = None,
        progress: RunProgress | None = None,
    ) -> ModelCall:
        """POST one chat completion with rate limiting and classified, logged retries.

        Request notes:
        - EVERY request is STREAMED and reassembled into the exact non-streaming message
          shape: chat completions via ``consume_chat_sse`` (``stream: true`` +
          ``stream_options.include_usage``), the Anthropic-native /v1/messages path via
          ``consume_anthropic_sse`` (its own typed-event SSE grammar). Streaming defeats
          the Vercel AI Gateway's ~340s wire-idle kill on long generations (Incident 2 in
          docs/model-trials/2026-07-12-smoke-campaign.md; the native path joined on
          2026-07-13 after adaptive sonnet-5 chess generations ran 380-400s and burned
          billed wall-kill corpses).
        - ``--enable-thinking`` maps to ``build_reasoning_payload`` (probe-verified; see
          its docstring for the Claude 5 adaptive-thinking limitation).

        Retry policy (see eval/throttle.py): 429/502/503/connection failures retry up to
        ``max_retries`` with Retry-After honored (a 429 penalizes the *shared* limiter so
        all threads pause); read timeouts and 500/504/524 may have been billed upstream,
        so they retry at most ``throttle.EXPENSIVE_MAX_ATTEMPTS`` times total; a stream
        that drops AFTER tokens arrived ("stream_drop") is billed the same way — the
        provider generated output we lost — and shares that cap, while a drop before any
        token remains a cheap connection retry. Permanent 4xx and parse failures never
        retry. Every attempt is recorded in ``ModelCall.attempts`` (mirrored into the
        results JSONL), including per-attempt ttft_ms / stream_chunks / partial character
        counts for streamed attempts, so retry pressure and billing discrepancies are
        diagnosable from our own data.

        After exhausting retries, ``content`` is the literal string "ERROR: ...", which
        resume logic later treats as incomplete.
        """
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

        data = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": self.max_tokens,
        }

        # Adaptive-thinking (Claude 5 family, Opus 4.7+) runs must go through the gateway's Anthropic-native
        # endpoint: adaptive thinking (and its display=summarized traces) cannot be
        # expressed on the OpenAI-compatible endpoint. See build_reasoning_payload.
        use_native_anthropic = self.enable_thinking and anthropic_adaptive_thinking(self.model)
        request_url = self.url

        if self.enable_thinking:
            if use_native_anthropic:
                request_url = ANTHROPIC_NATIVE_URL
                headers["anthropic-version"] = "2023-06-01"
                data["thinking"] = build_reasoning_payload(self.model)
            else:
                data["reasoning"] = build_reasoning_payload(self.model)

        # The 340s-wall fix: stream EVERY request so bytes move on the wire continuously.
        # The native path streamed too as of 2026-07-13 — adaptive sonnet-5 generations
        # ran 380-400s on hard chess tasks and died at the wall, each kill billed
        # upstream (see consume_anthropic_sse docstring). stream_options is an
        # OpenAI-compatible extension; the native endpoint takes bare "stream": true.
        use_streaming = True
        data["stream"] = True
        if not use_native_anthropic:
            data["stream_options"] = {"include_usage": True}

        transport = session if session is not None else requests
        attempts: list[dict[str, Any]] = []

        for attempt_number in range(1, self.max_retries + 1):
            if limiter is not None:
                limiter.acquire()

            attempt = {
                "attempt_no": attempt_number,
                "started_at": datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            }
            attempt_start = time.monotonic()
            response = None
            failure: Exception | None = None
            stream_info: dict[str, Any] | None = None

            try:
                response = transport.post(
                    request_url, headers=headers, data=json.dumps(data), timeout=self.timeout, stream=use_streaming
                )
            except Exception as transport_error:
                failure = transport_error

            attempt["http_status"] = response.status_code if response is not None else None

            # Success path: 2xx that also parses cleanly. For streamed requests "parsing"
            # is draining the SSE stream — that is where a long generation spends its life,
            # so duration_ms is computed after consumption, not after the POST returns.
            if response is not None and 200 <= response.status_code < 300:
                try:
                    if use_native_anthropic:
                        result, stream_info = consume_anthropic_sse(
                            response,
                            attempt_start,
                            on_delta=progress.on_stream_delta if progress is not None else None,
                        )
                        content, thinking_content, thinking_source, usage = parse_native_anthropic_message(result)
                        # Preserve the full native message (thinking blocks + signatures).
                        message = {
                            key: result.get(key)
                            for key in ("role", "content", "stop_reason", "model")
                            if result.get(key) is not None
                        }
                        provider_meta = {key: result.get(key) for key in ("id", "model") if result.get(key)}
                    else:
                        if use_streaming:
                            result, stream_info = consume_chat_sse(
                                response,
                                attempt_start,
                                on_delta=progress.on_stream_delta if progress is not None else None,
                            )
                        else:
                            result = response.json()
                        message = result["choices"][0]["message"]
                        content = (message.get("content") or "").strip()
                        thinking_content, thinking_source = extract_thinking(message)
                        usage = result.get("usage", {}) or {}
                        provider_meta = {
                            key: result.get(key) for key in ("id", "model", "provider", "created") if result.get(key)
                        }
                    duration_ms = int((time.monotonic() - attempt_start) * 1000)
                    attempt["duration_ms"] = duration_ms
                    attempt.update(error_class="ok", billed_risk=1, retry_after_s=None, wait_s=None, outcome="success")
                    if stream_info is not None:
                        attempt.update(stream_info)
                    attempts.append(attempt)
                    return ModelCall(
                        content=content,
                        thinking_content=thinking_content,
                        thinking_source=thinking_source,
                        usage=usage,
                        raw_message=message,
                        provider_meta=provider_meta,
                        attempts=attempts,
                        latency_ms=duration_ms,
                        ttft_ms=stream_info["ttft_ms"] if stream_info is not None else None,
                        ok=True,
                    )
                except Exception as parse_error:
                    # 2xx body we couldn't parse (or a stream that dropped): billed-risk —
                    # classification below decides whether regenerating is allowed.
                    failure = parse_error

            duration_ms = int((time.monotonic() - attempt_start) * 1000)
            attempt["duration_ms"] = duration_ms

            # Failure path: classify, decide, record, maybe sleep.
            if response is not None and not (200 <= response.status_code < 300):
                error_class, retriable, billed_risk = throttle.classify(response, None)
                retry_after_s = throttle.parse_retry_after(response.headers)
                error_detail = f"HTTP {response.status_code}: {response.text[:300]}"
            elif isinstance(failure, StreamDrop):
                # Mid-stream drop with tokens already received: the provider generated
                # (and billed) output we lost — retry under the billed-risk attempt cap.
                # A drop before any token is the old cheap pre-generation failure.
                stream_info = failure.stream_info
                if stream_info["stream_chunks"] > 0:
                    error_class, retriable, billed_risk = "stream_drop", True, True
                else:
                    error_class, retriable, billed_risk = throttle.classify(None, failure.cause)
                retry_after_s = None
                error_detail = (
                    f"{type(failure.cause).__name__} after {stream_info['stream_chunks']} streamed chunks"
                    f" ({stream_info['content_chars']} content + {stream_info['reasoning_chars']} reasoning chars):"
                    f" {failure.cause}"
                )
            else:
                error_class, retriable, billed_risk = throttle.classify(None, failure)
                retry_after_s = None
                body_hint = ""
                if response is not None and error_class == "parse":
                    try:
                        body_hint = f" body={response.text[:200]!r}"
                    except Exception:
                        body_hint = ""
                error_detail = f"{type(failure).__name__}: {failure}{body_hint}"

            if use_streaming and response is not None:
                # Release the pooled connection: an aborted or partially-read stream
                # would otherwise hold its socket.
                with contextlib.suppress(Exception):
                    response.close()

            attempt_budget = throttle.EXPENSIVE_MAX_ATTEMPTS if billed_risk else self.max_retries
            will_retry = retriable and attempt_number < attempt_budget
            wait_s = throttle.compute_backoff(attempt_number, retry_after_s) if will_retry else None
            attempt.update(
                error_class=error_class,
                billed_risk=int(billed_risk),
                retry_after_s=retry_after_s,
                wait_s=wait_s,
                outcome="retried" if will_retry else ("gave_up" if retriable else "fatal"),
            )
            if stream_info is not None:
                # Partial-progress evidence for billing forensics: how much of the
                # generation had streamed before the attempt died.
                attempt.update(stream_info)
            attempts.append(attempt)

            if not will_retry:
                return ModelCall(content=f"ERROR: {error_detail}", attempts=attempts, error=error_detail)

            billed_warning = " [BILLED-RISK: provider may have charged for the failed attempt]" if billed_risk else ""
            tqdm.write(
                f"[retry] attempt {attempt_number}/{attempt_budget} {error_class}"
                f" (status={attempt['http_status']}) sleeping {wait_s:.1f}s{billed_warning}"
            )
            if error_class == "http_429" and limiter is not None:
                # Pause every thread; the next acquire() drains through the bucket rather
                # than stampeding when the window reopens.
                limiter.penalize(wait_s)
            else:
                time.sleep(wait_s)

        # Defensive: the loop always returns from inside.
        return ModelCall(content="ERROR: retry loop exited unexpectedly", attempts=attempts, error="internal")

    def run_inference(
        self,
        tasks: list[dict[str, Any]],
        num_workers: int = 1,
        format_example_group: int = 1,
        output_path: str | None = None,
        save_interval: int = 10,
        existing_results: list[dict[str, Any]] = None,
        save_existing_first: bool = False,
        limiter: "throttle.RateLimiter | None" = None,
        record_hook=None,
    ) -> list[dict[str, Any]]:
        """Run inference on tasks, saving incrementally; returns results in input task order.

        One code path for any worker count: a ThreadPoolExecutor (requests are I/O-bound)
        with results consumed via as_completed and appended to the results JSONL every
        ``save_interval`` completions, in *completion* order — resume keys on task_id and
        the final rewrite in main() restores task order, so mid-file ordering is
        irrelevant and no completion is ever held back waiting for earlier submissions
        (the old multiprocessing in-order barrier caused exactly that). The shared
        ``limiter`` paces request starts across all threads. When resuming,
        ``save_existing_first`` rewrites the file with the already-complete results so
        the appends produce one coherent file.
        """
        if existing_results is None:
            existing_results = []

        # Save existing results first if we're resuming
        if save_existing_first and existing_results and output_path:
            self._save_existing_results(existing_results, output_path)

        print(f"Processing {len(tasks)} tasks with {num_workers} worker threads...")
        results: list[dict[str, Any]] = []
        pending_batch: list[dict[str, Any]] = []
        progress = RunProgress()
        monitor_stop = threading.Event()

        with ThreadPoolExecutor(max_workers=max(1, num_workers)) as executor:
            futures = [
                executor.submit(run_one_task, task, self, format_example_group, limiter, progress) for task in tasks
            ]
            with tqdm(total=len(futures), desc="Processing tasks") as pbar:
                monitor = threading.Thread(target=_progress_monitor, args=(pbar, progress, monitor_stop), daemon=True)
                monitor.start()
                try:
                    for future in as_completed(futures):
                        result = future.result()
                        progress.on_task_done(result["inference"].get("usage") or {})
                        pbar.update(1)
                        results.append(result)
                        pending_batch.append(result)
                        if record_hook is not None:
                            record_hook(result)
                        if output_path and len(pending_batch) >= save_interval:
                            self._save_incremental_results(pending_batch, output_path)
                            pending_batch = []
                finally:
                    monitor_stop.set()
                    monitor.join(timeout=5)

        if output_path and pending_batch:
            self._save_incremental_results(pending_batch, output_path)

        # Return in original task order
        task_id_to_result = {result["task_id"]: result for result in results}
        return [task_id_to_result[task["task_id"]] for task in tasks if task["task_id"] in task_id_to_result]

    def _save_incremental_results(self, new_results: list[dict[str, Any]], output_path: str):
        """Append a batch of new results to ``<output_dir>/<model_safe_name><suffixes>.jsonl``.

        The filename encodes the variant (-thinking / -piecearr / -fmt2), which is why the
        same flags must be passed to resume a run. Failures are warnings, not fatal — the
        full result set is rewritten at the end of main() anyway.
        """
        try:
            output_dir = Path(output_path)
            model_safe_name = build_run_key(self.model, self.enable_thinking, self.filename_suffix)
            results_file = output_dir / f"{model_safe_name}.jsonl"

            # Append new results to JSONL file
            results_file.parent.mkdir(parents=True, exist_ok=True)
            with open(results_file, "a") as jsonl_file:
                for result in new_results:
                    jsonl_file.write(json.dumps(result, ensure_ascii=False) + "\n")

            # Calculate current accuracy for progress info
            total = len(new_results)
            correct = sum(result["inference"]["is_correct"] for result in new_results)
            accuracy = correct / total if total > 0 else 0.0

            print(f"\nIncremental save: {results_file} (+{total} tasks, {accuracy:.3f} accuracy for new tasks)")

        except Exception as error:
            print(f"\nWarning: Failed to save incremental results: {error}")

    def _save_existing_results(self, existing_results: list[dict[str, Any]], output_path: str):
        """Overwrite the results file with previously completed results (resume bootstrap),
        so subsequent incremental appends continue a coherent file."""
        try:
            output_dir = Path(output_path)
            model_safe_name = build_run_key(self.model, self.enable_thinking, self.filename_suffix)
            results_file = output_dir / f"{model_safe_name}.jsonl"

            # Write existing results to file (overwrite)
            results_file.parent.mkdir(parents=True, exist_ok=True)
            with open(results_file, "w") as jsonl_file:
                for result in existing_results:
                    jsonl_file.write(json.dumps(result, ensure_ascii=False) + "\n")

            print(f"\nSaved {len(existing_results)} existing complete results to {results_file}")

        except Exception as error:
            print(f"\nWarning: Failed to save existing results: {error}")


def load_existing_results(results_file: Path) -> dict[str, dict[str, Any]]:
    """Load a results JSONL keyed by task_id (empty dict if the file doesn't exist).
    Duplicate task_ids resolve to the last line, i.e. the most recent attempt."""
    if not results_file.exists():
        return {}

    print(f"Loading existing results from {results_file}")

    existing_results = {}
    with open(results_file) as results_input:
        for line in results_input:
            if line.strip():
                result = json.loads(line.strip())
                task_id = result.get("task_id")
                if task_id:
                    existing_results[task_id] = result

    print(f"Loaded {len(existing_results)} existing results")
    return existing_results


def re_evaluate_results(results: list[dict[str, Any]], max_tokens: int) -> list[dict[str, Any]]:
    """Re-run extraction + scoring over saved responses (the --eval-only path).

    Uses only the stored ``response`` text — no API calls — so improved extraction or
    scoring logic can be re-applied to finished runs for free.
    """
    re_evaluated = []

    for result in results:
        # Make a copy to avoid modifying the original
        new_result = dict(result)

        # Get the response from existing inference
        response = result.get("inference", {}).get("response", "")

        # Re-extract the answer using the updated extraction function
        extracted, extraction_successful = extract_answer(response)

        # Re-evaluate with the correct answer
        answer_type = result.get("answer_type", "single")
        usage = result.get("inference", {}).get("usage", {})

        correct, error_type = evaluate_answer_with_error_type(
            extracted, result["correct_answer"], answer_type, extraction_successful, usage, max_tokens
        )

        # Update the inference results
        new_result["inference"]["extracted"] = extracted
        new_result["inference"]["extraction_successful"] = extraction_successful
        new_result["inference"]["is_correct"] = correct
        new_result["inference"]["error_type"] = error_type

        re_evaluated.append(new_result)

    return re_evaluated


def filter_incomplete_tasks(
    tasks: list[dict[str, Any]], existing_results: dict[str, dict[str, Any]], retry_capped: bool = False
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split tasks into (incomplete -> re-run, complete -> keep) for resume.

    A task counts as complete if its saved result has a non-empty response that isn't an
    "ERROR:..." retry-exhaustion marker, OR is scored ``max_token_reached`` — capped rows
    are recorded experimental outcomes (policy in
    docs/model-trials/2026-07-12-smoke-campaign.md) and count complete even when the
    response text is EMPTY, which is the common shape for thinking models that burn the
    whole token budget on reasoning and never emit an answer (observed 2026-07-13: seven
    deepseek caps with ~90K chars of thinking and response == "" bypassed the cap-keep
    logic through the response-emptiness check and were re-billed on resume).
    ``retry_capped`` (--retry-capped) restores upstream behavior: re-run every capped row.
    """
    incomplete_tasks = []
    complete_results = []
    max_token_retry_count = 0
    max_token_kept_count = 0

    for task in tasks:
        task_id = task.get("task_id")
        if task_id in existing_results:
            existing_result = existing_results[task_id]
            inference = existing_result.get("inference") or {}
            response = inference.get("response") or ""
            is_error_marker = response.startswith("ERROR")
            error_type = inference.get("error_type", "")
            # Check if inference was completed successfully (a capped row counts even
            # with an empty response — the budget went entirely to reasoning)
            if not is_error_marker and (response or error_type == "max_token_reached"):
                if error_type == "max_token_reached" and retry_capped:
                    incomplete_tasks.append(task)
                    max_token_retry_count += 1
                else:
                    if error_type == "max_token_reached":
                        max_token_kept_count += 1
                    complete_results.append(existing_result)
            else:
                incomplete_tasks.append(task)
        else:
            incomplete_tasks.append(task)

    if max_token_retry_count > 0:
        print(f"Found {max_token_retry_count} previous 'max_token_reached' results - retrying them (--retry-capped)")
    if max_token_kept_count > 0:
        print(
            f"Keeping {max_token_kept_count} previous 'max_token_reached' results as recorded outcomes "
            "(pass --retry-capped to re-run them)"
        )

    return incomplete_tasks, complete_results


def setup_results_db(args, tasks: list[dict[str, Any]]):
    """Open the SQLite index and register this run; returns (conn, run_id) or (None, None).

    Shared by normal and --eval-only modes. Honors --no-db. Provenance captured per
    run_key: model/variant flags, the exact reasoning payload that will be sent,
    dataset hash, best-effort git commit, and the full CLI namespace.
    """
    if args.no_db:
        return None, None
    db_path = args.db_path or (args.output_dir / storage.DEFAULT_DB_NAME)
    conn = storage.connect(db_path)
    dataset_hash = storage.compute_dataset_hash(args.dataset_root)
    storage.upsert_tasks(conn, tasks, dataset_hash=dataset_hash)
    run_key = build_run_key(
        args.model,
        args.enable_thinking,
        _build_variant_suffix(args.add_context, args.use_format_example_group),
    )
    run_id = storage.get_or_create_run(
        conn,
        run_key,
        {
            "model": args.model,
            "backend": "vercel-gateway",
            "enable_thinking": args.enable_thinking,
            "add_context": args.add_context,
            "format_example_group": args.use_format_example_group,
            "max_tokens": args.max_tokens,
            "reasoning_config": (build_reasoning_payload(args.model) if args.enable_thinking else None),
            "dataset_hash": dataset_hash,
            "git_commit": storage.git_commit_or_none(Path(__file__).parent.parent),
            "cli_args": vars(args),
        },
    )
    print(f"Results database: {db_path} (run_key={run_key}, run_id={run_id})")
    return conn, run_id


def make_db_record_hook(conn: sqlite3.Connection, run_id: int):
    """Build the per-result SQLite mirror hook (one short transaction per row).

    Commits immediately after every row. WAL allows exactly ONE writer at a time, and
    the previous batched-commit design (every save_interval rows) left the implicit
    write transaction open across the minutes-long idle gaps between task completions —
    starving every other runner process on the same DB (observed 2026-07-12: concurrent
    startups exhausted their jittered lock retries against a hook connection that was
    just *waiting* for its next task). Per-row commits hold the writer lock for
    milliseconds, and mirror volume is at most completions-per-second, so batching
    bought nothing.

    Non-fatal by contract: the JSONL is canonical and the DB is a rebuildable index, so
    a locked database must never kill a paid inference run — roll back (a failed upsert
    must not leave a dangling write transaction holding the lock), warn, and move on.
    """

    def record_to_db(result: dict[str, Any]) -> None:
        try:
            storage.upsert_result(conn, run_id, result)
            conn.commit()
        except sqlite3.OperationalError as exc:
            with contextlib.suppress(sqlite3.Error):
                conn.rollback()
            print(
                f"WARNING: SQLite mirror skipped task {result.get('task_id')}: {exc}. "
                "The row is safe in the results JSONL; rebuild the DB with "
                "`python eval/storage.py ingest results/*.jsonl --dataset-root benchmark`.",
                file=sys.stderr,
            )

    return record_to_db


def main():
    """CLI entry: run (or resume, or re-evaluate) a full benchmark pass and report stats.

    Three modes: --eval-only (rescore an existing results file, no API), resume (default —
    skip tasks already completed in the matching results file), and --no-resume (fresh run).
    All modes end identically: rewrite the results JSONL + pretty JSON, compute per-type /
    per-category / error-type stats into <name>_stats.json, and print the report tables.
    """
    load_env_file()
    args = parse_arguments()

    # Set num_workers for metadata (used in eval-only mode too)
    num_workers = args.workers

    # SQLite handle; stays None in eval-only mode (DB sync for rescoring lands separately)
    # and under --no-db.
    conn = None
    run_id = None

    # Check for eval-only mode first
    if args.eval_only:
        print("Running in EVAL-ONLY mode - re-evaluating existing results")

        # Get the results file path (include variant suffix to match the run)
        model_safe_name = build_run_key(
            args.model,
            args.enable_thinking,
            _build_variant_suffix(args.add_context, args.use_format_example_group),
        )
        results_file = args.output_dir / f"{model_safe_name}.jsonl"

        # Check if results file exists
        if not results_file.exists():
            print(f"ERROR: Results file does not exist: {results_file}")
            print("Eval-only mode requires existing results. Please run inference first.")
            return

        # Load existing results
        print(f"Loading results from {results_file}")
        results = []
        with open(results_file) as results_input:
            for line in results_input:
                if line.strip():
                    results.append(json.loads(line.strip()))

        print(f"Loaded {len(results)} results")

        # Re-evaluate all results
        print("Re-extracting answers and re-evaluating...")
        results = re_evaluate_results(results, args.max_tokens)

        # Load tasks to ensure we have complete task information
        print(f"Loading tasks from dataset root: {args.dataset_root}")
        tasks = load_tasks(args.dataset_root, args.max_tasks, args.N_samples_per_task)

        # Make sure task order is maintained
        task_id_to_task = {task["task_id"]: task for task in tasks}
        task_id_to_result = {result["task_id"]: result for result in results}

        # Ensure all task fields are present in results
        for result in results:
            task_id = result["task_id"]
            if task_id in task_id_to_task:
                task = task_id_to_task[task_id]
                # Update any missing fields from original task
                for field_name, field_value in task.items():
                    if field_name not in result:
                        result[field_name] = field_value

        # Save the re-evaluated results back to the JSONL file
        print(f"Saving re-evaluated results to {results_file}")
        with open(results_file, "w") as results_output:
            for result in results:
                results_output.write(json.dumps(result, ensure_ascii=False) + "\n")

        # Rescoring changes is_correct/error_type — keep the DB in step with the JSONL
        # (the common final-sync block upserts every row and re-finalizes with new stats).
        conn, run_id = setup_results_db(args, tasks)

        # Use dummy timing for eval-only mode
        start_time = time.time()
        end_time = time.time()

    else:
        # Normal mode (with or without resume)
        if num_workers > 1:
            print(f"Using {num_workers} workers for parallel processing")

        # Load and run
        print(f"Loading tasks from dataset root: {args.dataset_root}")
        if args.N_samples_per_task:
            print(f"Sampling {args.N_samples_per_task} tasks per task type")
        tasks = load_tasks(args.dataset_root, args.max_tasks, args.N_samples_per_task)
        print(f"Final task count: {len(tasks)}")

        # Check for resume mode
        if args.no_resume:
            print("Starting inference from scratch (--no-resume)")
            incomplete_tasks = tasks
            complete_results = []
        else:
            print("Checking for existing results to resume from...")
            # Check for existing results and filter incomplete tasks
            model_safe_name = build_run_key(
                args.model,
                args.enable_thinking,
                _build_variant_suffix(args.add_context, args.use_format_example_group),
            )
            results_file = args.output_dir / f"{model_safe_name}.jsonl"
            existing_results = load_existing_results(results_file)

            incomplete_tasks, complete_results = filter_incomplete_tasks(tasks, existing_results, args.retry_capped)

            print(f"Tasks already completed: {len(complete_results)}")
            print(f"Tasks needing inference: {len(incomplete_tasks)}")

        # SQLite results index (the JSONL stays canonical; the DB is derived + rebuildable).
        variant_suffix = _build_variant_suffix(args.add_context, args.use_format_example_group)
        conn, run_id = setup_results_db(args, tasks)

        if incomplete_tasks:
            inferencer = GatewayInferencer(
                args.model,
                args.add_context,
                args.max_retries,
                args.timeout,
                args.max_tokens,
                args.enable_thinking,
                filename_suffix=variant_suffix,
            )

            start_time = time.time()
            # For resume mode, save existing results first, then append new ones
            # For no-resume mode, just append new results
            save_existing_first = not args.no_resume and len(complete_results) > 0
            limiter = throttle.RateLimiter(requests_per_second=args.rps, burst=args.burst)

            record_hook = None
            if conn is not None:
                record_hook = make_db_record_hook(conn, run_id)

            try:
                new_results = inferencer.run_inference(
                    incomplete_tasks,
                    num_workers,
                    args.use_format_example_group,
                    str(args.output_dir),
                    args.save_interval,
                    complete_results,
                    save_existing_first,
                    limiter=limiter,
                    record_hook=record_hook,
                )
            except KeyboardInterrupt:
                # Partial rows are already on disk (JSONL appends + DB mirroring); mark the
                # run aborted so the DB is honest about incompleteness, then let the
                # interrupt propagate. Resuming with the same flags picks up cleanly.
                if conn is not None:
                    try:
                        conn.commit()
                        storage.finalize_run(conn, run_id, None, status="aborted")
                    except sqlite3.OperationalError as exc:
                        print(f"WARNING: could not mark DB run aborted: {exc}", file=sys.stderr)
                    conn.close()
                print("\nInterrupted — completed rows are saved; DB run marked 'aborted'. Resume with the same flags.")
                raise
            if conn is not None:
                try:
                    conn.commit()
                except sqlite3.OperationalError as exc:
                    print(f"WARNING: SQLite mid-run commit failed: {exc}. Continuing; the JSONL is canonical.",
                          file=sys.stderr)
            end_time = time.time()

            # A thinking run in which no task returned any trace almost certainly means the
            # reasoning parameter silently no-opped for this model (e.g. Claude 5 adaptive
            # thinking through the OpenAI-compatible endpoint) — flag it loudly rather than
            # letting a mislabeled '-thinking' results file masquerade as a thinking run.
            if args.enable_thinking and new_results:
                traceless = sum(
                    1 for result in new_results if result["inference"].get("thinking_source") in (None, "", "none")
                )
                if traceless == len(new_results):
                    print(
                        "\nWARNING: --enable-thinking was set but 0/"
                        f"{len(new_results)} responses contained any thinking trace "
                        "(thinking_source == 'none' everywhere). The reasoning parameter was "
                        "likely ignored for this model; see build_reasoning_payload docstring."
                    )

            # Combine complete and new results, maintaining original task order
            task_id_to_result = {}
            for result in complete_results + new_results:
                task_id_to_result[result["task_id"]] = result

            # Maintain original task order
            results = [task_id_to_result[task["task_id"]] for task in tasks if task["task_id"] in task_id_to_result]
        else:
            print("All tasks already completed!")
            start_time = time.time()
            results = complete_results
            end_time = time.time()

    # Calculate stats
    total = len(results)
    correct = sum(result["inference"]["is_correct"] for result in results)
    accuracy = correct / total

    # Calculate usage statistics
    usage_stats = calculate_total_usage(results)

    # Calculate error type distribution
    error_type_stats = {}
    for result in results:
        error_type = result["inference"].get("error_type", "unknown")
        if error_type not in error_type_stats:
            error_type_stats[error_type] = 0
        error_type_stats[error_type] += 1

    # Convert to rates
    error_type_rates = {}
    for error_type, count in error_type_stats.items():
        error_type_rates[error_type] = count / total if total > 0 else 0

    # Calculate per-task-type and per-task-category stats
    task_type_stats = {}
    task_category_stats = {}

    for result_index, result in enumerate(results):
        task_type = result["task_type"]
        # Extract task_category from the original task
        task_category = tasks[result_index].get("task_category", "unknown")
        usage = result.get("inference", {}).get("usage", {})

        # Task type stats
        if task_type not in task_type_stats:
            task_type_stats[task_type] = {"total": 0, "correct": 0, "cost": 0.0, "tokens": 0, "extraction_success": 0}
        task_type_stats[task_type]["total"] += 1
        task_type_stats[task_type]["cost"] += usage.get("cost", 0.0)
        task_type_stats[task_type]["tokens"] += usage.get("total_tokens", 0)
        if result["inference"]["is_correct"]:
            task_type_stats[task_type]["correct"] += 1
        if result["inference"]["extraction_successful"]:
            task_type_stats[task_type]["extraction_success"] += 1

        # Task category stats
        if task_category not in task_category_stats:
            task_category_stats[task_category] = {
                "total": 0,
                "correct": 0,
                "cost": 0.0,
                "tokens": 0,
                "extraction_success": 0,
            }
        task_category_stats[task_category]["total"] += 1
        task_category_stats[task_category]["cost"] += usage.get("cost", 0.0)
        task_category_stats[task_category]["tokens"] += usage.get("total_tokens", 0)
        if result["inference"]["is_correct"]:
            task_category_stats[task_category]["correct"] += 1
        if result["inference"]["extraction_successful"]:
            task_category_stats[task_category]["extraction_success"] += 1

    # Add accuracy, cost, and extraction success calculations to task type stats
    for _task_type, stats in task_type_stats.items():
        stats["accuracy"] = stats["correct"] / stats["total"] if stats["total"] > 0 else 0.0
        stats["extraction_success_rate"] = stats["extraction_success"] / stats["total"] if stats["total"] > 0 else 0.0
        stats["avg_cost"] = stats["cost"] / stats["total"] if stats["total"] > 0 else 0.0
        stats["avg_tokens"] = stats["tokens"] / stats["total"] if stats["total"] > 0 else 0.0

    # Add accuracy, cost, and extraction success calculations to task category stats
    for _task_category, stats in task_category_stats.items():
        stats["accuracy"] = stats["correct"] / stats["total"] if stats["total"] > 0 else 0.0
        stats["extraction_success_rate"] = stats["extraction_success"] / stats["total"] if stats["total"] > 0 else 0.0
        stats["avg_cost"] = stats["cost"] / stats["total"] if stats["total"] > 0 else 0.0
        stats["avg_tokens"] = stats["tokens"] / stats["total"] if stats["total"] > 0 else 0.0

    # Create model-based filenames with variant suffix
    model_safe_name = build_run_key(
        args.model,
        args.enable_thinking,
        _build_variant_suffix(args.add_context, args.use_format_example_group),
    )
    results_file = args.output_dir / f"{model_safe_name}.jsonl"
    stats_file = args.output_dir / f"{model_safe_name}_stats.json"

    # Save results as JSONL (one result per line)
    results_file.parent.mkdir(parents=True, exist_ok=True)
    with open(results_file, "w") as results_output:
        for result in results:
            results_output.write(json.dumps(result, ensure_ascii=False) + "\n")

    # Save stats as separate JSON file (the same object lands in runs.stats below)
    stats_payload = {
        "model": args.model,
        "accuracy": accuracy,
        "format_correct_rate": usage_stats["extraction_success_rate"],
        "total": total,
        "correct": correct,
        "format_correct": usage_stats["extraction_success_count"],
        "error_type_stats": error_type_stats,
        "error_type_rates": error_type_rates,
        "time": end_time - start_time,
        "usage": usage_stats,
        "task_type_stats": task_type_stats,
        "task_category_stats": task_category_stats,
        "metadata": {
            "dataset_root": str(args.dataset_root),
            "n_samples_per_task": args.N_samples_per_task,
            "max_tasks": args.max_tasks,
            "add_context": args.add_context,
            "workers": num_workers,
            "format_example_group": args.use_format_example_group,
            "enable_thinking": args.enable_thinking,
            "backend": "vercel-gateway",
            "rps": args.rps,
            "burst": args.burst,
        },
    }
    with open(stats_file, "w") as stats_output:
        json.dump(stats_payload, stats_output, indent=2)

    # Final DB sync: upsert every row (idempotent — catches rows completed in previous
    # resumed invocations that this invocation didn't re-run), then finalize the run.
    # Non-fatal: JSONL + stats are already on disk; the DB can always be rebuilt.
    if conn is not None:
        try:
            for result_number, result in enumerate(results, 1):
                storage.upsert_result(conn, run_id, result)
                # Bound how long the single WAL writer slot is held: a full-run sync is
                # thousands of rows, and sibling runner processes write on completions.
                if result_number % 200 == 0:
                    conn.commit()
            storage.finalize_run(conn, run_id, stats_payload, status="complete")
        except sqlite3.OperationalError as exc:
            print(
                f"WARNING: final SQLite sync failed: {exc}. Results JSONL and stats are saved; "
                "rebuild the DB with `python eval/storage.py ingest results/*.jsonl --dataset-root benchmark`.",
                file=sys.stderr,
            )
        conn.close()

    # Print per-task-category stats with cost and extraction success information
    print("\nPER-TASK-CATEGORY PERFORMANCE:")
    print("-" * 120)
    print(
        f"{'Category':30} {'Accuracy':>10} {'Format Rate':>12} {'Count':>12} {'Cost':>10} {'Avg Cost':>12} {'Tokens':>10}"
    )
    print("-" * 120)
    for task_category, stats in sorted(task_category_stats.items()):
        category_accuracy = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
        extraction_rate = stats["extraction_success"] / stats["total"] if stats["total"] > 0 else 0
        print(
            f"{task_category:30} {category_accuracy:10.3f} {extraction_rate:12.3f} {stats['correct']:>5}/{stats['total']:<5} "
            f"${stats['cost']:>8.4f} ${stats['avg_cost']:>10.4f} {int(stats['tokens']):>10,}"
        )

    # Print per-task-type stats with cost and extraction success information
    print("\nPER-TASK-TYPE PERFORMANCE:")
    print("-" * 120)
    print(
        f"{'Task Type':30} {'Accuracy':>10} {'Format Rate':>12} {'Count':>12} {'Cost':>10} {'Avg Cost':>12} {'Tokens':>10}"
    )
    print("-" * 120)
    for task_type, stats in sorted(task_type_stats.items()):
        type_accuracy = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
        extraction_rate = stats["extraction_success"] / stats["total"] if stats["total"] > 0 else 0
        print(
            f"{task_type:30} {type_accuracy:10.3f} {extraction_rate:12.3f} {stats['correct']:>5}/{stats['total']:<5} "
            f"${stats['cost']:>8.4f} ${stats['avg_cost']:>10.4f} {int(stats['tokens']):>10,}"
        )

    print(f"\nOVERALL ACCURACY: {accuracy:.3f} ({correct}/{total})")
    print(
        f"FORMAT FOLLOWING RATE: {usage_stats['extraction_success_rate']:.3f} ({usage_stats['extraction_success_count']}/{total})"
    )

    print("\nERROR TYPE BREAKDOWN:")
    for error_type, count in sorted(error_type_stats.items()):
        rate = error_type_rates[error_type]
        print(f"  {error_type}: {rate:.3f} ({count}/{total})")

    print(f"\nTIME: {end_time - start_time:.1f}s")
    print("\nUSAGE STATISTICS:")
    print(f"Total Cost: ${usage_stats['total_cost']:.4f}")
    print(f"Average Cost per Task: ${usage_stats['avg_cost_per_task']:.4f}")
    print(f"Cost per Correct Answer: ${usage_stats['total_cost'] / correct if correct > 0 else 0:.4f}")
    print(f"Total Tokens: {usage_stats['total_tokens']:,}")
    print(f"  - Prompt Tokens: {usage_stats['total_prompt_tokens']:,}")
    print(f"  - Completion Tokens: {usage_stats['total_completion_tokens']:,}")
    print(f"Average Tokens per Task: {usage_stats['avg_tokens_per_task']:.1f}")
    print(
        f"Cost per 1K Tokens: ${usage_stats['total_cost'] / (usage_stats['total_tokens'] / 1000) if usage_stats['total_tokens'] > 0 else 0:.4f}"
    )
    print(
        f"Tokens per Dollar: {usage_stats['total_tokens'] / usage_stats['total_cost'] if usage_stats['total_cost'] > 0 else 0:.0f}"
    )
    if usage_stats["error_count"] > 0:
        print(f"API Errors: {usage_stats['error_count']}")
    print(f"\nRESULTS SAVED: {results_file}")
    print(f"STATS SAVED: {stats_file}")


def parse_arguments():
    """Parse CLI flags.

    Caveats (upstream defaults kept as-is):
    - Path defaults resolve two directories above this script (upstream's layout), which
      lands *outside* this repo — always pass --dataset-root benchmark --output-dir results.
    """
    parser = argparse.ArgumentParser()

    script_dir = Path(__file__).parent
    default_dataset_root = script_dir.parent.parent / "data" / "benchmark"
    default_output_dir = script_dir.parent.parent / "results"

    parser.add_argument(
        "--dataset-root", type=Path, default=default_dataset_root, help="Root directory containing JSONL task files"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="anthropic/claude-sonnet-4.5",
        help="Gateway model slug, e.g. anthropic/claude-haiku-4.5 (canonical fleet: docs/model-fleet.md)",
    )
    parser.add_argument("--output-dir", type=Path, default=default_output_dir, help="Directory to save results")
    parser.add_argument("--max-tasks", type=int, default=None)
    parser.add_argument(
        "--N-samples-per-task",
        type=int,
        default=None,
        help="Number of samples to run per task type (default: None, use all)",
    )
    parser.add_argument("--add-context", action="store_true")
    parser.add_argument(
        "--workers",
        type=int,
        default=24,
        help="Worker threads covering in-flight requests (default: 24). Throughput is governed "
        "by --rps/--burst, not workers — raising this beyond the rate ceiling adds nothing.",
    )
    parser.add_argument(
        "--rps",
        type=float,
        default=2.0,
        help="Rate limit: request starts per second shared across all workers (default: 2.0)",
    )
    parser.add_argument(
        "--burst",
        type=int,
        default=4,
        help="Rate limit: token-bucket burst size (default: 4)",
    )
    parser.add_argument("--save-interval", type=int, default=10, help="Save results every N tasks (default: 10)")
    parser.add_argument(
        "--max-retries",
        type=int,
        default=4,
        help="Maximum attempts for cheap-retriable failures (default: 4); failures that may "
        "already be billed (timeouts, 500/504/524) are capped at 2 attempts regardless",
    )
    parser.add_argument("--timeout", type=int, default=6000, help="Request timeout in seconds (default: 6000)")
    parser.add_argument("--max-tokens", type=int, default=4096 * 2)
    parser.add_argument(
        "--use-format-example-group",
        type=int,
        choices=[1, 2],
        default=1,
        help="Which format example group to use (1 or 2, default: 1)",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Start inference from scratch, ignore existing results (default: resume from existing)",
    )
    parser.add_argument(
        "--retry-capped",
        action="store_true",
        help="On resume, re-run tasks whose previous result was 'max_token_reached' (upstream's original "
        "behavior). Default: keep them as recorded experimental outcomes without re-billing.",
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        default=None,
        help=f"SQLite results database path (default: <output-dir>/{storage.DEFAULT_DB_NAME})",
    )
    parser.add_argument(
        "--no-db",
        action="store_true",
        help="Skip the SQLite results index entirely (JSONL outputs are unaffected)",
    )
    parser.add_argument(
        "--enable-thinking", action="store_true", help="Enable reasoning/thinking mode for models that support it"
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Re-evaluate existing results without running inference (regenerates stats and pretty JSON)",
    )

    return parser.parse_args()


if __name__ == "__main__":
    main()
