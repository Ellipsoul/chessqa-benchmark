"""One-shot probe: which reasoning payload shape actually enables extended thinking?

Background: the paper's harness sent ``reasoning: {"effort": "medium"}``. Through
the Vercel AI Gateway that shape yielded ZERO reasoning tokens across a full 50-task
Anthropic run — the gateway maps a token *budget* to Anthropic's thinking, and effort-only
appears to be ignored. This script tests candidate shapes with one tiny request each
(~cents total) and reports which produce real thinking, so the winning shape can be
codified in ``build_reasoning_payload`` with evidence.

Usage:
    python eval/probe_reasoning.py [--model anthropic/claude-haiku-4.5] [--max-tokens 8192]
"""

import argparse
import json

import requests

from run_benchmark import GATEWAY_URL, extract_thinking, load_env_file, resolve_api_key

CANDIDATE_PAYLOADS = [
    ("effort-only (current, believed broken)", {"effort": "medium"}),
    ("enabled-only", {"enabled": True}),
    ("enabled + effort", {"enabled": True, "effort": "medium"}),
    ("budget (max_tokens=4096)", {"max_tokens": 4096}),
]

PROBE_PROMPT = (
    "You are given a chess position in FEN: r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3.\n"
    "Is f7 currently defended adequately? Think carefully, then answer in one sentence.\n"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="anthropic/claude-haiku-4.5")
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--timeout", type=int, default=300)
    args = parser.parse_args()

    load_env_file()
    api_key = resolve_api_key()
    url = GATEWAY_URL
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    print(f"Probing {args.model} via {url}\n")
    print(f"{'payload':40} {'reasoning_toks':>14} {'thinking_source':>16} {'detail types'}")
    print("-" * 110)

    for label, reasoning_payload in CANDIDATE_PAYLOADS:
        body = {
            "model": args.model,
            "messages": [{"role": "user", "content": PROBE_PROMPT}],
            "max_tokens": args.max_tokens,
            "reasoning": reasoning_payload,
        }
        try:
            response = requests.post(url, headers=headers, data=json.dumps(body), timeout=args.timeout)
            response.raise_for_status()
            result = response.json()
            message = result["choices"][0]["message"]
            usage = result.get("usage", {})
            reasoning_tokens = usage.get("completion_tokens_details", {}).get("reasoning_tokens", 0)
            _, thinking_source = extract_thinking(message)
            detail_types = sorted(
                {d.get("type", "?") for d in (message.get("reasoning_details") or []) if isinstance(d, dict)}
            )
            print(f"{label:40} {reasoning_tokens:>14} {thinking_source:>16} {detail_types}")
        except Exception as error:
            print(f"{label:40} FAILED: {type(error).__name__}: {str(error)[:80]}")

    print("\nAcceptance: a payload with reasoning_tokens > 0 and thinking_source == 'full_text'.")


if __name__ == "__main__":
    main()
