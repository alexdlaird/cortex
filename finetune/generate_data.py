__copyright__ = "Copyright (c) 2026 Alex Laird"
__license__ = "MIT"

import argparse
import json
import logging
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))

import subprocess

import requests
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue

from config import (
    COLLECTION_NAME,
    FINETUNE_DATA_DIR,
    OLLAMA_BASE_URL,
    OLLAMA_CHAT_MODEL,
    QDRANT_URL,
    SEED_DATA_PATHS,
    TRAINING_SYSTEM_PROMPT,
)
from run_helper import banner, follow

logger = logging.getLogger(__name__)

GENERATION_PROMPT = (
    "You are a coding assistant named cortex. Given a code or documentation chunk, "
    "generate ONE clear technical question a developer might ask about it and a "
    "thorough, direct answer. Respond ONLY with valid JSON: "
    "{\"question\": \"...\", \"answer\": \"...\"}"
)

REQUEST_TIMEOUT = 180
MAX_ATTEMPTS = 3
CONSECUTIVE_FAILURE_LIMIT = 10
PROGRESS_EVERY = 50


def scroll_chunks(client, collection, sample_every, include_tests=False):
    offset = None
    index = 0
    scroll_filter = None if include_tests else Filter(
        must=[FieldCondition(key="chunk_type", match=MatchValue(value="source"))]
    )
    while True:
        points, offset = client.scroll(
            collection_name=collection,
            limit=100,
            offset=offset,
            with_payload=True,
            with_vectors=False,
            scroll_filter=scroll_filter,
        )
        for point in points:
            if index % sample_every == 0:
                node_data = json.loads(point.payload.get("_node_content", "{}"))
                text = node_data.get("text", "").strip()
                if text:
                    yield text
            index += 1
        if offset is None:
            break


def load_seed_pairs(seed_path):
    pairs = []
    if not seed_path or not seed_path.exists():
        return pairs
    with open(seed_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                pairs.append(json.loads(line))
            except Exception:
                pass
    return pairs


def inject_system_prompt(record):
    convs = [c for c in record.get("conversations", []) if c.get("from") != "system"]
    return {"conversations": [{"from": "system", "value": TRAINING_SYSTEM_PROMPT}] + convs}


def _parse_json_response(content):
    content = content.removeprefix("```json").removesuffix("```").strip()
    try:
        return json.loads(re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', content))
    except json.JSONDecodeError:
        pass
    fields = {}
    for key in ("question", "thinking", "answer"):
        m = re.search(rf'"{key}"\s*:\s*"((?:[^"\\]|\\.)*)"', content, re.DOTALL)
        if m:
            fields[key] = m.group(1)
    return fields if fields else None


def _post_chat(ollama_base_url, model, chunk_text):
    payload = {
        "model": model,
        "stream": False,
        "messages": [
            {"role": "system", "content": GENERATION_PROMPT},
            {"role": "user", "content": chunk_text},
        ],
    }
    last_error = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            response = requests.post(f"{ollama_base_url}/api/chat", json=payload, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            return response
        except requests.HTTPError as e:
            if e.response.status_code < 500 or attempt == MAX_ATTEMPTS:
                raise
            last_error = e
        except (requests.Timeout, requests.ConnectionError) as e:
            if attempt == MAX_ATTEMPTS:
                raise
            last_error = e
        logger.warning(f"Ollama request failed ({last_error.__class__.__name__}), retrying (attempt {attempt}/{MAX_ATTEMPTS})")


def generate_pair(chunk_text, ollama_base_url, model):
    try:
        response = _post_chat(ollama_base_url, model, chunk_text)
        content = response.json()["message"]["content"].strip()
        pair = _parse_json_response(content)
        if not pair:
            raise ValueError("Could not extract fields from response")

        question = pair.get("question", "").strip()
        answer = pair.get("answer", "").strip()
        if not question or not answer:
            return None

        return question, answer
    except Exception as e:
        logger.warning(f"Skipping chunk — generation failed: {e}")
    return None


def to_sharegpt(question, answer):
    return {"conversations": [
        {"from": "system", "value": TRAINING_SYSTEM_PROMPT},
        {"from": "human", "value": question},
        {"from": "gpt", "value": answer},
    ]}


def _format_eta(seconds):
    if seconds <= 0:
        return "0:00:00"
    hours, rem = divmod(int(seconds), 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}"


def _count_total_chunks(client, collection, sample_every, limit, include_tests):
    scroll_filter = None if include_tests else Filter(
        must=[FieldCondition(key="chunk_type", match=MatchValue(value="source"))]
    )
    total = client.count(collection_name=collection, count_filter=scroll_filter).count
    sampled = max(total // max(sample_every, 1), 1)
    if limit:
        sampled = min(sampled, limit)
    return sampled


def generate_data(collection, limit, sample_every, output_path, ollama_base_url, model, append, seed_path, include_tests=False):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    is_new_file = mode == "w" or not output_path.exists() or output_path.stat().st_size == 0

    seen_questions = set()
    if not is_new_file:
        with open(output_path) as f:
            for line in f:
                try:
                    record = json.loads(line)
                    q = next((c["value"] for c in record["conversations"] if c["from"] == "human"), None)
                    if q:
                        seen_questions.add(q)
                except Exception:
                    pass
        logger.info(f"Resuming from {output_path}: {len(seen_questions)} existing questions loaded.")

    client = QdrantClient(url=QDRANT_URL)
    sampled_chunks = _count_total_chunks(client, collection, sample_every, limit, include_tests)
    logger.info(f"Will process up to {sampled_chunks:,} chunks from collection '{collection}'.")

    total, skipped, written = 0, 0, 0
    consecutive_failures = 0
    start = time.monotonic()

    with open(output_path, mode) as out:
        if is_new_file:
            for path in (seed_path if isinstance(seed_path, list) else [seed_path]):
                seed_pairs = load_seed_pairs(path)
                for record in seed_pairs:
                    q = next((c["value"] for c in record.get("conversations", []) if c["from"] == "human"), None)
                    if q:
                        seen_questions.add(q)
                    out.write(json.dumps(inject_system_prompt(record)) + "\n")
                if seed_pairs:
                    logger.info(f"Prepended {len(seed_pairs)} seed pairs from {path}")

        for chunk_text in scroll_chunks(client, collection, sample_every, include_tests=include_tests):
            if limit and total >= limit:
                break
            total += 1

            result = generate_pair(chunk_text, ollama_base_url, model)
            if result is None:
                skipped += 1
                consecutive_failures += 1
                if consecutive_failures >= CONSECUTIVE_FAILURE_LIMIT:
                    raise SystemExit(
                        f"Aborting — {consecutive_failures} consecutive chunks failed. "
                        f"Check Ollama health and review logs. "
                        f"Partial output saved to {output_path} ({written} pairs written, {skipped} skipped)."
                    )
                continue

            consecutive_failures = 0

            question, answer = result
            if question in seen_questions:
                skipped += 1
                continue

            seen_questions.add(question)
            out.write(json.dumps(to_sharegpt(question, answer)) + "\n")
            out.flush()
            written += 1

            if written % PROGRESS_EVERY == 0:
                elapsed = time.monotonic() - start
                rate = total / elapsed if elapsed > 0 else 0
                remaining = max(sampled_chunks - total, 0)
                eta = remaining / rate if rate > 0 else 0
                pct = 100 * total / sampled_chunks if sampled_chunks else 0
                logger.info(
                    f"  {written} pairs written ({skipped} skipped) — "
                    f"{pct:.1f}% ({total:,}/{sampled_chunks:,}), "
                    f"{rate:.2f} chunks/s, ETA {_format_eta(eta)}"
                )

    logger.info(f"Done — {written} pairs written to {output_path} ({skipped} skipped out of {total} chunks)")


def prompt_run_config(collection, qdrant_url, include_tests=False):
    """Interactively prompt the user for validation vs full run and sample-every size."""
    client = QdrantClient(url=qdrant_url)
    if include_tests:
        total_points = client.count(collection_name=collection).count
    else:
        total_points = client.count(
            collection_name=collection,
            count_filter=Filter(must=[FieldCondition(key="chunk_type", match=MatchValue(value="source"))]),
        ).count

    print(f"\nQdrant collection '{collection}' has {total_points:,} chunks.")
    print("\nRun a validation set first? [Y/n] ", end="", flush=True)
    val = input().strip().lower()
    if val in ("", "y"):
        return 10, 200

    secs_per_chunk = 3
    print(f"\nEstimated times at ~{secs_per_chunk}s/chunk:")
    for n in [1, 10, 25, 50, 100, 200]:
        chunks = total_points // n
        hours = (chunks * secs_per_chunk) / 3600
        print(f"  --sample-every {n:>3}  →  ~{chunks:>7,} chunks  (~{hours:.1f}h)")

    print(f"\nEnter --sample-every value (or 1 for full set): ", end="", flush=True)
    sample_every = int(input().strip() or "50")
    return sample_every, 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser(description="Generate ShareGPT training data from RAG corpus.")
    parser.add_argument("--limit", type=int, default=0, help="Max chunks to process (0 = all)")
    parser.add_argument("--sample-every", type=int, default=None, help="Take every Nth chunk (skips interactive prompt)")
    parser.add_argument("--output", type=Path, default=None, help="Output JSONL path")
    parser.add_argument("--collection", default=COLLECTION_NAME, help="Qdrant collection name")
    parser.add_argument("--ollama-model", default=OLLAMA_CHAT_MODEL, help="Ollama model for generation")
    parser.add_argument("--append", action="store_true", help="Append to existing output instead of overwriting")
    parser.add_argument("--include-tests", action="store_true", help="Include test file chunks in training data")
    parser.add_argument("--bg", action="store_true", help="Run in background after prompts (internal use)")
    args = parser.parse_args()

    output_path = args.output or (FINETUNE_DATA_DIR / "sharegpt.jsonl")

    if args.sample_every is not None:
        sample_every = args.sample_every
        limit = args.limit
    else:
        sample_every, limit = prompt_run_config(args.collection, QDRANT_URL, include_tests=args.include_tests)

    if not args.bg:
        log_path = Path("logs") / "generate-data.log"
        log_path.parent.mkdir(exist_ok=True)
        cmd = [
            sys.executable, __file__,
            "--sample-every", str(sample_every),
            "--collection", args.collection,
            "--ollama-model", args.ollama_model,
            "--output", str(output_path),
            "--bg",
        ]
        if limit:
            cmd += ["--limit", str(limit)]
        if args.append:
            cmd += ["--append"]
        if args.include_tests:
            cmd += ["--include-tests"]
        with open(log_path, "w") as log_file:
            proc = subprocess.Popen(cmd, stdout=log_file, stderr=log_file, start_new_session=True)
        follow(proc, log_path, "generate-data")
    else:
        banner("GENERATE-DATA — STARTING")
        generate_data(
            collection=args.collection,
            limit=limit,
            sample_every=sample_every,
            output_path=output_path,
            ollama_base_url=OLLAMA_BASE_URL,
            model=args.ollama_model,
            append=args.append,
            seed_path=SEED_DATA_PATHS,
            include_tests=args.include_tests,
        )
        banner("GENERATE-DATA — DONE")
