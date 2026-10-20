import argparse
import ast
import collections
import dataclasses
import datetime
import hashlib
import json
import random
import re
import shlex
import sys
import unicodedata
from collections.abc import Iterator
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download, list_repo_files, snapshot_download

from proprioception.families import FAMILIES, MODEL_ROOT

MAX_PROMPT_WORDS = 224
MIN_PROMPT_WORDS = 2
MAX_PROMPT_CHARS = 2048
ASCII_SHARE = 0.5
NGRAM = 13
TRANSCRIPT = re.compile(r"^\s*(User|Assistant|System|Human)\s*:|(User|Assistant)\s*:\s*$", re.IGNORECASE)
PERSONA_TASK = re.compile(
    r"^(A|An|As a)\s+[a-z].{0,200}?\b(is|decides|wants|owns|runs|plans)\b"
    r"(?:.*?\s1\.\s.*?\s2\.\s|.*?[Ss]ub-?[Pp]roblem)",
    re.DOTALL,
)


@dataclasses.dataclass(frozen=True)
class Source:
    repo: str
    prefix: str
    suffix: str
    column: str


SOURCES: dict[str, Source] = {
    "antidoom": Source("LiquidAI/antidoom-mix-v1.0", "train", ".jsonl", "conversations"),
    "hermes3": Source("NousResearch/Hermes-3-Dataset", "hermes-3", ".jsonl", "conversations"),
    "smoltalk2": Source("HuggingFaceTB/smoltalk2", "SFT/", ".parquet", "messages"),
    "magpie-ultra": Source("argilla/magpie-ultra-v1.0", "top_300k", ".parquet", "messages"),
    "tulu-3": Source("allenai/tulu-3-sft-mixture", "data/", ".parquet", "messages"),
    "wildchat": Source("allenai/WildChat-4.8M", "data/", ".parquet", "conversation"),
}


CONTROL_STRINGS: re.Pattern | None = None


def control_strings(checkpoints: list[Path]) -> re.Pattern | None:
    from transformers import AutoTokenizer

    added = {token for path in checkpoints for token in AutoTokenizer.from_pretrained(str(path)).get_added_vocab()}
    return re.compile("|".join(map(re.escape, sorted(added, key=len, reverse=True)))) if added else None


def usable_prompt(text: str) -> str | None:
    text = " ".join(text.split())
    words = text.split()
    if not (MIN_PROMPT_WORDS <= len(words) <= MAX_PROMPT_WORDS):
        return None
    if len(text) > MAX_PROMPT_CHARS:
        return None
    if CONTROL_STRINGS is not None and CONTROL_STRINGS.search(text):
        return None
    if TRANSCRIPT.search(text) or PERSONA_TASK.search(text):
        return None
    if sum(c.isascii() for c in text) / len(text) < ASCII_SHARE:
        return None
    return text


def dedup_key(prompt: str) -> str:
    folded = unicodedata.normalize("NFKC", prompt).casefold()
    return " ".join(re.sub(r"[^\w\s]+", " ", folded).split())


def ngram_keys(prompt: str, size: int = NGRAM) -> set[bytes]:
    tokens = prompt.casefold().split()
    return {
        hashlib.blake2b(" ".join(tokens[i : i + size]).encode(), digest_size=8).digest()
        for i in range(len(tokens) - size + 1)
    }


def first_user_turn(row: dict, column: str) -> str | None:
    value = row.get(column)
    if isinstance(value, str) and value.lstrip().startswith("["):
        try:
            value = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return None
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        for message in value:
            if not isinstance(message, dict):
                return None
            if message.get("role") == "user" or message.get("from") == "human":
                content = message.get("content") or message.get("value") or ""
                return content if isinstance(content, str) else None
    return None


def choose_files(files: list[str], max_files: int, seed_key: str) -> list[str]:
    ordered = sorted(files)
    if len(ordered) <= max_files:
        return ordered
    rng = random.Random(seed_key)
    return rng.sample(ordered, max_files)


def iter_rows(path: Path) -> Iterator[dict]:
    if path.suffix == ".jsonl":
        with path.open() as handle:
            for line in handle:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue
    else:
        for batch in pq.ParquetFile(path).iter_batches(batch_size=2048):
            yield from batch.to_pylist()


def collect(source: Source, name: str, cap: int, max_files: int, seed: int) -> tuple[list[str], list[str], int]:
    files = choose_files(
        [
            f
            for f in list_repo_files(source.repo, repo_type="dataset")
            if f.startswith(source.prefix) and f.endswith(source.suffix)
        ],
        max_files,
        f"{seed}:{name}",
    )
    prompts: list[str] = []
    seen: set[str] = set()
    read: list[str] = []
    rows_read = 0
    for name_in_repo in files:
        local = Path(hf_hub_download(source.repo, name_in_repo, repo_type="dataset"))
        read.append(name_in_repo)
        for row in iter_rows(local):
            rows_read += 1
            raw = first_user_turn(row, source.column)
            if raw is None:
                continue
            prompt = usable_prompt(raw)
            if prompt is None:
                continue
            key = dedup_key(prompt)
            if key in seen:
                continue
            seen.add(key)
            prompts.append(prompt)
            if len(prompts) >= cap:
                break
        print(f"  {name}: {len(prompts)} prompts after {name_in_repo} ({rows_read} rows)", flush=True)
        if len(prompts) >= cap:
            break
    return prompts, read, rows_read


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--datasets", nargs="*", default=sorted(SOURCES), choices=sorted(SOURCES))
    parser.add_argument("--num-prompts", type=int, default=100_000)
    parser.add_argument("--per-dataset", type=int, default=40_000)
    parser.add_argument("--max-files", type=int, default=8)
    parser.add_argument("--num-eval", type=int, default=200)
    parser.add_argument("--ngram", type=int, default=NGRAM)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    global CONTROL_STRINGS
    tokenizer_checkpoints = sorted({MODEL_ROOT / spec.directory for spec in FAMILIES.values()})
    tokenizer_checkpoints = [path for path in tokenizer_checkpoints if (path / "tokenizer_config.json").exists()]
    CONTROL_STRINGS = control_strings(tokenizer_checkpoints)
    print(f"control strings from {len(tokenizer_checkpoints)} tokenizers: {[p.name for p in tokenizer_checkpoints]}")

    collected: dict[str, list[str]] = {}
    files_read: dict[str, list[str]] = {}
    rows_read: dict[str, int] = {}
    revisions: dict[str, str] = {}
    for name in args.datasets:
        source = SOURCES[name]
        print(f"{name}: {source.repo}", flush=True)
        revisions[name] = Path(
            snapshot_download(source.repo, repo_type="dataset", allow_patterns=["*.md", "*.json"])
        ).name
        collected[name], files_read[name], rows_read[name] = collect(
            source, name, args.per_dataset, args.max_files, args.seed
        )

    rng = random.Random(args.seed)
    seen: set[str] = set()
    bank: list[dict] = []
    kept = collections.Counter()
    for name in args.datasets:
        for prompt in collected[name]:
            key = dedup_key(prompt)
            if key in seen:
                continue
            seen.add(key)
            kept[name] += 1
            bank.append({"prompt": prompt, "source": name})
    rng.shuffle(bank)
    dropped_ngram: collections.Counter[str] = collections.Counter()
    if args.ngram:
        seen_grams: set[bytes] = set()
        kept_rows = []
        for row in bank:
            grams = ngram_keys(row["prompt"], args.ngram)
            if grams & seen_grams:
                dropped_ngram[row["source"]] += 1
                continue
            seen_grams |= grams
            kept_rows.append(row)
        bank = kept_rows
    if len(bank) < args.num_prompts:
        raise ValueError(f"only {len(bank)} distinct prompts survive deduplication; asked for {args.num_prompts}")
    bank = bank[: args.num_prompts]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(bank, indent=1))

    provenance = {
        "built": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "command": " ".join(shlex.quote(part) for part in sys.argv),
        "num_prompts": len(bank),
        "num_eval": args.num_eval,
        "seed": args.seed,
        "filters": {
            "words": [MIN_PROMPT_WORDS, MAX_PROMPT_WORDS],
            "chars": MAX_PROMPT_CHARS,
            "ascii_share": ASCII_SHARE,
            "content": None,
            "control_strings_from": [path.name for path in tokenizer_checkpoints],
        },
        "ngram": args.ngram,
        "dropped_by_ngram": dict(dropped_ngram.most_common()),
        "datasets": {
            name: {
                "repo": SOURCES[name].repo,
                "revision": revisions[name],
                "column": SOURCES[name].column,
                "files": files_read[name],
                "rows_read": rows_read[name],
                "usable": len(collected[name]),
                "after_cross_dedup": int(kept[name]),
                "dropped_by_ngram": int(dropped_ngram[name]),
                "in_bank": sum(row["source"] == name for row in bank),
            }
            for name in args.datasets
        },
    }
    args.output.with_suffix(".provenance.json").write_text(json.dumps(provenance, indent=1) + "\n")
    total_usable = sum(len(v) for v in collected.values())
    print(f"\n{len(bank)} prompts written to {args.output}")
    print(f"collected {total_usable} usable, {total_usable - len(seen)} cross-dataset duplicates dropped")
    print(f"{sum(dropped_ngram.values())} more dropped as {args.ngram}-gram near-duplicates")
    for name in args.datasets:
        print(
            f"  {name}: usable {len(collected[name])}, new {kept[name]}, in bank {provenance['datasets'][name]['in_bank']}"
        )


if __name__ == "__main__":
    main()
