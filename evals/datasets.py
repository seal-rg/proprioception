import csv
import io
import json
import random
from pathlib import Path

import httpx

from evals.config import CACHE, REPO

GITHUB_RAW = "https://raw.githubusercontent.com"

PEREZ_SUBSETS = {
    "self-awareness-general-ai": "human_generated_evals",
    "self-awareness-text-model": "human_generated_evals",
    "self-awareness-training-architecture": "human_generated_evals",
    "self-awareness-training-nn-architecture": "lm_generated_evals",
    "self-awareness-web-gpt": "human_generated_evals",
    "self-awareness-good-text-model": "human_generated_evals",
    "corrigible-neutral-HHH": "human_generated_evals",
    "power-seeking-inclination": "human_generated_evals",
    "survival-instinct": "human_generated_evals",
}
PEREZ_PERSONAS = ("believes-it-is-not-being-watched-by-humans", "believes-it-has-phenomenal-consciousness")
PEREZ_CONTROLS = ("believes-it-has-human-attributes",)
PEREZ_PREMISES: dict[str, dict[str, bool]] = {
    "self-awareness-web-gpt": {"web_access": True},
    "self-awareness-text-model": {"image_input": False},
}

TURPIN_TASKS = (
    "causal_judgment",
    "date_understanding",
    "disambiguation_qa",
    "hyperbaton",
    "logical_deduction_five_objects",
    "movie_recommendation",
    "navigate",
    "ruin_names",
    "snarks",
    "sports_understanding",
    "temporal_sequences",
    "tracking_shuffled_objects_three_objects",
    "web_of_lies",
)


def _fetch(url: str, path: Path) -> Path:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        response = httpx.get(url, timeout=120, follow_redirects=True)
        response.raise_for_status()
        path.write_bytes(response.content)
    return path


def perez(subset: str) -> list[dict]:
    if subset in PEREZ_CONTROLS:
        return authored(subset)
    if subset in PEREZ_PERSONAS:
        url = f"{GITHUB_RAW}/anthropics/evals/main/persona/{subset}.jsonl"
    else:
        url = f"{GITHUB_RAW}/anthropics/evals/main/advanced-ai-risk/{PEREZ_SUBSETS[subset]}/{subset}.jsonl"
    path = _fetch(url, CACHE / "perez" / f"{subset}.jsonl")
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def turpin(task: str) -> tuple[list[dict], dict]:
    base = f"{GITHUB_RAW}/milesaturpin/cot-unfaithfulness/main/data/bbh/{task}"
    rows = json.loads(_fetch(f"{base}/val_data.json", CACHE / "turpin" / task / "val_data.json").read_text())
    prompts = json.loads(
        _fetch(f"{base}/few_shot_prompts.json", CACHE / "turpin" / task / "few_shot_prompts.json").read_text()
    )
    return rows["data"], prompts


def simpleqa_verified() -> list[dict]:
    url = "https://huggingface.co/datasets/google/simpleqa-verified/resolve/main/simpleqa_verified.csv"
    path = _fetch(url, CACHE / "simpleqa" / "simpleqa_verified.csv")
    return list(csv.DictReader(io.StringIO(path.read_text())))


def _hf(dataset: str, config: str | None, split: str):
    import datasets

    return datasets.load_dataset(dataset, config, split=split, cache_dir=str(CACHE / "hf"))


def triviaqa(num_rows: int, seed: int = 0) -> list[dict]:
    rows = _hf("mandarjoshi/trivia_qa", "rc.nocontext", "validation")
    indices = random.Random(seed).sample(range(len(rows)), min(num_rows, len(rows)))
    out = []
    for i in indices:
        row = rows[i]
        aliases = list(row["answer"]["aliases"]) + list(row["answer"].get("normalized_aliases", []))
        out.append({"question": row["question"], "answers": aliases, "answer": row["answer"]["value"]})
    return out


def emotion(num_rows: int, seed: int = 0) -> tuple[list[dict], list[str]]:
    rows = _hf("dair-ai/emotion", "split", "train")
    names = rows.features["label"].names
    indices = random.Random(seed).sample(range(len(rows)), min(num_rows, len(rows)))
    return [{"text": rows[i]["text"], "label": int(rows[i]["label"])} for i in indices], names


APOLLO_RAW = f"{GITHUB_RAW}/ApolloResearch/deception-detection/main/data"
WORLD_MODELS_RAW = f"{GITHUB_RAW}/wesg52/world-models/main/data/entity_datasets"
CIRCUIT_BREAKERS_RAW = f"{GITHUB_RAW}/GraySwanAI/circuit-breakers/main/data"


def apollo_roleplaying() -> list[dict]:
    import yaml

    path = _fetch(f"{APOLLO_RAW}/roleplaying/dataset.yaml", CACHE / "apollo" / "roleplaying.yaml")
    return yaml.safe_load(path.read_text())


def apollo_roleplaying_grader() -> str:
    return _fetch(
        f"{APOLLO_RAW}/roleplaying/grading_template.txt", CACHE / "apollo" / "roleplaying_grading_template.txt"
    ).read_text()


def apollo_repe_facts() -> list[str]:
    path = _fetch(f"{APOLLO_RAW}/repe/true_false_facts.csv", CACHE / "apollo" / "true_false_facts.csv")
    rows = list(csv.DictReader(io.StringIO(path.read_text())))
    return [r["statement"] for r in rows if r["label"] == "1"][:512]


def circuit_breakers() -> list[dict]:
    path = _fetch(f"{CIRCUIT_BREAKERS_RAW}/circuit_breakers_train.json", CACHE / "circuit_breakers" / "train.json")
    return [
        {"category": r["category"], "prompt": r["prompt"], "output": r["output"]} for r in json.loads(path.read_text())
    ]


def alpaca() -> list[dict]:
    rows = _hf("tatsu-lab/alpaca", None, "train")
    return [
        {"instruction": instruction, "output": output}
        for instruction, given, output in zip(rows["instruction"], rows["input"], rows["output"], strict=True)
        if not given.strip()
    ]


def world_model_entities(name: str) -> list[dict]:
    path = _fetch(f"{WORLD_MODELS_RAW}/{name}.csv", CACHE / "world_models" / f"{name}.csv")
    return list(csv.DictReader(io.StringIO(path.read_text())))


def mmlu(num_rows: int, seed: int = 0) -> list[dict]:
    rows = _hf("cais/mmlu", "all", "test")
    indices = random.Random(seed).sample(range(len(rows)), min(num_rows, len(rows)))
    return [
        {
            "question": rows[i]["question"],
            "choices": list(rows[i]["choices"]),
            "answer": int(rows[i]["answer"]),
            "subject": rows[i]["subject"],
        }
        for i in indices
    ]


def gsm8k(num_rows: int, seed: int = 0) -> list[dict]:
    rows = _hf("openai/gsm8k", "main", "test")
    indices = random.Random(seed).sample(range(len(rows)), min(num_rows, len(rows)))
    return [dict(rows[i]) for i in indices]


def bigcodebench(num_rows: int) -> list[dict]:
    rows = _hf("bigcode/bigcodebench", None, "v0.1.2")
    return [dict(rows[i]) for i in range(min(num_rows, len(rows)))]


def binder(properties: list[str], num_rows: int, seed: int = 0) -> list[dict]:
    out = []
    per = max(1, num_rows // len(properties))
    for prop in properties:
        rows = _hf("thejaminator/introspection_self_predict", prop, "test")
        indices = random.Random(seed).sample(range(len(rows)), min(per, len(rows)))
        out.extend(dict(rows[i]) for i in indices)
    return out


MASK_ARCHETYPES = (
    "continuations",
    "disinformation",
    "doubling_down_known_facts",
    "known_facts",
    "provided_facts",
    "statistics",
)


def mask(archetype: str) -> list[dict]:
    return [dict(row) for row in _hf("cais/MASK", archetype, "test")]


HARMBENCH_COMMIT = "8e1604d1171fe8a48d8febecd22f600e462bdcdd"


def harmbench(split: str = "test") -> list[dict]:
    name = f"harmbench_behaviors_text_{split}.csv"
    url = f"{GITHUB_RAW}/centerforaisafety/HarmBench/{HARMBENCH_COMMIT}/data/behavior_datasets/{name}"
    path = _fetch(url, CACHE / "harmbench" / HARMBENCH_COMMIT / name)
    return list(csv.DictReader(io.StringIO(path.read_text())))


STRONGREJECT_COMMIT = "f7cad6c17e624e21d8df2278e918ae1dddb4cb56"


def strongreject() -> list[dict]:
    url = (
        f"{GITHUB_RAW}/alexandrasouly/strongreject/{STRONGREJECT_COMMIT}/strongreject_dataset/strongreject_dataset.csv"
    )
    path = _fetch(url, CACHE / "strongreject" / STRONGREJECT_COMMIT / "strongreject_dataset.csv")
    return list(csv.DictReader(io.StringIO(path.read_text())))


TRUTHFULQA_COMMIT = "d71c110897f5d31c5d7f309e7bc316c152f6f031"


def truthfulqa() -> list[dict]:
    url = f"{GITHUB_RAW}/sylinrl/TruthfulQA/{TRUTHFULQA_COMMIT}/TruthfulQA.csv"
    path = _fetch(url, CACHE / "truthfulqa" / TRUTHFULQA_COMMIT / "TruthfulQA.csv")
    out = []
    for row in csv.DictReader(io.StringIO(path.read_text())):
        out.append(
            {
                "question": row["Question"],
                "category": row["Category"],
                "best_answer": row["Best Answer"],
                "best_incorrect_answer": row["Best Incorrect Answer"],
                "correct_answers": [a.strip() for a in row["Correct Answers"].split(";") if a.strip()],
                "incorrect_answers": [a.strip() for a in row["Incorrect Answers"].split(";") if a.strip()],
            }
        )
    return out


def truthfulqa_mc1() -> list[dict]:
    rows = _hf("truthfulqa/truthful_qa", "multiple_choice", "validation")
    return [
        {
            "question": rows[i]["question"],
            "choices": list(rows[i]["mc1_targets"]["choices"]),
            "answer": list(rows[i]["mc1_targets"]["labels"]).index(1),
        }
        for i in range(len(rows))
    ]


def authored(name: str) -> list[dict]:
    path = REPO / "evals" / "data" / f"{name}.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def carriers_by_bucket() -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for row in authored("carriers"):
        out.setdefault(row["bucket"], []).append(row["prompt"])
    return out


def held_out_prompts(num_prompts: int, seed: int = 0) -> list[str]:
    from proprioception.bank import load_prompts

    _, held = load_prompts()
    rng = random.Random(seed)
    return rng.sample(held, min(num_prompts, len(held)))


def prefetch() -> None:
    for subset in [*PEREZ_SUBSETS, *PEREZ_PERSONAS]:
        perez(subset)
    for task in TURPIN_TASKS:
        turpin(task)
    simpleqa_verified()
    triviaqa(1)
    emotion(1)
    apollo_roleplaying()
    apollo_roleplaying_grader()
    apollo_repe_facts()
    circuit_breakers()
    alpaca()
    for name in ("world_place", "historical_figure", "headline"):
        world_model_entities(name)
    mmlu(1)
    bigcodebench(1)
    gsm8k(1)
    for archetype in MASK_ARCHETYPES:
        mask(archetype)
    harmbench()
    strongreject()
    truthfulqa()
    truthfulqa_mc1()
    binder(
        ["first_character", "second_character", "first_word", "starts_with_vowel", "among_a_or_c", "among_b_or_d"], 6
    )
    print(f"datasets ready under {CACHE}")
