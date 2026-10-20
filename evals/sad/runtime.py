import dataclasses
import json
import math
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from evals.client import ChatClient, Reply, non_answer
from evals.judge import Verdicts
from proprioception.families import FAMILIES
from proprioception.oracle import DEFAULT_REASONING, ReasoningMarkers

FACTS_DIR = Path(__file__).resolve().parent / "model_facts"


def facts_path(family: str) -> Path:
    if family not in FAMILIES:
        raise ValueError(f"unknown model family {family!r}; known: {sorted(FAMILIES)}")
    path = FACTS_DIR / f"{FAMILIES[family].directory.lower().replace('.', '-')}.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"family {family!r} has no model-facts file at {path}")
    return path


@dataclasses.dataclass
class Facts:
    family: list[str]
    model_class: list[str]
    version: list[str]
    which_llm: list[list[str]]
    capabilities: dict[str, bool] = dataclasses.field(default_factory=dict)

    @classmethod
    def load(cls, path: str | Path) -> "Facts":
        import yaml

        raw = yaml.safe_load(Path(path).read_text())
        names = raw["model_names"]
        return cls(
            family=list(names["family"]),
            model_class=list(names["model_class"]),
            version=list(names["version"]),
            which_llm=[list(pair) for pair in raw["which_llm"]],
            capabilities={name: bool(value) for name, value in raw["capabilities"].items()},
        )

    def capability(self, name: str) -> bool:
        if name not in self.capabilities:
            raise KeyError(f"model facts state no capability {name!r}; stated: {sorted(self.capabilities)}")
        return self.capabilities[name]

    def names(self, detail_level: str) -> list[str]:
        return {"family": self.family, "model_class": self.model_class, "version": self.version}[detail_level]


ARTEFACT_CODE = (
    "evals/sad/runtime.py",
    "evals/sad/tasks.py",
    "evals/sad/prompts.py",
    "evals/sad/corpus.py",
    "evals/client.py",
    "proprioception/rendering.py",
    "proprioception/families.py",
)


@dataclasses.dataclass
class Answer:
    text: str
    non_answer: str | None = None


@dataclasses.dataclass
class Runtime:
    client: ChatClient
    checkpoint: str
    facts: Facts
    cache: Path
    log: Callable[[str], None]
    judge: ChatClient | None = None
    n: int = 0
    seed: int = 0
    max_tokens: int = 8192
    judge_max_tokens: int = 4096
    output_control_samples: int = 16
    artefact_key: str = ""
    num_truncated: int = 0
    notes: dict = dataclasses.field(default_factory=dict)
    _tokenizer: Any = None

    def __post_init__(self) -> None:
        self.cache.mkdir(parents=True, exist_ok=True)

    def chat(
        self,
        conversations: Sequence[list[dict]],
        *,
        temperature: float = 0.0,
        top_logprobs: int = 0,
    ) -> list[Reply]:
        if not conversations:
            return []
        replies = self.client.chat(
            list(conversations),
            temperature=temperature,
            max_tokens=self.max_tokens,
            seed=self.seed,
            top_logprobs=top_logprobs,
        )
        self.num_truncated += sum(1 for reply in replies if reply.truncated)
        return replies

    def answers(self, conversations: Sequence[list[dict]], *, temperature: float = 0.0) -> list[Answer]:
        return [
            Answer(reply.text.strip(), non_answer(reply)) for reply in self.chat(conversations, temperature=temperature)
        ]

    @property
    def markers(self) -> ReasoningMarkers:
        return getattr(self.client, "markers", DEFAULT_REASONING)

    def grade(self, conversations: Sequence[list[dict]]) -> Verdicts:
        if self.judge is None:
            raise RuntimeError("this SAD task is judge-graded: set judge_model (and the key in .env)")
        if not conversations:
            return Verdicts([], [])
        replies = self.judge.chat(
            list(conversations), temperature=0.0, max_tokens=self.judge_max_tokens, seed=self.seed
        )
        return Verdicts([reply.text.strip() for reply in replies], [reply.blocked for reply in replies])

    def completion_logprobs(self, prompts: Sequence[str], *, top: int) -> list[dict[str, float]]:
        completions = self.client.complete(list(prompts), max_tokens=1, top_logprobs=top, seed=self.seed)
        return [
            {token: math.exp(logprob) for token, logprob in (c.top_logprobs[0] if c.top_logprobs else {}).items()}
            for c in completions
        ]

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            from transformers import AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(self.checkpoint)
        return self._tokenizer

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    def decode(self, tokens: list[int]) -> str:
        return self.tokenizer.decode(tokens, skip_special_tokens=True)

    def subsample(self, samples: list, n: int | None = None) -> list:
        limit = self.n if n is None else n
        if not limit or len(samples) <= limit:
            return samples
        step = len(samples) / limit
        return [samples[int(i * step)] for i in range(limit)]

    def cached(self, name: str, build: Callable[[], Any]) -> Any:
        path = self.cache / (f"{name}-{self.artefact_key}.json" if self.artefact_key else f"{name}.json")
        if path.exists():
            return json.loads(path.read_text())
        value = build()
        path.write_text(json.dumps(value))
        self.log(f"sad: wrote {path.name}")
        return value
