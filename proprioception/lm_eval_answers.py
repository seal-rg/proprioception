import contextvars
import json
import logging
import statistics
from pathlib import Path

from lm_eval.evaluator import simple_evaluate
from lm_eval.loggers import EvaluationTracker
from lm_eval.models.openai_completions import LocalChatCompletion
from lm_eval.models.utils import postprocess_generated_text
from lm_eval.tasks import TaskManager
from lm_eval.utils import make_table

from proprioception.oracle import ReasoningMarkers

TASKS_DIR = Path(__file__).parent / "lm_eval_tasks"
DEFAULT_MAX_GEN_TOKS = 32768
MIN_REQUEST_TIMEOUT = 1200.0
REQUEST_SECONDS_PER_TOKEN = 0.1

logger = logging.getLogger(__name__)
_RESPONSE_STATS: contextvars.ContextVar[list | None] = contextvars.ContextVar("response_stats", default=None)


def request_timeout(gen_kwargs: str) -> float:
    budget = DEFAULT_MAX_GEN_TOKS
    for item in gen_kwargs.split(","):
        key, _, value = item.partition("=")
        if key.strip() == "max_gen_toks" and value.strip():
            budget = int(value)
    return max(MIN_REQUEST_TIMEOUT, REQUEST_SECONDS_PER_TOKEN * budget)


def response_stats(choice: dict, usage: dict) -> dict:
    message = choice.get("message") or {}
    content = message.get("content")
    reasoning = message.get("reasoning_content") or message.get("reasoning") or ""
    return {
        "completion_tokens": usage.get("completion_tokens"),
        "reasoning_chars": len(reasoning),
        "content_chars": None if content is None else len(content),
        "finish_reason": choice.get("finish_reason"),
    }


def share(hits: int, total: int) -> float:
    return hits / total if total else float("nan")


def generation_summary(records: list[dict]) -> dict:
    by_task: dict[str, list[dict]] = {}
    for record in records:
        by_task.setdefault(str(record.get("task")), []).append(record)
    summary = {}
    for task, rows in by_task.items():
        answered = [row for row in rows if "finish_reason" in row]
        tokens = [row["completion_tokens"] for row in answered if row.get("completion_tokens") is not None]
        reasoning = [row["reasoning_chars"] for row in answered if row["reasoning_chars"]]
        summary[task] = {
            "requests": len(rows),
            "timed_out": sum(1 for row in rows if row.get("timed_out")),
            "reasoned": share(sum(1 for row in answered if row["reasoning_chars"]), len(answered)),
            "reasoning_unclosed": share(sum(1 for row in answered if row["content_chars"] is None), len(answered)),
            "truncated": share(sum(1 for row in answered if row["finish_reason"] == "length"), len(answered)),
            "completion_tokens_mean": statistics.mean(tokens) if tokens else None,
            "completion_tokens_median": statistics.median(tokens) if tokens else None,
            "reasoning_chars_median": statistics.median(reasoning) if reasoning else None,
        }
    return summary


def prompt_key(message) -> str:
    return getattr(message, "prompt", message)


def visible_answer(content: str, finish_reason: str | None, markers: ReasoningMarkers) -> str:
    tail = content.rsplit(markers.close, maxsplit=1)[-1]
    if finish_reason == "length":
        tail = tail.split(markers.open, maxsplit=1)[0]
    return tail.replace(markers.open, "").strip()


class AnswerChatCompletion(LocalChatCompletion):
    def __init__(self, *, markers: ReasoningMarkers, **kwargs):
        super().__init__(**kwargs)
        self.markers = markers
        self.num_timeouts = 0
        self.request_stats: dict[str, list[dict]] = {}
        self.generation_records: list[dict] = []

    async def amodel_call(self, session, sem, messages, **kwargs):
        collected: list[dict] = []
        token = _RESPONSE_STATS.set(collected)
        try:
            return await super().amodel_call(session, sem, messages, **kwargs)
        except TimeoutError:
            self.num_timeouts += len(messages)
            collected[:] = [{"timed_out": True} for _ in messages]
            logger.warning("request timed out after %.0f s; answered empty", self.timeout)
            return [""] * len(messages)
        finally:
            _RESPONSE_STATS.reset(token)
            for message, stats in zip(messages, collected):
                self.request_stats.setdefault(prompt_key(message), []).append(stats)

    def generate_until(self, requests, disable_tqdm: bool = False) -> list[str]:
        stops = []
        for request in requests:
            context, gen_kwargs, *rest = request.args
            gen_kwargs = dict(gen_kwargs)
            stops.append(gen_kwargs.pop("until", None))
            request.arguments = (context, gen_kwargs, *rest)
        replies = super().generate_until(requests, disable_tqdm=disable_tqdm)
        for request in requests:
            pending = self.request_stats.get(prompt_key(request.args[0])) or [{}]
            stats = pending.pop(0) if len(pending) > 1 else pending[0]
            self.generation_records.append(
                {"task": getattr(request, "task_name", None), "doc_id": getattr(request, "doc_id", None), **stats}
            )
        return [
            postprocess_generated_text(reply, stop=stop, think_end_token=None)
            for reply, stop in zip(replies, stops, strict=True)
        ]

    def parse_generations(self, outputs: dict | list[dict], **kwargs) -> list[str | None]:
        answers = []
        stats = _RESPONSE_STATS.get()
        for output in outputs if isinstance(outputs, list) else [outputs]:
            choices = sorted(output.get("choices") or [], key=lambda choice: choice["index"])
            for choice in choices or [{}]:
                if stats is not None:
                    stats.append(response_stats(choice, output.get("usage") or {}))
                content = (choice.get("message") or {}).get("content")
                answers.append(
                    None if content is None else visible_answer(content, choice.get("finish_reason"), self.markers)
                )
        return answers


def run_lm_eval(
    *,
    checkpoint: str,
    base_url: str,
    tasks: list[str],
    gen_kwargs: str,
    output: Path,
    markers: ReasoningMarkers,
    num_concurrent: int = 32,
    limit: int = 0,
    timeout: float | None = None,
) -> dict:
    model = AnswerChatCompletion(
        markers=markers,
        model=checkpoint,
        base_url=f"{base_url}/v1/chat/completions",
        num_concurrent=num_concurrent,
        timeout=request_timeout(gen_kwargs) if timeout is None else timeout,
    )
    tracker = EvaluationTracker(output_path=str(output))
    results = simple_evaluate(
        model=model,
        model_args={"model": checkpoint},
        tasks=[*tasks],
        gen_kwargs=gen_kwargs,
        apply_chat_template=True,
        log_samples=True,
        evaluation_tracker=tracker,
        limit=limit or None,
        task_manager=TaskManager(include_path=str(TASKS_DIR)),
    )
    if results is None:
        raise RuntimeError("lm_eval returned no results")
    payload: dict = dict(results)
    payload["request_timeouts"] = model.num_timeouts
    payload["generation_stats"] = generation_summary(model.generation_records)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "generation_stats.jsonl").open("w") as handle:
        for record in model.generation_records:
            handle.write(json.dumps(record) + "\n")
    samples = payload.pop("samples")
    tracker.save_results_aggregated(results=payload, samples=samples)
    for task in payload["configs"]:
        tracker.save_results_samples(task_name=task, samples=samples[task])
    print(make_table(payload), flush=True)
    return payload
