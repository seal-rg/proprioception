import dataclasses
import enum
import sys
import types
import typing
from pathlib import Path

import tomli_w
import tomllib

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "configs" / "reference"
sys.path.insert(0, str(REPO))
UNSET = object()
REFERENCE_NAMES = {"eval": "template_eval"}
REQUIRED = object()


def choices(annotation) -> str:
    args = typing.get_args(annotation)
    if typing.get_origin(annotation) is typing.Literal:
        return " | ".join(repr(a) for a in args)
    if typing.get_origin(annotation) in (typing.Union, types.UnionType):
        names = [a.__name__ for a in args if isinstance(a, type) and a is not type(None)]
        if len(names) > 1 and all(hasattr(a, "model_fields") for a in args if a is not type(None)):
            return " | ".join(names)
    return ""


def plain(value):
    if value is None:
        return UNSET
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {f.name: plain(getattr(value, f.name)) for f in dataclasses.fields(value)}
    if hasattr(value, "model_dump"):
        return pydantic_defaults(type(value), value)
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, enum.Enum):
        return value.value
    return value


def dataclass_defaults(cls) -> tuple[dict, dict]:
    values, notes = {}, {}
    hints = typing.get_type_hints(cls)
    for f in dataclasses.fields(cls):
        if not f.init:
            continue
        if f.default is not dataclasses.MISSING:
            values[f.name] = plain(f.default)
        elif f.default_factory is not dataclasses.MISSING:
            values[f.name] = plain(f.default_factory())
        else:
            values[f.name] = REQUIRED
        if note := choices(hints.get(f.name)):
            notes[f.name] = note
    return values, notes


NOTES: dict[tuple[str, ...], str] = {}


def pydantic_defaults(cls, instance=None, path: tuple[str, ...] = ()) -> dict:
    from pydantic_core import PydanticUndefined

    values = {}
    for name, f in cls.model_fields.items():
        if note := choices(f.annotation):
            NOTES[(*path, name)] = note
        annotation = f.annotation
        if instance is not None and name in instance.__dict__:
            value = instance.__dict__[name]
        elif f.default is not PydanticUndefined:
            value = f.default
        elif f.default_factory is not None:
            try:
                value = f.default_factory()
            except TypeError:
                value = UNSET
        elif isinstance(annotation, type) and hasattr(annotation, "model_fields"):
            values[name] = pydantic_defaults(annotation, path=(*path, name))
            continue
        else:
            values[name] = REQUIRED
            continue
        optional = [a for a in typing.get_args(annotation) if isinstance(a, type) and hasattr(a, "model_fields")]
        if value is None and optional:
            NOTES[(*path, name)] = "optional section, absent by default"
            values[name] = pydantic_defaults(optional[0], path=(*path, name))
            continue
        values[name] = (
            pydantic_defaults(type(value), value, (*path, name))
            if hasattr(type(value), "model_fields")
            else plain(value)
        )
    return values


def render(values: dict, notes: dict[tuple[str, ...], str], path: tuple[str, ...] = (), header: str = "") -> list[str]:
    scalars, tables, arrays = [], [], []
    for key, value in values.items():
        if isinstance(value, dict) and value:
            tables.append((key, value))
        elif isinstance(value, list) and value and all(isinstance(v, dict) for v in value):
            arrays.append((key, value))
        else:
            scalars.append((key, value))
    lines = [header] if header else []
    for key, value in scalars:
        note = notes.get((*path, key), "")
        suffix = f"  # {note}" if note else ""
        if value is UNSET:
            lines.append(f"# {key} =  # unset{'; ' + note if note else ''}")
        elif value is REQUIRED:
            lines.append(f"# {key} =  # required{'; ' + note if note else ''}")
        else:
            cleaned = strip(value)
            lines.append(f"{key} = {literal(cleaned)}{suffix}")
    for key, value in tables:
        dotted = ".".join((*path, key))
        note = notes.get((*path, key), "")
        lines += ["", *render(value, notes, (*path, key), f"[{dotted}]" + (f"  # {note}" if note else ""))]
    for key, entries in arrays:
        dotted = ".".join((*path, key))
        for entry in entries:
            lines += ["", *render(entry, notes, (*path, key), f"[[{dotted}]]")]
    return lines


def literal(value) -> str:
    if isinstance(value, list):
        return "[" + ", ".join(literal(v) for v in value) + "]"
    if isinstance(value, dict):
        return "{" + ", ".join(f"{k} = {literal(v)}" for k, v in value.items()) + "}"
    return tomli_w.dumps({"v": value}).strip()[len("v = ") :]


def strip(value):
    if isinstance(value, dict):
        return {k: strip(v) for k, v in value.items() if v is not UNSET and v is not REQUIRED}
    if isinstance(value, list):
        return [strip(v) for v in value if v is not UNSET]
    return value


def write(name: str, values: dict, notes: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    text = "\n".join(render(values, notes)).lstrip("\n") + "\n"
    tomllib.loads(text)
    (OUT / f"{name}.toml").write_text(text)
    print(f"wrote {OUT / name}.toml")


def stage_references() -> None:
    from proprioception.config import STAGES

    for stage, cls in STAGES.items():
        values, notes = dataclass_defaults(cls)
        write(REFERENCE_NAMES.get(stage, stage), {"cmd": stage, **values}, {(k,): v for k, v in notes.items()})


def evals_reference() -> None:
    from evals.config import EvalConfig, ModelSpec

    values, notes = dataclass_defaults(EvalConfig)
    model, _ = dataclass_defaults(ModelSpec)
    model.pop("name")
    values["models"] = {"base": model}
    suite_values, suite_notes = suite_defaults()
    values["suite"] = suite_values
    write("eval_suite", values, {(k,): v for k, v in notes.items()} | suite_notes)


def suite_defaults() -> tuple[dict, dict]:
    import ast
    import inspect

    from evals.suites import SUITES, load

    values, notes = {}, {}
    for name in SUITES:
        module = load(name)
        source = inspect.getsource(module)
        table: dict = {}
        for node in ast.walk(ast.parse(source)):
            if not (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "arg"
                and len(node.args) == 2
                and isinstance(node.args[0], ast.Constant)
            ):
                continue
            key = node.args[0].value
            try:
                default = plain(eval(compile(ast.Expression(node.args[1]), name, "eval"), vars(module)))
            except Exception:
                default = UNSET
                notes[("suite", name, key)] = (
                    f"default computed at run time: {ast.get_source_segment(source, node.args[1])}"
                )
            if key in table and table[key] != default:
                notes[("suite", name, key)] = "read with more than one default"
            table.setdefault(key, default)
        values[name] = dict(sorted(table.items())) if table else {}
    return values, notes


def prime_references() -> None:
    from prime_rl.configs.inference import InferenceConfig
    from prime_rl.configs.orchestrator import EvalEnvConfig, TrainEnvConfig
    from prime_rl.configs.rl import RLConfig

    import proprioception.taskset as taskset

    NOTES.clear()
    proprio = pydantic_defaults(taskset.ProprioceptionConfig, path=("orchestrator", "train", "env", "taskset"))
    proprio["task"] = pydantic_defaults(
        taskset.ProprioceptionTaskConfig, path=("orchestrator", "train", "env", "taskset", "task")
    )
    proprio["task"]["user"] = pydantic_defaults(
        taskset.ProprioceptionUserConfig, path=("orchestrator", "train", "env", "taskset", "task", "user")
    )
    proprio["id"] = "proprioception"
    rl = {"cmd": "rl", **pydantic_defaults(RLConfig)}
    for section, env_cls in (("train", TrainEnvConfig), ("eval", EvalEnvConfig)):
        env = pydantic_defaults(env_cls, path=("orchestrator", section, "env"))
        env["taskset"] = proprio
        rl["orchestrator"][section]["env"] = [env]
    for (*head, key), note in list(NOTES.items()):
        if head[:3] == ["orchestrator", "train", "env"]:
            NOTES[("orchestrator", "eval", "env", *head[3:], key)] = note
    write("rl", rl, dict(NOTES))
    NOTES.clear()
    write("inference", pydantic_defaults(InferenceConfig), dict(NOTES))


if __name__ == "__main__":
    stage_references()
    evals_reference()
    prime_references()
