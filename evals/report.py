import json
from pathlib import Path


def _flatten(prefix: str, value, out: dict) -> None:
    if isinstance(value, dict):
        for key, inner in value.items():
            _flatten(f"{prefix}.{key}" if prefix else str(key), inner, out)
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        out[prefix] = value
    elif value is None:
        out[prefix] = None


def report(results_dir: Path) -> str:
    models = sorted(p.name for p in results_dir.iterdir() if p.is_dir())
    suites: dict[str, dict[str, dict]] = {}
    for model in models:
        for path in sorted((results_dir / model).glob("*.json")):
            if path.stem.endswith(".stale"):
                continue
            payload = json.loads(path.read_text())
            flat: dict = {}
            _flatten("", payload.get("summary", {}), flat)
            suites.setdefault(path.stem, {})[model] = flat
    lines = [f"# {results_dir.name}", ""]
    for suite, per_model in suites.items():
        keys: list[str] = []
        for flat in per_model.values():
            keys.extend(k for k in flat if k not in keys)
        present = [m for m in models if m in per_model]
        lines.append(f"## {suite}")
        lines.append("")
        lines.append("| metric | " + " | ".join(present) + " |")
        lines.append("|---|" + "---|" * len(present))
        for key in keys:
            cells = []
            for model in present:
                value = per_model[model].get(key)
                cells.append("" if value is None else (f"{value:.3f}" if isinstance(value, float) else str(value)))
            lines.append(f"| {key} | " + " | ".join(cells) + " |")
        lines.append("")
    return "\n".join(lines)
