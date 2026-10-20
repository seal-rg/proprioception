import json
from pathlib import Path

BANK = Path(__file__).resolve().parents[1] / "data/prompt_bank.json"
BANK_EVAL_PROMPTS = 200


def load_prompts(path: str | Path = BANK, held_out: int = BANK_EVAL_PROMPTS) -> tuple[list[str], list[str]]:
    prompts = [row["prompt"] for row in json.loads(Path(path).read_text())]
    if held_out >= len(prompts):
        raise ValueError(f"held_out={held_out} needs a bank larger than {len(prompts)} prompts")
    return prompts[:-held_out], prompts[-held_out:]
