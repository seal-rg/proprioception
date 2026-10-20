import re

from lm_eval.tasks.aime import utils as upstream

BARE_LAST_LINE = re.compile(r"[\s*$]*(-?\d+)[\s*$.]*")
SENTENCE_FINAL = re.compile(r".*\bis[\s*$]*(-?\d+)[\s*$]*\.?[\s*$]*", re.DOTALL)
FINAL_EQUATION = re.compile(r".*=\s*(\d+)\s*(?:\$+|\\\])[\s*]*\.?[\s*]*")
PLAIN_EQUATION = re.compile(r".*=[\s*]*(\d+)[\s*]*\.?[\s*]*")
AIME_ANSWERS = range(1000)
STATED_ANSWER = re.compile(
    r"(?:^|\n)[ \t*#>]*(?:final\s+)?answer\W{0,6}(?:is\W{0,6})?(-?\d+)\b|\banswer\s+is\W{0,6}(-?\d+)\b", re.IGNORECASE
)


def stated_answer(reply: str) -> str | None:
    boxed = upstream.last_boxed_only_string(reply)
    if boxed is not None:
        try:
            content = upstream.remove_boxed(boxed)
        except (AssertionError, IndexError):
            content = None
        if content is not None:
            return content
    matches = STATED_ANSWER.findall(reply)
    if matches:
        line, sentence = matches[-1]
        return line or sentence
    lines = [line for line in reply.splitlines() if line.strip()]
    if not lines:
        return None
    last = BARE_LAST_LINE.fullmatch(lines[-1]) or SENTENCE_FINAL.fullmatch(lines[-1])
    if last:
        return last.group(1)
    equation = FINAL_EQUATION.fullmatch(lines[-1]) or PLAIN_EQUATION.fullmatch(lines[-1])
    return equation.group(1) if equation and int(equation.group(1)) in AIME_ANSWERS else None


def matches(answer: str | None, target: str) -> bool:
    if answer is None:
        return False
    if answer.strip().lstrip("-").isdigit() and target.strip().isdigit():
        return int(answer) == int(target)
    return upstream.is_equiv(answer, target)


def process_results(doc: dict, results: list[str]) -> dict[str, int]:
    target = str(doc[next(k for k in doc if k.lower() == "answer")])
    return {
        "exact_match": upstream.process_results(doc, results)["exact_match"],
        "answer_match": int(matches(stated_answer(results[0]), target)),
    }


def math_doc_to_text_cot(doc: dict) -> str:
    return "Problem:\n" + doc["problem"] + "\n\nSolution: Let's think step by step."
