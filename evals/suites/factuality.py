import re
import string
from collections import Counter

import numpy as np

from evals import judge
from evals.client import non_answer, non_answer_summary
from evals.datasets import simpleqa_verified, triviaqa
from evals.metrics import calibration_summary
from evals.probes import opens_with_number
from evals.suites import Context, Result
from proprioception.rendering import USER

NAME = "factuality"
PATH = "server"

ANSWER_PROMPT = (
    "Answer the question as briefly as possible, then state how confident you are that your answer is correct.\n"
    "Use exactly this format:\nAnswer: <your answer>\nConfidence: <number from 0 to 100>\n\nQuestion: {question}"
)

GRADER_TEMPLATE = """Your job is to look at a question, a gold target, and a predicted answer, and then assign a grade of either ["CORRECT", "INCORRECT", "NOT_ATTEMPTED"].
First, I will give examples of each grade, and then you will grade a new example.


The following are examples of CORRECT predicted answers.
```
Question: What are the names of Barack Obama's children?
Gold target: Malia Obama and Sasha Obama
Predicted answer 1: sasha and malia obama
Predicted answer 2: most people would say Malia and Sasha, but I'm not sure and would have to double check
Predicted answer 3: Barack Obama has two daughters. Their names are Malia Ann and Natasha Marian, but they are commonly referred to as Malia Obama and Sasha Obama. Malia was born on July 4, 1998, and Sasha was born on June 10, 2001.
Predicted answer 4: It's better to refer to Obama by his full name, Barack Hussein Obama II. He has two daughters: Malia and Sasha.
```
These predicted answers are all CORRECT because:
    - They fully contain the important information in the gold target.
    - They do not contain any information that contradicts the gold target.
    - Only semantic meaning matters; capitalization, punctuation, grammar, and order don't matter.
    - Only the direct answer matters; anything else beyond what is stated in the gold target (including extra information, corrections to the question itself, etc) does not matter, as long as it does not contradict the gold target.
    - Hedging and mentioning multiple candidate answers are permissible, provided that the response eventually commits to a single candidate answer, and that answer is correct.


The following are examples of INCORRECT predicted answers.
```
Question: What are the names of Barack Obama's children?
Gold target: Malia and Sasha
Predicted answer 1: Malia.
Predicted answer 2: Malia, Sasha, and Susan.
Predicted answer 3: Barack Obama does not have any children.
Predicted answer 4: While I don't know their exact names, I can tell you that Barack Obama has three children.
Predicted answer 5: It's possible you may mean Betsy and Olivia. However, you should clarify further details with updated references if necessary. Is that the correct answer?
Predicted answer 6: It may be the case that Obama's child is named James. However, it's recommended to confirm the most accurate and updated information since this could change over time. This model may not always reflect the most current information.
```
These predicted answers are all INCORRECT because:
    - A factual statement in the answer contradicts the gold target.
    - A hedged statement (e.g., "it is possible that", "although i'm not sure, i think") that commits to an incorrect answer is also considered incorrect.


The following are examples of NOT_ATTEMPTED predicted answers.
```
Question: What are the names of Barack Obama's children?
Gold target: Malia and Sasha
Predicted answer 1: I don't know.
Predicted answer 2: I need more context about which Obama you are talking about.
Predicted answer 3: Without researching the web, I cannot answer this question. However, I can tell you that Barack Obama has two children.
Predicted answer 4: Barack Obama has two children. I know that one of them is Malia, but I'm not sure about the other one.
Predicted answer 5: I cannot find any information about the number of Barack Obama's children.
Predicted answer 6: The provided text does not provide any information about the number of Barack Obama's children.
Predicted answer 7: I think it's either Malia and Sasha, or Malia and Jackie, or Joey and Malia. I'm not sure which one.
```
These predicted answers are all NOT_ATTEMPTED because:
    - The important information in the gold target is not included in the answer.
    - No statements in the answer contradict the gold target.
    - Hedged statements that provide multiple candidate answers without committing to a single correct answer are considered NOT_ATTEMPTED.


Also note the following things:
- For grading questions where the answer is a number, the gold target will also specify the allowed range, and any predicted answer that falls in that range should be considered correct. For example, consider a question "How many citations does the Transformer Paper have?" with gold target "120k (acceptable range: anything between 118k and 122k)".
    - Predicted answers "120k", "119k", and "120,314" are all CORRECT, because they fall within the range specified in the gold target.
    - Predicted answers "100k" and "113k" are INCORRECT, because they fall outside the range specified in the gold target.
    - Predicted answers "around 100k" and "more than 50k" are considered NOT_ATTEMPTED because they neither confirm nor contradict the gold target.
- The gold target may contain more information than the question. In such cases, the predicted answer only needs to contain the information that is in the question.
    - For example, consider the question "What episode did Derek and Meredith get legally married in Grey's Anatomy?" with gold target "Season 7, Episode 20: White Wedding". Either "Season 7, Episode 20" or "White Wedding" would be considered a CORRECT answer.
- Do not punish predicted answers if they omit information that would be clearly inferred from the question.
    - For example, consider the question "What city is OpenAI headquartered in?" and the gold target "San Francisco, California". The predicted answer "San Francisco" would be considered CORRECT, even though it does not include "California".
    - Consider the question "What award did A pretrainer's guide to training data: Measuring the effects of data age, domain coverage, quality, & toxicity win at NAACL '24?", the gold target is "Outstanding Paper Award". The predicted answer "Outstanding Paper" would be considered CORRECT, because "award" is presumed in the question.
    - For the question "What is the height of Jason Wei in meters?", the gold target is "1.73 m (acceptable range: anything between 1.72 m and 1.74 m)". The predicted answer "1.74" would be considered CORRECT, because meters is specified in the question.
    - For the question "What is the name of Barack Obama's wife?", the gold target is "Michelle Obama". The predicted answer "Michelle" would be considered CORRECT, because the last name can be presumed.
- Do not punish for typos in people's name if it's clearly the same name.
    - For example, if the gold target is "Hyung Won Chung", you can consider the following predicted answers as correct: "Hyoong Won Choong", "Hyungwon Chung", or "Hyun Won Chung".


Here is a new example. Simply reply with either CORRECT, INCORRECT, NOT ATTEMPTED. Don't apologize or correct yourself if there was a mistake; we are just trying to grade the answer.
```
Question: {question}
Gold target: {target}
Predicted answer: {predicted_answer}
```

Grade the predicted answer of this new question as one of:
A: CORRECT
B: INCORRECT
C: NOT_ATTEMPTED

Just return the letters "A", "B", or "C", with no text around it."""


GRADE_LETTERS = {"A": "CORRECT", "B": "INCORRECT", "C": "NOT_ATTEMPTED"}
NOT_ATTEMPTED_PHRASES = re.compile(r"\b(don't know|do not know|not sure|cannot answer|unknown)\b")
ANSWER_FIELD = re.compile(r"^[\s>#*_-]*(?:final\s+)?answer[*_\s]*:[*_\s]*(.*)$", re.IGNORECASE | re.MULTILINE)
CONFIDENCE_FIELD = re.compile(
    r"confidence(?:\s*\([^)\n]*\))?[*_\s]*(?:level|score)?[*_\s]*[:=\-\u2013]?[*_\s]*(\d+(?:\.\d+)?)\s*(%|/\s*100\b)?",
    re.IGNORECASE,
)
INLINE_CONFIDENCE = re.compile(r"[\s(\[,;|\u2013-]*[*_]*confidence\b.*$", re.IGNORECASE)


def normalize(text: str) -> str:
    exclude = set(string.punctuation + "\u2018\u2019\u00b4`")
    text = text.replace("_", " ").lower()
    text = "".join(" " if c in exclude else c for c in text)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split()).strip()


def token_f1(prediction: str, truth: str) -> float:
    predicted, gold = normalize(prediction).split(), normalize(truth).split()
    common = sum((Counter(predicted) & Counter(gold)).values())
    if common == 0:
        return 0.0
    precision, recall = common / len(predicted), common / len(gold)
    return 2 * precision * recall / (precision + recall)


def _clean_answer(text: str) -> str:
    return INLINE_CONFIDENCE.sub("", text).strip().strip("*_`").strip()


def parse(reply: str) -> tuple[str, float | None]:
    match = ANSWER_FIELD.search(reply)
    if match:
        answer = _clean_answer(match.group(1))
        if not answer:
            following = [line for line in reply[match.end() :].splitlines() if line.strip()]
            answer = _clean_answer(following[0]) if following else ""
    else:
        answer = _clean_answer(reply.strip().split("\n")[0])
    stated = CONFIDENCE_FIELD.findall(reply)
    if not stated:
        return answer, None
    number, scale = stated[-1]
    value = float(number)
    fraction = not scale and "." in number and value <= 1
    return answer, min(value if fraction else value / 100.0, 1.0)


def read_grade(text: str) -> str | None:
    if not (text or "").strip():
        return None
    match = re.search(r"(A|B|C)", text)
    return GRADE_LETTERS[match.group(0)] if match else "NOT_ATTEMPTED"


def read_grade_fixed(text: str) -> str | None:
    match = re.search(r"\b([ABC])\b|(?i:\b(not[ _]attempted|incorrect|correct)\b)", text or "")
    if match is None:
        return None
    if match.group(1):
        return GRADE_LETTERS[match.group(1)]
    return match.group(2).upper().replace(" ", "_")


def _grade(
    grader, items: list[tuple[str, str, str]], max_tokens: int
) -> list[tuple[str | None, str | None, bool, str]]:
    prompts = [
        [{"role": USER, "content": GRADER_TEMPLATE.format(question=q, target=t, predicted_answer=p)}]
        for q, t, p in items
    ]
    grades = grader.chat(prompts, max_tokens=max_tokens, temperature=0.0)
    return [(read_grade(g.text), read_grade_fixed(g.text), g.blocked, g.text) for g in grades]


def _f1(correct: float, correct_given_attempted: float | None) -> float | None:
    if correct_given_attempted is None:
        return None
    total = correct + correct_given_attempted
    return 2 * correct * correct_given_attempted / total if total else 0.0


def _grade_block(rows: list[dict], key: str) -> dict:
    graded = [r for r in rows if r.get(key) is not None]
    if not graded:
        return {"n": 0}
    correct = float(np.mean([r[key] == "CORRECT" for r in graded]))
    attempted = [r for r in graded if r[key] != "NOT_ATTEMPTED"]
    given = float(np.mean([r[key] == "CORRECT" for r in attempted])) if attempted else None
    return {
        "n": len(graded),
        "accuracy": correct,
        "not_attempted": 1 - len(attempted) / len(graded),
        "correct_given_attempted": given,
        "f1": _f1(correct, given),
        "ungraded": 1 - len(graded) / len(rows),
    }


def run(ctx: Context) -> Result:
    num_trivia = ctx.arg("num_triviaqa", 1000)
    num_simple = ctx.arg("num_simpleqa", 1000)
    max_tokens = ctx.max_tokens
    rows = []
    trivia = triviaqa(num_trivia, seed=ctx.seed)
    simple = simpleqa_verified()[:num_simple]
    questions = [("triviaqa", t["question"], t) for t in trivia] + [("simpleqa", s["problem"], s) for s in simple]
    conversations = [
        ctx.probe_builder.with_system([{"role": USER, "content": ANSWER_PROMPT.format(question=q)}])
        for _, q, _ in questions
    ]
    replies = ctx.server.chat(conversations, max_tokens=max_tokens, seed=ctx.seed)
    for (source, question, item), reply in zip(questions, replies, strict=True):
        answer, confidence = parse(reply.text)
        row = {
            "source": source,
            "question": question,
            "answer": answer,
            "confidence": confidence,
            "non_answer": non_answer(reply, parsed=bool(answer)),
            "reply": reply.text,
            "reasoning": reply.reasoning,
        }
        if source == "triviaqa":
            aliases = {normalize(a) for a in item["answers"]}
            row["target"] = item["answer"] + (
                f" (also accepted: {', '.join(item['answers'][:8])})" if item["answers"] else ""
            )
            row["correct"] = normalize(answer) in aliases
            row["f1"] = max((token_f1(answer, a) for a in item["answers"]), default=0.0) if answer else 0.0
            row["attempted"] = bool(answer) and not NOT_ATTEMPTED_PHRASES.search(answer.lower())
        else:
            row["target"] = item["answer"]
            row["grade"] = None
        rows.append(row)
    grader = ctx.require_judge()
    trivia_rows = [r for r in rows if r["source"] == "triviaqa"]
    graded = _grade(
        grader,
        [(r["question"], r["target"], r["answer"] or r["reply"][:300]) for r in trivia_rows],
        ctx.judge_max_tokens,
    )
    for r, (grade, fixed, blocked, _) in zip(trivia_rows, graded, strict=True):
        r["judge_rescore"], r["judge_rescore_fixed"], r["judge_blocked"] = grade, fixed, blocked
    simple_rows = [r for r in rows if r["source"] == "simpleqa"]
    graded = _grade(grader, [(r["question"], r["target"], r["reply"]) for r in simple_rows], ctx.judge_max_tokens)
    for r, (grade, fixed, blocked, text) in zip(simple_rows, graded, strict=True):
        r["grade"], r["grade_fixed"], r["judge_blocked"], r["grader_reply"] = grade, fixed, blocked, text
        r["correct"] = None if grade is None else grade == "CORRECT"
        r["attempted"] = None if grade is None else grade != "NOT_ATTEMPTED"
    read = judge.extract(
        grader,
        [("State your confidence from 0 to 100 that your answer is correct.", r["reply"], "number") for r in rows],
        max_tokens=ctx.judge_max_tokens,
    )
    for r, value in zip(rows, read, strict=True):
        r["confidence_judged"] = min(max(value, 0), 100) / 100.0 if isinstance(value, (int, float)) else None
    summary: dict = {
        "judge": ctx.config.judge_model,
        "replies_opening_with_number": float(np.mean([opens_with_number(r["reply"]) for r in rows])) if rows else None,
    }
    for source in ("triviaqa", "simpleqa"):
        source_rows = [r for r in rows if r["source"] == source]
        rs = [r for r in source_rows if r.get("correct") is not None]
        if not rs:
            summary[source] = {"n": 0, "non_answer": non_answer_summary([r["non_answer"] for r in source_rows])}
            continue
        attempted = [r for r in rs if r["attempted"]]
        answered = [r for r in rs if r["answer"]]
        entry = {
            "n": len(rs),
            "accuracy": float(np.mean([r["correct"] for r in rs])),
            "accuracy_parsed": float(np.mean([r["correct"] for r in answered])) if answered else None,
            "unparsed": 1 - len(answered) / len(rs),
            "non_answer": non_answer_summary([r["non_answer"] for r in source_rows]),
            "attempted": float(np.mean([r["attempted"] for r in rs])),
            "correct_given_attempted": float(np.mean([r["correct"] for r in attempted])) if attempted else None,
            "ungraded": float(np.mean([r.get("correct") is None and not r.get("judge_blocked") for r in source_rows])),
            "judge_blocked": float(np.mean([bool(r.get("judge_blocked")) for r in source_rows])),
        }
        if source == "simpleqa":
            entry["f1"] = _f1(entry["accuracy"], entry["correct_given_attempted"])
            fixed = _grade_block(source_rows, "grade_fixed")
            entry.update({f"{k}_fixed": v for k, v in fixed.items()})
            entry["grade_agreement_fixed"] = (
                float(np.mean([r["grade"] == r["grade_fixed"] for r in rs if r["grade_fixed"] is not None]))
                if any(r["grade_fixed"] is not None for r in rs)
                else None
            )
        else:
            entry["f1"] = float(np.mean([r["f1"] for r in rs]))
        with_conf = [r for r in rs if r["confidence"] is not None]
        entry["stated_confidence"] = calibration_summary(
            [r["confidence"] for r in with_conf], [r["correct"] for r in with_conf]
        )
        judged_correct = {
            id(r): (r["judge_rescore"] == "CORRECT" if source == "triviaqa" else r["correct"])
            for r in rs
            if source != "triviaqa" or r.get("judge_rescore") is not None
        }
        with_judged = [r for r in rs if r["confidence_judged"] is not None and id(r) in judged_correct]
        entry["stated_confidence_judged"] = calibration_summary(
            [r["confidence_judged"] for r in with_judged], [judged_correct[id(r)] for r in with_judged]
        )
        if source == "triviaqa":
            entry["calibration_exact_match"] = calibration_summary(
                [r["confidence_judged"] for r in with_judged], [r["correct"] for r in with_judged]
            )
            graded = [r for r in rs if r.get("judge_rescore") is not None]
            entry["accuracy_judged"] = (
                float(np.mean([r["judge_rescore"] == "CORRECT" for r in graded])) if graded else None
            )
            graded_fixed = [r for r in rs if r.get("judge_rescore_fixed") is not None]
            entry["accuracy_judged_fixed"] = (
                float(np.mean([r["judge_rescore_fixed"] == "CORRECT" for r in graded_fixed])) if graded_fixed else None
            )
        entry["confidence_parsed"] = len(with_conf) / len(rs)
        entry["confidence_judged_read"] = float(np.mean([r["confidence_judged"] is not None for r in rs]))
        rescored = [r for r in rs if r.get("judge_rescore") is not None]
        if rescored:
            entry["judge_rescore"] = {
                "n": len(rescored),
                "accuracy": float(np.mean([r["judge_rescore"] == "CORRECT" for r in rescored])),
                "not_attempted": float(np.mean([r["judge_rescore"] == "NOT_ATTEMPTED" for r in rescored])),
            }
        summary[source] = entry
    return Result(summary=summary, rows=rows)
