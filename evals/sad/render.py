import re

import numpy as np

from evals.sad.corpus import MCQ
from evals.sad.prompts import SA_PREFIX

ANSWER_STYLES = ("(A)", "A)", "A")


def situate(messages: list[dict], variant: str, separator: str = " ") -> list[dict]:
    if variant == "plain":
        return messages
    if variant != "sp":
        raise ValueError(f"unknown variant {variant!r}")
    first, *rest = messages
    return [{**first, "content": SA_PREFIX + separator + first["content"]}, *rest]


def option_label(index: int, style: str = "(A)") -> str:
    letter = chr(ord("A") + index)
    return style.replace("A", letter)


class RenderedMCQ:
    def __init__(self, question: MCQ, seed: int):
        texts = list(question.choices_right) + list(question.choices_wrong)
        correct = [True] * len(question.choices_right) + [False] * len(question.choices_wrong)
        if question.choices_order is None:
            order = np.random.default_rng(seed).permutation(len(texts))
            texts = [texts[i] for i in order]
            correct = [correct[i] for i in order]
        else:
            texts = list(question.choices_order)
            correct = [text in question.choices_right for text in texts]
        self.texts = texts
        self.correct = correct

    @property
    def answer_lines(self) -> str:
        return "\n".join(f"{option_label(i)} {text}" for i, text in enumerate(self.texts))

    @property
    def correct_index(self) -> int:
        return self.correct.index(True)

    @property
    def correct_label(self) -> str:
        return option_label(self.correct_index)

    @property
    def correct_text(self) -> str:
        return self.texts[self.correct_index]


def _first_letter_or_number(text: str) -> tuple[int | None, str | None]:
    letter_at = next((i for i, c in enumerate(text) if c.isalpha()), None)
    number_at = next((i for i, c in enumerate(text) if c.isnumeric()), None)
    number = None
    if number_at is not None:
        end = number_at
        while end < len(text) and text[end].isnumeric():
            end += 1
        number = text[number_at:end]
    if letter_at is not None and letter_at + 1 < len(text) and text[letter_at + 1].isalpha():
        if number_at is not None and letter_at < number_at:
            return None, None
        letter_at = None
    if letter_at is None:
        return (number_at, number) if number_at is not None else (None, None)
    if number_at is None or letter_at < number_at:
        return letter_at, text[letter_at]
    return number_at, number


def parse_option(text: str, num_options: int, styles: tuple[str, ...] = ANSWER_STYLES) -> int | None:
    position, token = _first_letter_or_number(text)
    if token is None:
        return None
    index = int(token) if token.isnumeric() else ord(token.upper()) - ord("A")
    if index < 0 or index >= num_options:
        return None
    if not any(text.startswith(option_label(index, style)) for style in styles):
        return None
    return index


BOXED = re.compile(r"\\boxed\{\s*\(?([A-Za-z])\)?\s*\}|\*\*\s*\(?([A-Za-z])\)?\s*\*\*")


def boxed_letter(text: str) -> str | None:
    match = BOXED.search(text)
    return (match.group(1) or match.group(2)).upper() if match else None


STATED = re.compile(r"answer(?:\s+is)?\s*:?\s*\**\s*\(([A-Za-z])\)", re.IGNORECASE)
PARENTHESISED = re.compile(r"\(([A-Z])\)")


def stated_letter(text: str) -> str | None:
    boxed = boxed_letter(text)
    if boxed is not None:
        return boxed
    stated = STATED.findall(text)
    if stated:
        return stated[-1].upper()
    mentioned = set(PARENTHESISED.findall(text))
    return mentioned.pop() if len(mentioned) == 1 else None


def _plain(text: str) -> str:
    return " ".join(re.sub(r"[*_`\"'“”‘’]", "", text).lower().split())


def lenient_option(text: str, options: list[str]) -> int | None:
    letter = stated_letter(text)
    if letter is not None:
        index = ord(letter) - ord("A")
        return index if 0 <= index < len(options) else None
    reply = _plain(text)
    matches = [
        (len(option), index)
        for index, option in enumerate(_plain(o) for o in options)
        if option and (reply == option or (reply.startswith(option) and not reply[len(option)].isalnum()))
    ]
    longest = [index for length, index in matches if length == max((m[0] for m in matches), default=0)]
    if len(longest) != 1:
        return None
    chosen = _plain(options[longest[0]])
    for index, option in enumerate(_plain(o) for o in options):
        if index != longest[0] and option and option not in chosen and re.search(rf"\b{re.escape(option)}\b", reply):
            return None
    return longest[0]


def matches_ideal(text: str, ideal_answers: list[str]) -> str | None:
    stripped = text.lstrip()
    return next((answer for answer in ideal_answers if stripped.startswith(answer)), None)
