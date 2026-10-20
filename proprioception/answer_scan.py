import re

DASHES = "-‐‑‒–—−"
QUOTES = "'\"“”‘’`"
LAYER_WORDS = {"layer", "layers", "depth", "level", "l"}
RANK_WORDS = {"top", "rank", "ranked"}
ANSWER_WORDS = {"expert", "experts", "id", "answer", "number", "index", "is", "was"}
SHORT_REPLY_CHARS = 80
_LEAD = r"[*`\"'“”‘’(\[]*\s*#?\s*"
_END = r"(?![\w.%])"
_MARKED = [
    re.compile(r"\\boxed\{\s*(\d+)\s*\}"),
    re.compile(r"\bexpert(?:[\s_-]*(?:id|number|index|no\.?))?[\s_-]*[:#=]?\s*" + _LEAD + r"(\d+)" + _END, re.I),
    re.compile(r"\bid_(\d+)" + _END, re.I),
    re.compile(r"\banswer\s*(?:is|:|=)\s*" + _LEAD + r"(\d+)" + _END, re.I),
    re.compile(r"\bid\s*[:=]\s*" + _LEAD + r"(\d+)" + _END, re.I),
    re.compile(r"(?m)^[ \t#>*`\"'“”‘’]*(\d+)[ \t*`\"'“”‘’.]*$"),
    re.compile(
        r"\b(?:expert|answer|id|choice|pick|winner|result|leader)s?\b[^\n.!?]{0,40}?\b(?:is|was|would be)\s*[:=]?\s*"
        + _LEAD
        + r"(\d+)"
        + _END,
        re.I,
    ),
    re.compile(
        r"\b(?:expert|answer|id|choice|pick|winner|result|leader)s?\b[^\n.!?]{0,40}?(?:[:=]|->|→)\s*"
        + _LEAD
        + r"(\d+)"
        + _END,
        re.I,
    ),
    re.compile(r"\*\*\s*#?\s*(\d+)\s*\*\*"),
]


def _words_before(text: str, position: int, count: int = 2) -> list[str]:
    return [w.lower() for w in re.findall(r"[A-Za-z]+|\d+", text[max(0, position - 60) : position])[-count:]]


def _probed_token(question: str) -> str | None:
    match = re.search(r"token\s*['\"“‘](.*?)['\"”’]", question)
    return None if match is None else match.group(1).strip()


def excluded(reply: str, start: int, end: int, question: str) -> bool:
    before, after = reply[start - 1 : start] if start else "", reply[end : end + 1]
    if re.match(r"[A-Za-z_%]", after):
        return True
    expert_prefix = re.search(r"(?:expert|id)[\s_" + re.escape(DASHES) + r"]?$", reply[max(0, start - 7) : start], re.I)
    if re.match(r"[A-Za-z_]", before) and not expert_prefix:
        return True
    if (before == "." and start >= 2 and reply[start - 2].isdigit()) or (
        after == "." and reply[end + 1 : end + 2].isdigit()
    ):
        return True
    if (
        before
        and before in DASHES
        and re.search(r"[A-Za-z]$", reply[max(0, start - 2) : start - 1])
        and not expert_prefix
    ):
        return True
    line_start = reply.rfind("\n", 0, start) + 1
    if re.fullmatch(r"[ \t#>*\-]*", reply[line_start:start]) and re.match(r"[.)]\s+\S", reply[end : end + 3]):
        return True
    if (
        reply[:start].count("$") % 2 == 1
        or re.search(r"\\\(|\\\[", reply[max(0, start - 40) : start])
        or re.search(r"<[^>]*$", reply[max(0, start - 80) : start])
    ):
        return True
    if _probed_token(question) == reply[start:end] and (
        re.search(r"token\s*[\"'“‘’]?\s*$", reply[max(0, start - 12) : start]) or (before != "" and before in QUOTES)
    ):
        return True
    previous = _words_before(reply, start)
    if previous and previous[-1] in ANSWER_WORDS:
        return False
    return any(w in LAYER_WORDS for w in previous) or bool(previous and previous[-1] in RANK_WORDS)


def first_strict(reply: str, question: str) -> tuple[int, str] | None:
    for match in re.finditer(r"\d+", reply):
        if not excluded(reply, match.start(), match.end(), question):
            return match.start(), match.group()
    return None


def marked(reply: str, question: str) -> tuple[int, str] | None:
    found = set()
    for pattern in _MARKED:
        for match in pattern.finditer(reply):
            start, end = match.span(1)
            if not excluded(reply, start, end, question):
                found.add((start, match.group(1)))
    if found:
        return max(found)
    if len(reply.strip()) <= SHORT_REPLY_CHARS:
        return first_strict(reply, question)
    return None


READERS = {"first_strict": first_strict, "marked": marked}
