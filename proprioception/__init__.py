import importlib

__all__ = [
    "ConversationRouting",
    "RoutingOracle",
    "find_word_token_index",
    "ProprioceptionConfig",
    "ProprioceptionData",
    "ProprioceptionTask",
    "ProprioceptionTaskset",
    "ProprioceptionUser",
    "ProprioceptionUserConfig",
]

_HOMES = dict.fromkeys(__all__, "proprioception.taskset")
_HOMES.update(dict.fromkeys(("ConversationRouting", "RoutingOracle", "find_word_token_index"), "proprioception.oracle"))


def __getattr__(name: str):
    home = _HOMES.get(name)
    if home is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(importlib.import_module(home), name)
