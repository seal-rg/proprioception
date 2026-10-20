import dataclasses
from collections import Counter, defaultdict

import numpy as np


@dataclasses.dataclass
class LookupTable:
    layers: list[int]
    counts: dict[tuple[int, int], Counter] = dataclasses.field(default_factory=lambda: defaultdict(Counter))
    seen: set = dataclasses.field(default_factory=set)
    num_positions: int = 0

    def add(self, key: object, ids: list[int], routed: np.ndarray, end: int) -> None:
        if key in self.seen:
            return
        self.seen.add(key)
        end = min(end, routed.shape[0], len(ids))
        for pos in range(end):
            for layer in self.layers:
                self.counts[(ids[pos], layer)][int(routed[pos, layer, 0])] += 1
        self.num_positions += end

    def modal(self, token_id: int, layer: int, *, min_count: int = 1) -> int | None:
        counter = self.counts.get((token_id, layer))
        if not counter or sum(counter.values()) < min_count:
            return None
        return counter.most_common(1)[0][0]

    def lexicality(self) -> dict[str, float]:
        out = {}
        for layer in self.layers:
            agree = total = 0
            for (_, key_layer), counter in self.counts.items():
                if key_layer == layer:
                    agree += counter.most_common(1)[0][1]
                    total += sum(counter.values())
            out[f"layer={layer}"] = agree / total if total else float("nan")
        return out
