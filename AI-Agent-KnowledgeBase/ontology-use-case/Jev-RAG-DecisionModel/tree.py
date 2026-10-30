from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Node:
    id: str
    description: str
    children: tuple[Node, ...] = ()
    payload: Mapping = field(default_factory=dict)

    @property
    def is_leaf(self) -> bool:
        return not self.children


def walk(root: Node) -> Iterator[Node]:
    """Pre-order traversal, root excluded."""
    for child in root.children:
        yield child
        yield from walk(child)
