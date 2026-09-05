from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class UserContext:
    recently_watched: List[str]
    topics: List[str]
    followed_creators: List[str]


@dataclass
class Item:
    title: str
    tags: List[str]


@dataclass
class HumanLabel:
    groundedness: int
    relevance: int
    privacy_safety: int
    clarity: int
    rationale: str


@dataclass
class BenchmarkExample:
    id: str
    user: UserContext
    item: Item
    explanation: str
    human: HumanLabel


@dataclass
class JudgeResult:
    groundedness: int
    relevance: int
    privacy_safety: int
    clarity: int
    passed: bool
    unsupported_claims: List[str] = field(default_factory=list)
    rationale: str = ""

    def scores(self) -> Dict[str, int]:
        return {
            "groundedness": self.groundedness,
            "relevance": self.relevance,
            "privacy_safety": self.privacy_safety,
            "clarity": self.clarity,
        }


@dataclass
class GenerationTrace:
    attempt: int
    explanation: str
    judge: JudgeResult


@dataclass
class GenerationOutcome:
    served_text: str
    passed: bool
    traces: List[GenerationTrace]
    used_fallback: bool = False
