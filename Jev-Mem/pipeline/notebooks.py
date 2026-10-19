"""Shared notebook wiring; notebook code delegates to the tested pipeline."""

from dataclasses import dataclass
from types import SimpleNamespace

from .cli import _providers
from .experiment import run_experiment


@dataclass
class Session:
    mode: str
    encoder: object
    decisions: object
    answerer: object
    judge: object
    manifest: dict

    def run(self, dataset, output):
        return run_experiment(
            dataset,
            output,
            mode=self.mode,
            encoder=self.encoder,
            decisions=self.decisions,
            answerer=self.answerer,
            judge=self.judge,
            manifest=self.manifest,
        )

    def close(self):
        if hasattr(self.decisions, "close"):
            self.decisions.close()


def make_session(mode="offline", embedding=None, decisions=None):
    options = SimpleNamespace(
        mode=mode,
        embedding=embedding,
        decisions=decisions,
        answer_model=None,
        answer_base_url=None,
        judge_model=None,
        judge_base_url=None,
        embedding_model=None,
        embedding_base_url=None,
    )
    return Session(mode, *_providers(options))
