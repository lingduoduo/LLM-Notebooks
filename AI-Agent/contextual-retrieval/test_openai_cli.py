from pathlib import Path


ROOT = Path(__file__).parent


def test_cli_surfaces_do_not_offer_provider_selection():
    files = [
        ROOT / "main.py",
        ROOT / "index_local_laws_contextual.py",
        ROOT / "evaluation" / "evaluate.py",
    ]
    source = "\n".join(path.read_text(encoding="utf-8") for path in files)
    assert "--provider" not in source
    assert "--llm-provider" not in source
    assert ".llm.provider" not in source


def test_evaluation_records_model_without_provider():
    source = (ROOT / "evaluation" / "evaluate.py").read_text(encoding="utf-8")
    assert '"llm_model": self.agent.model' in source
    assert '"llm_provider"' not in source
