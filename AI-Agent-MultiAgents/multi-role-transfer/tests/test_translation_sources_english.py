"""English source-reference entries preserve the original lookup IDs."""

import json

import fill_translation_map as fill
import translate_validation as tv


def test_source_reference_contains_english_translations():
    sources = tv.load_sources()
    translations = tv.load_translations()
    assert set(sources) == set(translations)
    assert all(sources[key] == translations[key] for key in sources)
    assert not any(tv.CJK.search(value) for value in sources.values())


def test_saving_a_translation_keeps_source_reference_in_english(tmp_path, monkeypatch):
    map_file = tmp_path / "translation_map.json"
    sources_file = tmp_path / "translation_sources.json"
    map_file.write_text('{"schema_version": 1, "translations": {}}')
    monkeypatch.setattr(fill, "MAP_FILE", map_file)
    monkeypatch.setattr(fill, "SOURCES_FILE", sources_file)
    original = "第一段"
    key = tv.source_key(original)
    fill.save({key: "First paragraph"}, {key: original})
    saved = json.loads(sources_file.read_text())
    assert saved["sources"][key] == "First paragraph"
    assert "English" in saved["note"]
