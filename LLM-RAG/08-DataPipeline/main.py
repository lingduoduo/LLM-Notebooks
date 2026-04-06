import argparse
import json
import os
import re
from typing import Any, Dict, List

import requests
import yaml

from metadata_module import bind_metadata as legacy_bind_metadata
from ocr_module import OCREngine as LegacyOCREngine


def load_config() -> Dict[str, Any]:
    """Load the pipeline configuration from config.yaml."""
    config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def is_rule_enabled(process_rules: Dict[str, Any], rule_id: str, default: bool = False) -> bool:
    """Read preprocessing flags from either list-based or dict-based config formats."""
    pre_processing = process_rules.get("pre_processing", [])

    if isinstance(pre_processing, list):
        for rule in pre_processing:
            if rule.get("id") == rule_id:
                return rule.get("enabled", default)

    return process_rules.get(rule_id, default)


class OCREngine:
    """Extract text through an OCR API, or fall back to the local PaddleOCR module."""

    def __init__(self, config: Dict[str, Any]):
        self.ocr_config = config.get("ocr", {})
        self.api_url = self.ocr_config.get("api_url")
        self.timeout = self.ocr_config.get("timeout", 60)
        self.legacy_engine = None if self.api_url else LegacyOCREngine()

    def extract_text(self, file_path: str) -> str:
        """Extract plain text from a document."""
        if self.api_url:
            with open(file_path, "rb") as file:
                response = requests.post(
                    self.api_url,
                    files={"file": file},
                    timeout=self.timeout,
                )

            if response.status_code != 200:
                raise RuntimeError(
                    f"OCR extraction failed: {response.status_code}, {response.text}"
                )

            data = response.json()
            if "text" in data:
                return str(data["text"])
            if "data" in data and "text" in data["data"]:
                return str(data["data"]["text"])

            raise RuntimeError(f"Unsupported OCR response format: {data}")

        result = self.legacy_engine.extract_text(file_path)
        if isinstance(result, dict):
            return " ".join(str(item) for item in result.get("rec_texts", []) if str(item).strip())
        return str(result)


class RemoveExtraSpaces:
    def __init__(self, enabled: bool = True):
        self.enabled = enabled

    def apply(self, text: str) -> str:
        if not self.enabled:
            return text
        return re.sub(r"\s+", " ", text).strip()


class RemoveURLs:
    def __init__(self, enabled: bool = True):
        self.enabled = enabled

    def apply(self, text: str) -> str:
        if not self.enabled:
            return text
        return re.sub(r"https?://\S+|www\.\S+|\w+@\w+\.\w+", "", text)


class RemoveSpecialChars:
    def __init__(self, enabled: bool = False):
        self.enabled = enabled

    def apply(self, text: str) -> str:
        if not self.enabled:
            return text
        return re.sub(r"[^\w\s\u4e00-\u9fff]", "", text)


class TextProcessor:
    """Preprocess and split OCR text before uploading it to Dify."""

    def __init__(self, config: Dict[str, Any]):
        self.process_rules = config["dify"]["process_rules"]
        self.plugins = [
            RemoveExtraSpaces(
                enabled=is_rule_enabled(self.process_rules, "remove_extra_spaces", True)
            ),
            RemoveURLs(
                enabled=is_rule_enabled(self.process_rules, "remove_urls_emails", True)
                or is_rule_enabled(self.process_rules, "remove_urls", True)
            ),
            RemoveSpecialChars(
                enabled=is_rule_enabled(self.process_rules, "remove_special_chars", False)
            ),
        ]

    def preprocess(self, text: str) -> str:
        for plugin in self.plugins:
            text = plugin.apply(text)
        return text

    def segment(self, text: str) -> List[str]:
        segmentation = self.process_rules.get("segmentation", {})
        separator = segmentation.get("separator", r"\n+")
        parts = [part.strip() for part in re.split(separator, text) if part and part.strip()]

        max_length = segmentation.get("max_length") or segmentation.get("max_tokens", 1000)
        segments: List[str] = []

        for part in parts:
            if len(part) <= max_length:
                segments.append(part)
                continue

            for index in range(0, len(part), max_length):
                chunk = part[index:index + max_length].strip()
                if chunk:
                    segments.append(chunk)

        return segments


class DifyUploader:
    """Upload processed text segments to a Dify dataset."""

    def __init__(self, config: Dict[str, Any]):
        dify_config = config["dify"]
        self.api_key = dify_config["api_key"]
        self.dataset_id = dify_config["dataset_id"]
        self.base_url = f"https://api.dify.ai/v1/datasets/{self.dataset_id}"
        self.process_rules = dify_config["process_rules"]
        self.indexing_technique = dify_config.get("indexing_technique", "high_quality")
        self.timeout = dify_config.get("timeout", 60)

    def upload_by_text(self, segments: List[str]) -> List[str]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        document_ids: List[str] = []

        for index, content in enumerate(segments):
            payload = {
                "name": f"segment_{index}.txt",
                "text": content,
                "indexing_technique": self.indexing_technique,
                "process_rule": {"mode": "automatic"},
                "doc_form": "text_model",
                "doc_language": "English",
            }

            response = requests.post(
                f"{self.base_url}/document/create-by-text",
                headers=headers,
                data=json.dumps(payload),
                timeout=self.timeout,
            )
            print(f"Segment {index} upload status: {response.status_code}")

            if response.status_code not in (200, 201):
                print(f"Upload failed for segment {index}: {response.text}")
                continue

            data = response.json()
            document_id = data.get("document", {}).get("id")
            if document_id:
                document_ids.append(document_id)

        return document_ids


def build_metadata(config: Dict[str, Any], file_path: str) -> List[Dict[str, str]]:
    """Build metadata values from the configured meta fields."""
    metadata_config = config.get("meta") or config.get("metadata") or {}
    basename = os.path.basename(file_path)
    metadata: List[Dict[str, str]] = []

    for field in metadata_config.get("fields", []):
        value = field.get("value", "")
        if field.get("value_from") == "filename":
            value = basename
        metadata.append({"name": field["name"], "value": value})

    return metadata


def run_pipeline(file_path: str, debug: bool = False) -> List[str]:
    """Run the full OCR to Dify pipeline for a single file."""
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"Input file not found: {file_path}")

    config = load_config()

    ocr_engine = OCREngine(config)
    raw_text = ocr_engine.extract_text(file_path)
    print(f"OCR text length: {len(raw_text)}")
    if debug:
        print(f"[DEBUG] OCR text preview: {raw_text[:500]}")

    processor = TextProcessor(config)
    cleaned_text = processor.preprocess(raw_text)
    segments = processor.segment(cleaned_text)
    print(f"Generated segments: {len(segments)}")
    if debug:
        print(f"[DEBUG] Segments: {segments}")

    if not segments:
        raise RuntimeError("No text segments were generated after preprocessing.")

    uploader = DifyUploader(config)
    document_ids = uploader.upload_by_text(segments)
    print(f"Uploaded documents: {document_ids}")

    if document_ids:
        metadata = build_metadata(config, file_path)
        for document_id in document_ids:
            legacy_bind_metadata(document_id, metadata)
        if debug:
            print(f"[DEBUG] Metadata: {metadata}")

    return document_ids


def parse_args() -> argparse.Namespace:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    default_file = os.path.join(base_dir, "example.png")

    parser = argparse.ArgumentParser(description="Run the OCR to Dify pipeline.")
    parser.add_argument(
        "--file",
        default=default_file,
        help="Path to the input image or PDF file.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print intermediate debugging information.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_pipeline(args.file, debug=args.debug)


if __name__ == "__main__":
    main()
