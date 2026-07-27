#!/usr/bin/env python3
"""Unified command-line interface for the Perception Tools MCP server."""

import argparse
import asyncio
import importlib
import inspect
import json
import logging
import tempfile
import types
import typing
from pathlib import Path


CATEGORIES = {
    "search": "Search",
    "multimodal": "Multimodal",
    "filesystem": "File System",
    "public": "Public Data",
    "private": "Private Data",
}


class Tool(typing.NamedTuple):
    """A perception tool exposed through the CLI and MCP server."""

    name: str
    category: str
    module: str
    func: str
    desc: str
    online: bool = False
    note: str = ""


TOOLS: list[Tool] = [
    Tool("web_search", "search", "search_tools", "search_web",
         "Search the web with DuckDuckGo", True),
    Tool("knowledge_base_search", "search", "search_tools", "search_knowledge_base",
         "Search a local knowledge-base directory"),
    Tool("download", "search", "search_tools", "download_file",
         "Download a URL with size and overwrite safeguards", True),
    Tool("google_search_enhanced", "search", "google_search_enhanced", "google_search_api",
         "Use Google Custom Search with DuckDuckGo fallback", True,
         "Google search uses GOOGLE_API_KEY and GOOGLE_CSE_ID when configured"),
    Tool("webpage_reader", "multimodal", "multimodal_tools", "read_webpage",
         "Extract text and links from a webpage", True),
    Tool("webpage_read_enhanced", "multimodal", "google_search_enhanced",
         "read_webpage_content", "Extract cleaned webpage content", True),
    Tool("document_reader", "multimodal", "multimodal_tools", "read_document",
         "Read PDF, DOCX, and PPTX documents"),
    Tool("pdf_extract", "multimodal", "document_processing_tools", "extract_pdf_text",
         "Extract text from selected PDF pages"),
    Tool("docx_extract", "multimodal", "document_processing_tools", "extract_docx_content",
         "Extract Word document content"),
    Tool("pptx_extract", "multimodal", "document_processing_tools", "extract_pptx_content",
         "Extract PowerPoint content"),
    Tool("csv_parse", "multimodal", "document_processing_tools", "extract_csv_content",
         "Parse CSV data"),
    Tool("image_parser", "multimodal", "multimodal_tools", "parse_image",
         "Read image metadata with optional AI analysis", False,
         "AI analysis requires OPENAI_API_KEY"),
    Tool("image_ocr", "multimodal", "media_processing_tools", "extract_text_ocr",
         "Extract image text with OCR", False, "Requires Tesseract"),
    Tool("image_analyze", "multimodal", "media_processing_tools", "analyze_image_ai",
         "Analyze an image with an OpenAI vision model", False,
         "Requires OPENAI_API_KEY"),
    Tool("image_metadata", "multimodal", "media_processing_tools", "get_image_metadata",
         "Read image metadata"),
    Tool("video_parser", "multimodal", "multimodal_tools", "parse_video",
         "Read video metadata and sample frames"),
    Tool("video_keyframes", "multimodal", "media_processing_tools",
         "extract_video_keyframes", "Extract keyframes from a video"),
    Tool("video_analyze", "multimodal", "media_processing_tools", "analyze_video_ai",
         "Analyze video keyframes with an OpenAI vision model", False,
         "Requires OPENAI_API_KEY"),
    Tool("audio_transcribe", "multimodal", "media_processing_tools",
         "transcribe_audio_whisper", "Transcribe audio with local Whisper or OpenAI",
         False, "Local mode requires Whisper; API fallback requires OPENAI_API_KEY"),
    Tool("audio_metadata", "multimodal", "media_processing_tools", "extract_audio_metadata",
         "Read audio metadata"),
    Tool("audio_trim", "multimodal", "media_processing_tools", "trim_audio",
         "Trim audio to a time range"),
    Tool("youtube_transcript", "multimodal", "multimodal_tools",
         "extract_youtube_transcript", "Retrieve a YouTube transcript", True),
    Tool("youtube_download", "multimodal", "multimodal_tools", "download_youtube_video",
         "Download a YouTube video", True),
    Tool("file_reader", "filesystem", "filesystem_tools", "read_file",
         "Read a local file with encoding and length controls"),
    Tool("grep", "filesystem", "filesystem_tools", "grep_search",
         "Search file contents with a regular expression"),
    Tool("text_summarizer", "filesystem", "filesystem_tools", "summarize_text",
         "Summarize or truncate long text"),
    Tool("weather", "public", "public_data_tools", "get_weather",
         "Get current weather from Open-Meteo", True),
    Tool("stock_price", "public", "public_data_tools", "get_stock_price",
         "Get a stock quote from Yahoo Finance", True),
    Tool("crypto_price", "public", "public_data_tools", "get_crypto_price",
         "Get a cryptocurrency price from CoinGecko", True),
    Tool("currency_converter", "public", "public_data_tools", "convert_currency",
         "Convert currencies with a public exchange-rate service", True),
    Tool("wikipedia_search", "public", "public_data_tools", "search_wikipedia",
         "Search Wikipedia and return a summary", True),
    Tool("arxiv_search", "public", "public_data_tools", "search_arxiv",
         "Search ArXiv papers", True),
    Tool("wayback_search", "public", "public_data_tools", "search_wayback",
         "Find Internet Archive snapshots", True),
    Tool("location_search", "public", "public_data_tools", "search_location",
         "Geocode a location with Nominatim", True),
    Tool("poi_search", "public", "public_data_tools", "search_poi",
         "Find nearby points of interest with Overpass", True),
    Tool("yfinance_quote", "public", "yahoo_finance_tools", "get_stock_quote",
         "Get a detailed Yahoo Finance quote", True),
    Tool("yfinance_historical", "public", "yahoo_finance_tools", "get_historical_data",
         "Get historical market data", True),
    Tool("yfinance_company_info", "public", "yahoo_finance_tools", "get_company_info",
         "Get company profile information", True),
    Tool("yfinance_financials", "public", "yahoo_finance_tools",
         "get_financial_statements", "Get company financial statements", True),
    Tool("pubchem_search", "public", "pubchem_tools", "search_compounds",
         "Search PubChem compounds", True),
    Tool("pubchem_properties", "public", "pubchem_tools", "get_compound_properties",
         "Get PubChem compound properties", True),
    Tool("pubchem_synonyms", "public", "pubchem_tools", "get_compound_synonyms",
         "Get PubChem compound synonyms", True),
    Tool("pubchem_similar", "public", "pubchem_tools", "search_similar_compounds",
         "Search structurally similar PubChem compounds", True),
    Tool("wiki_article_full", "public", "wiki_enhanced", "get_article_content",
         "Get full Wikipedia article content", True),
    Tool("wiki_article_categories", "public", "wiki_enhanced", "get_article_categories",
         "Get Wikipedia article categories", True),
    Tool("wiki_article_links", "public", "wiki_enhanced", "get_article_links",
         "Get links from a Wikipedia article", True),
    Tool("wiki_article_history", "public", "wiki_enhanced", "get_article_history",
         "Get a historical Wikipedia revision", True),
    Tool("arxiv_paper_details", "public", "arxiv_enhanced", "get_paper_details",
         "Get detailed ArXiv paper metadata", True),
    Tool("arxiv_download", "public", "arxiv_enhanced", "download_paper",
         "Download an ArXiv PDF", True),
    Tool("arxiv_categories", "public", "arxiv_enhanced", "get_arxiv_categories",
         "List ArXiv subject categories", True),
    Tool("wayback_archived_content", "public", "wayback_enhanced",
         "get_archived_content", "Retrieve archived webpage content", True),
    Tool("calendar_events", "private", "private_data_tools", "get_calendar_events",
         "Read Google Calendar events", True, "Requires Google OAuth"),
    Tool("notion_search", "private", "private_data_tools", "search_notion",
         "Search a Notion workspace", True, "Requires NOTION_API_KEY"),
]

TOOLS_BY_NAME = {tool.name: tool for tool in TOOLS}


def _load_callable(tool: Tool):
    """Lazy-load the callable for a registry entry."""
    module = importlib.import_module(f".{tool.module}", package=__package__)
    return getattr(module, tool.func)


def _coerce(value: str, annotation):
    """Convert a command-line string using a callable annotation."""
    origin = typing.get_origin(annotation)
    if origin in (typing.Union, types.UnionType):
        args = [item for item in typing.get_args(annotation) if item is not type(None)]
        annotation = args[0] if args else str
        origin = typing.get_origin(annotation)

    if annotation is bool:
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    if annotation in (int, float, str):
        return annotation(value)
    if origin in (list, tuple):
        values = [item.strip() for item in value.split(",") if item.strip()]
        return values if origin is list else tuple(values)
    return value


def _parse_params(function, pairs: list[str]) -> dict[str, typing.Any]:
    """Parse key=value pairs for a tool callable."""
    signature = inspect.signature(function)
    kwargs: dict[str, typing.Any] = {}
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"Expected key=value, received {pair!r}")
        key, value = pair.split("=", 1)
        if key not in signature.parameters:
            valid = ", ".join(signature.parameters)
            raise ValueError(f"Unknown parameter {key!r}. Available parameters: {valid}")
        kwargs[key] = _coerce(value, signature.parameters[key].annotation)
    return kwargs


def _unwrap(result) -> dict[str, typing.Any]:
    """Normalize TextContent, JSON strings, and dictionaries."""
    if isinstance(result, dict):
        return result
    payload = getattr(result, "text", result)
    if isinstance(payload, str):
        parsed = json.loads(payload)
        if isinstance(parsed, dict):
            return parsed
    raise TypeError(f"Unsupported tool response: {type(result).__name__}")


def cmd_list(args: argparse.Namespace) -> int:
    """List registered tools by category."""
    selected = [args.category] if args.category else list(CATEGORIES)
    print(f"\nPerception Tools MCP Server ({len(TOOLS)} tools)")
    print("=" * 72)
    for category in selected:
        tools = [tool for tool in TOOLS if tool.category == category]
        print(f"\n{CATEGORIES[category]} ({len(tools)} tools)")
        for tool in tools:
            flags = []
            if tool.online:
                flags.append("network")
            if tool.note:
                flags.append(tool.note)
            suffix = f" [{' | '.join(flags)}]" if flags else ""
            print(f"  {tool.name:<28} {tool.desc}{suffix}")
    print("\nUse `perception-tools info <tool>` for parameters.")
    return 0


def _example_for(function) -> str:
    values = {
        "query": "example",
        "pattern": "async",
        "directory": ".",
        "file_path": "document.txt",
        "location": "New York",
        "symbol": "AAPL",
        "url": "https://example.com",
    }
    parts = []
    for name, parameter in inspect.signature(function).parameters.items():
        if parameter.default is inspect.Parameter.empty:
            parts.append(f"{name}={values.get(name, 'value')}")
    return " ".join(parts)


def cmd_info(args: argparse.Namespace) -> int:
    """Show details for one tool."""
    tool = TOOLS_BY_NAME.get(args.tool)
    if not tool:
        print(f"Unknown tool: {args.tool}", file=__import__("sys").stderr)
        return 2
    try:
        function = _load_callable(tool)
    except ImportError as exc:
        print(f"Cannot load {tool.name}: {exc}", file=__import__("sys").stderr)
        return 1

    print(f"\nTool: {tool.name}")
    print(f"Category: {CATEGORIES[tool.category]}")
    print(f"Description: {tool.desc}")
    print(f"Implementation: perception_tools.{tool.module}:{tool.func}")
    print(f"Network required: {'yes' if tool.online else 'no'}")
    if tool.note:
        print(f"Note: {tool.note}")
    print("\nParameters:")
    for name, parameter in inspect.signature(function).parameters.items():
        default = (
            "required"
            if parameter.default is inspect.Parameter.empty
            else f"default={parameter.default!r}"
        )
        print(f"  {name}: {parameter.annotation!s} ({default})")
    print(f"\nExample: perception-tools run {tool.name} {_example_for(function)}".rstrip())
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    """Invoke one registered tool."""
    tool = TOOLS_BY_NAME.get(args.tool)
    if not tool:
        print(f"Unknown tool: {args.tool}", file=__import__("sys").stderr)
        return 2
    try:
        function = _load_callable(tool)
        kwargs = _parse_params(function, args.params)
        result = asyncio.run(function(**kwargs))
        print(json.dumps(_unwrap(result), indent=2, ensure_ascii=False))
        return 0
    except (ImportError, TypeError, ValueError) as exc:
        print(f"Tool invocation failed: {exc}", file=__import__("sys").stderr)
        return 2
    except Exception as exc:
        print(f"Tool execution failed: {type(exc).__name__}: {exc}",
              file=__import__("sys").stderr)
        return 1


async def _demo(offline: bool) -> None:
    """Run a research-assistant perception flow."""
    from .filesystem_tools import grep_search, read_file
    from .search_tools import search_knowledge_base

    print("\nPerception Tools end-to-end demo")
    print("Scenario: inspect local research notes, then enrich them with public data.")
    if offline:
        print("Mode: offline; network steps are skipped.")

    with tempfile.TemporaryDirectory(prefix="perception-tools-") as temp_dir:
        root = Path(temp_dir)
        notes = root / "notes.md"
        notes.write_text(
            "# MCP research notes\n\n"
            "Model Context Protocol standardizes exchanges between agents, tools, "
            "and data sources. Perception tools give an agent controlled access to "
            "external information.\n",
            encoding="utf-8",
        )
        budget = root / "budget.md"
        budget.write_text(
            "# Budget\n\nThe research budget is 200 USD.\n",
            encoding="utf-8",
        )

        print("\n[1/3] File-system perception")
        grep = _unwrap(await grep_search("Protocol", temp_dir, "*.md"))
        print(f"  grep found {grep.get('total_found', 0)} matches")
        read = _unwrap(await read_file(str(notes), max_length=200))
        preview = str(read.get("message", "")).splitlines()[0]
        print(f"  first line: {preview}")

        print("\n[2/3] Local knowledge-base search")
        search = _unwrap(await search_knowledge_base("MCP", temp_dir, top_k=3))
        total = search.get("message", {}).get("total_found", 0)
        print(f"  knowledge-base search found {total} files")

        print("\n[3/3] Public data")
        if offline:
            print("  skipped in offline mode")
        else:
            from .public_data_tools import convert_currency, search_wikipedia

            currency = _unwrap(await convert_currency(200, "USD", "EUR"))
            print(f"  currency conversion success: {currency.get('success', False)}")
            wiki = _unwrap(await search_wikipedia("Model Context Protocol", sentences=2))
            print(f"  Wikipedia search success: {wiki.get('success', False)}")

    print("\nDemo complete.")


def cmd_demo(args: argparse.Namespace) -> int:
    """Run the demo command."""
    asyncio.run(_demo(offline=args.offline))
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line parser."""
    parser = argparse.ArgumentParser(
        prog="perception-tools",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "Perception Tools MCP Server\n"
            "Tools for Search, Multimodal processing, File System access, "
            "Public Data, and Private Data."
        ),
        epilog=(
            "Examples:\n"
            "  perception-tools list\n"
            "  perception-tools list --category filesystem\n"
            "  perception-tools info weather\n"
            "  perception-tools run grep pattern=async directory=.\n"
            "  perception-tools demo --offline"
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True, metavar="<command>")

    list_parser = subparsers.add_parser("list", help="List registered tools")
    list_parser.add_argument("--category", choices=list(CATEGORIES),
                             help="Only list one tool category")
    list_parser.set_defaults(handler=cmd_list)

    info_parser = subparsers.add_parser("info", help="Show a tool signature and example")
    info_parser.add_argument("tool", help="Tool name from the list command")
    info_parser.set_defaults(handler=cmd_info)

    run_parser = subparsers.add_parser("run", help="Invoke a tool and print JSON")
    run_parser.add_argument("tool", help="Tool name from the list command")
    run_parser.add_argument("params", nargs="*", metavar="key=value",
                            help="Tool arguments in key=value form")
    run_parser.set_defaults(handler=cmd_run)

    demo_parser = subparsers.add_parser("demo", help="Run an end-to-end demo")
    demo_parser.add_argument("--offline", action="store_true",
                             help="Skip all network operations")
    demo_parser.set_defaults(handler=cmd_demo)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the command-line interface."""
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
    args = build_parser().parse_args(argv)
    return int(args.handler(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
