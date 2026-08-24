"""Catalog of 120+ cross-domain tool definitions for active discovery.

Each tool has a realistic OpenAI function schema. The catalog spans finance,
news, web, arXiv, files, GitHub, code, maps, weather, media, language, email,
databases, ecommerce, social, crypto, and utilities. It intentionally includes
generic near-synonym tools that compete with specialists under full injection.
Execution is lightweight and mocked because the experiment measures tool choice.
"""

from typing import Dict, List


def _tool(name: str, description: str, params: Dict) -> Dict:
    """Build an OpenAI function-calling tool schema."""
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": params,
                "required": list(params.keys()),
            },
        },
    }


def _s(desc: str) -> Dict:
    return {"type": "string", "description": desc}


def _i(desc: str) -> Dict:
    return {"type": "integer", "description": desc}


# ---------------------------------------------------------------------------
# Tool definitions grouped by domain
# ---------------------------------------------------------------------------

_DEFS: List[Dict] = []

# --- finance (10 specialized tools) ---
_DEFS += [
    _tool("get_stock_price", "Get a stock's live price, percentage change, and volume from a specialized financial source.",
          {"symbol": _s("Ticker symbol, such as AAPL or TSLA")}),
    _tool("get_stock_history", "Get historical candlestick data for a stock.",
          {"symbol": _s("Ticker symbol"), "range": _s("Time range, such as 1mo or 1y")}),
    _tool("get_company_financials", "Get a public company's financial statements, including revenue, profit, and balance sheet.",
          {"symbol": _s("Ticker symbol")}),
    _tool("get_forex_rate", "Get the live foreign-exchange rate between two fiat currencies.",
          {"base": _s("Base currency, such as USD"), "quote": _s("Quote currency, such as JPY")}),
    _tool("get_crypto_price", "Get a cryptocurrency's live USD price.",
          {"symbol": _s("Cryptocurrency symbol, such as BTC or ETH")}),
    _tool("get_market_index", "Get the live level of a stock-market index such as the S&P 500 or Nasdaq.",
          {"index": _s("Index symbol, such as SPX or IXIC")}),
    _tool("get_earnings_calendar", "Get a company's earnings calendar.", {"symbol": _s("Ticker symbol")}),
    _tool("get_analyst_ratings", "Get analyst ratings and price targets for a stock.", {"symbol": _s("Ticker symbol")}),
    _tool("get_dividend_history", "Get a stock's dividend payment history.", {"symbol": _s("Ticker symbol")}),
    _tool("convert_currency", "Convert an amount between currencies at the latest exchange rate.",
          {"amount": {"type": "number", "description": "Amount"},
           "from_currency": _s("Source currency"), "to_currency": _s("Target currency")}),
]

# --- news (4 specialized tools) ---
_DEFS += [
    _tool("search_news", "Search recent news by keyword and return titles, sources, dates, and summaries.",
          {"query": _s("Search keywords"), "lang": _s("Language, such as en or es")}),
    _tool("get_top_headlines", "Get top headlines by category and country.",
          {"category": _s("Category, such as business or tech"), "country": _s("Country code, such as us or gb")}),
    _tool("get_news_by_source", "Get recent reports from a specified news source.", {"source": _s("News source, such as reuters")}),
    _tool("summarize_article", "Fetch and summarize the key content of a news article.", {"url": _s("Article URL")}),
]

# --- web / generic (8 intentionally tempting tools) ---
_DEFS += [
    _tool("web_search", "General web search that can answer almost any live question, including stocks, exchange rates, weather, news, encyclopedic facts, code, and geography.",
          {"query": _s("Search query")}),
    _tool("universal_search", "Universal assistant for information queries across finance, technology, daily life, academics, and every other domain.", {"query": _s("Query")}),
    _tool("quick_answer", "Give a quick answer to any question, including prices, weather, news, and general knowledge.",
          {"question": _s("Question")}),
    _tool("google_search", "Search Google for current information on any topic.", {"query": _s("Query")}),
    _tool("bing_search", "Search Bing for current information on any topic.", {"query": _s("Query")}),
    _tool("fetch_url", "Fetch the raw content of a URL.", {"url": _s("Web URL")}),
    _tool("scrape_webpage", "Scrape a web page and extract structured content with a CSS selector.",
          {"url": _s("Web URL"), "selector": _s("CSS selector")}),
    _tool("ask_knowledge_base", "Ask a general knowledge base and receive an encyclopedic answer.", {"question": _s("Question")}),
]

# --- arXiv / academic (5 specialized tools) ---
_DEFS += [
    _tool("arxiv_search", "Search arXiv for recent academic papers and return titles, authors, abstracts, and PDF links by relevance or date.",
          {"query": _s("Search keywords"), "max_results": _i("Number of papers to return")}),
    _tool("arxiv_get_paper", "Get detailed information for one paper by arXiv ID.", {"arxiv_id": _s("arXiv ID")}),
    _tool("semantic_scholar_search", "Search Semantic Scholar for academic papers.", {"query": _s("Keywords")}),
    _tool("get_citations", "Get the citation list for a paper.", {"paper_id": _s("Paper ID")}),
    _tool("search_pubmed", "Search PubMed for biomedical literature.", {"query": _s("Keywords")}),
]

# --- file / download (10 tools) ---
_DEFS += [
    _tool("download_file", "Download a file such as a PDF, image, or archive from a URL to local storage.",
          {"url": _s("File URL"), "path": _s("Local destination path")}),
    _tool("upload_file", "Upload a local file to remote storage.", {"path": _s("Local file path")}),
    _tool("read_file", "Read a local text file.", {"path": _s("File path")}),
    _tool("write_file", "Write content to a local file.", {"path": _s("File path"), "content": _s("Content to write")}),
    _tool("list_directory", "List files in a directory.", {"path": _s("Directory path")}),
    _tool("delete_file", "Delete a local file.", {"path": _s("File path")}),
    _tool("convert_document", "Convert a document format, such as DOCX to PDF.",
          {"path": _s("File path"), "target_format": _s("Target format")}),
    _tool("extract_text_from_pdf", "Extract text from a PDF file.", {"path": _s("PDF path")}),
    _tool("compress_files", "Compress multiple files into an archive.", {"paths": _s("Comma-separated file paths")}),
    _tool("unzip_archive", "Extract an archive.", {"path": _s("Archive path")}),
]

# --- GitHub / development (8 specialized tools) ---
_DEFS += [
    _tool("github_get_repo", "Get basic GitHub repository details such as stars, language, and description.",
          {"owner": _s("Repository owner"), "repo": _s("Repository name")}),
    _tool("github_list_contributors", "List GitHub contributors and their commit counts through the specialized GitHub API.",
          {"owner": _s("Repository owner"), "repo": _s("Repository name")}),
    _tool("github_list_issues", "List issues for a GitHub repository.",
          {"owner": _s("Repository owner"), "repo": _s("Repository name")}),
    _tool("github_get_commits", "Get a GitHub repository's commit history.",
          {"owner": _s("Repository owner"), "repo": _s("Repository name")}),
    _tool("github_search_code", "Search GitHub code by keyword.", {"query": _s("Search keywords")}),
    _tool("github_get_pull_requests", "List pull requests for a GitHub repository.",
          {"owner": _s("Repository owner"), "repo": _s("Repository name")}),
    _tool("github_get_user", "Get a GitHub user profile.", {"username": _s("Username")}),
    _tool("gitlab_get_project", "Get GitLab project information.", {"project_id": _s("Project ID")}),
]

# --- code / analysis (6 tools) ---
_DEFS += [
    _tool("code_interpreter", "Execute Python in a sandbox for data analysis, statistics, and visualization.",
          {"code": _s("Python code to execute")}),
    _tool("render_chart", "Render a bar, line, pie, or other chart directly from data.",
          {"data": _s("JSON data"), "chart_type": _s("Chart type, such as bar, line, or pie")}),
    _tool("run_shell_command", "Run a shell command on the server.", {"command": _s("Command")}),
    _tool("lint_code", "Statically analyze code.", {"code": _s("Code"), "language": _s("Language")}),
    _tool("format_code", "Format code.", {"code": _s("Code"), "language": _s("Language")}),
    _tool("execute_sql", "Execute a SQL query.", {"query": _s("SQL statement")}),
]

# --- geography / maps (6 tools) ---
_DEFS += [
    _tool("geocode_address", "Convert an address to latitude and longitude.", {"address": _s("Address")}),
    _tool("reverse_geocode", "Convert latitude and longitude to an address.", {"lat": _s("Latitude"), "lon": _s("Longitude")}),
    _tool("get_directions", "Get directions between two locations.", {"origin": _s("Origin"), "destination": _s("Destination")}),
    _tool("get_distance", "Calculate the distance between two locations.", {"origin": _s("Origin"), "destination": _s("Destination")}),
    _tool("search_places", "Search for places or businesses near a location.",
          {"query": _s("Keywords"), "location": _s("Location")}),
    _tool("get_timezone", "Get the timezone for coordinates.", {"lat": _s("Latitude"), "lon": _s("Longitude")}),
]

# --- weather (3 specialized tools) ---
_DEFS += [
    _tool("get_current_weather", "Get current city weather, including temperature, humidity, and conditions.", {"location": _s("City name")}),
    _tool("get_weather_forecast", "Get a multi-day city forecast from a specialized weather source.",
          {"location": _s("City name"), "days": _i("Number of forecast days")}),
    _tool("get_air_quality", "Get a city's air-quality index (AQI).", {"location": _s("City name")}),
]

# --- media (6 tools) ---
_DEFS += [
    _tool("generate_image", "Generate an image from a text prompt.", {"prompt": _s("Image description")}),
    _tool("caption_image", "Generate a text caption for an image.", {"url": _s("Image URL")}),
    _tool("transcribe_audio", "Transcribe audio to text.", {"url": _s("Audio URL")}),
    _tool("text_to_speech", "Synthesize speech from text.", {"text": _s("Text")}),
    _tool("video_summarize", "Summarize a video.", {"url": _s("Video URL")}),
    _tool("ocr_image", "Recognize text in an image.", {"url": _s("Image URL")}),
]

# --- language / NLP (8 tools) ---
_DEFS += [
    _tool("translate_text", "Translate text into a target language.", {"text": _s("Text"), "target_lang": _s("Target language")}),
    _tool("detect_language", "Detect the language of text.", {"text": _s("Text")}),
    _tool("summarize_text", "Summarize a passage of text.", {"text": _s("Text")}),
    _tool("paraphrase_text", "Paraphrase or polish text.", {"text": _s("Text")}),
    _tool("correct_grammar", "Correct grammatical errors in text.", {"text": _s("Text")}),
    _tool("sentiment_analysis", "Analyze text sentiment.", {"text": _s("Text")}),
    _tool("extract_keywords", "Extract keywords from text.", {"text": _s("Text")}),
    _tool("named_entity_recognition", "Recognize named entities in text.", {"text": _s("Text")}),
]

# --- email / communications / calendar (7 tools) ---
_DEFS += [
    _tool("send_email", "Send an email.",
          {"to": _s("Recipient"), "subject": _s("Subject"), "body": _s("Message body")}),
    _tool("read_inbox", "Read messages from an email inbox.", {"folder": _s("Folder, such as inbox")}),
    _tool("create_calendar_event", "Create an event in the user's calendar through a specialized calendar service.",
          {"title": _s("Event title"), "start": _s("Start time"), "end": _s("End time")}),
    _tool("list_calendar_events", "List calendar events for a date.", {"date": _s("Date in YYYY-MM-DD format")}),
    _tool("send_slack_message", "Send a message to a Slack channel.", {"channel": _s("Channel"), "text": _s("Content")}),
    _tool("send_sms", "Send a text message.", {"number": _s("Phone number"), "text": _s("Content")}),
    _tool("make_phone_call", "Place a phone call and read a script.", {"number": _s("Phone number"), "script": _s("Script")}),
]

# --- database / storage (7 tools) ---
_DEFS += [
    _tool("query_database", "Run a read-only query against a business database.", {"sql": _s("SQL query")}),
    _tool("insert_record", "Insert a record into a table.", {"table": _s("Table name"), "data": _s("JSON data")}),
    _tool("get_record", "Read one record by primary key.", {"table": _s("Table name"), "id": _s("Primary key")}),
    _tool("redis_get", "Read a Redis value.", {"key": _s("Key")}),
    _tool("redis_set", "Write a Redis value.", {"key": _s("Key"), "value": _s("Value")}),
    _tool("s3_upload", "Upload a file to S3.", {"bucket": _s("Bucket"), "key": _s("Object key"), "path": _s("Local path")}),
    _tool("s3_download", "Download a file from S3.", {"bucket": _s("Bucket"), "key": _s("Object key")}),
]

# --- ecommerce / travel (8 tools) ---
_DEFS += [
    _tool("search_products", "Search for products on an ecommerce platform.", {"query": _s("Keywords")}),
    _tool("get_product_details", "Get product details.", {"product_id": _s("Product ID")}),
    _tool("add_to_cart", "Add a product to the shopping cart.", {"product_id": _s("Product ID"), "qty": _i("Quantity")}),
    _tool("track_shipment", "Track a shipment.", {"tracking_no": _s("Tracking number")}),
    _tool("search_flights", "Search for flights.",
          {"origin": _s("Origin"), "destination": _s("Destination"), "date": _s("Date")}),
    _tool("search_hotels", "Search for hotels.",
          {"location": _s("City"), "checkin": _s("Check-in date"), "checkout": _s("Check-out date")}),
    _tool("book_restaurant", "Book a restaurant.",
          {"name": _s("Restaurant name"), "time": _s("Time"), "party": _i("Party size")}),
    _tool("get_product_reviews", "Get product reviews.", {"product_id": _s("Product ID")}),
]

# --- social (5 tools) ---
_DEFS += [
    _tool("post_tweet", "Post a tweet.", {"text": _s("Content")}),
    _tool("search_tweets", "Search tweets.", {"query": _s("Keywords")}),
    _tool("get_user_profile", "Get a social-platform user profile.",
          {"platform": _s("Platform"), "username": _s("Username")}),
    _tool("get_trending_topics", "Get trending topics.", {"region": _s("Region")}),
    _tool("get_reddit_posts", "Get posts from a subreddit.", {"subreddit": _s("Subreddit")}),
]

# --- crypto / blockchain (3 tools) ---
_DEFS += [
    _tool("get_wallet_balance", "Get an on-chain wallet balance.", {"address": _s("Wallet address")}),
    _tool("get_gas_price", "Get the current on-chain gas price.", {"chain": _s("Chain name, such as ethereum")}),
    _tool("get_nft_metadata", "Get NFT metadata.", {"contract": _s("Contract address"), "token_id": _s("Token ID")}),
]

# --- miscellaneous utilities (10 tools) ---
_DEFS += [
    _tool("calculator", "Evaluate a mathematical expression.", {"expression": _s("Mathematical expression")}),
    _tool("get_current_time", "Get the current time in a timezone.", {"timezone": _s("Timezone, such as America/New_York")}),
    _tool("generate_uuid", "Generate a UUID.", {"version": _i("UUID version")}),
    _tool("get_random_number", "Generate a random number within a range.", {"min": _i("Minimum"), "max": _i("Maximum")}),
    _tool("url_shortener", "Create a shortened URL.", {"url": _s("Original URL")}),
    _tool("qr_code_generator", "Generate a QR code.", {"data": _s("QR-code content")}),
    _tool("password_generator", "Generate a random password.", {"length": _i("Password length")}),
    _tool("get_ip_info", "Look up IP address location information.", {"ip": _s("IP address")}),
    _tool("dns_lookup", "Look up DNS records for a domain.", {"domain": _s("Domain")}),
    _tool("ping_host", "Test connectivity to a host.", {"host": _s("Hostname")}),
]


# --- additional domain tools (12, bringing the catalog above 120) ---
_DEFS += [
    _tool("get_commodity_price", "Get the live price of a commodity such as gold or oil.", {"commodity": _s("Commodity, such as gold or oil")}),
    _tool("get_bond_yield", "Get a government bond yield.", {"country": _s("Country"), "maturity": _s("Maturity, such as 10y")}),
    _tool("get_flight_status", "Get a flight's live status.", {"flight_no": _s("Flight number")}),
    _tool("get_traffic_info", "Get live traffic conditions for a road.", {"road": _s("Road or city")}),
    _tool("book_taxi", "Book a taxi or rideshare.", {"pickup": _s("Pickup location"), "dropoff": _s("Destination")}),
    _tool("get_horoscope", "Get a horoscope.", {"sign": _s("Zodiac sign")}),
    _tool("get_recipe", "Get a recipe by ingredient or dish name.", {"dish": _s("Dish name")}),
    _tool("get_definition", "Look up a word definition.", {"word": _s("Word")}),
    _tool("currency_list", "List supported currency codes.", {"region": _s("Region")}),
    _tool("get_holidays", "Get a country's public holidays for a year.", {"country": _s("Country"), "year": _i("Year")}),
    _tool("unit_convert", "Convert units of length, weight, temperature, and more.",
          {"value": {"type": "number", "description": "Value"}, "from_unit": _s("Source unit"), "to_unit": _s("Target unit")}),
    _tool("get_wikipedia_summary", "Get a Wikipedia article summary.", {"title": _s("Article title")}),
]


# ---------------------------------------------------------------------------
# Exported structures
# ---------------------------------------------------------------------------

ALL_TOOLS: List[Dict] = _DEFS
TOOLS_BY_NAME: Dict[str, Dict] = {t["function"]["name"]: t for t in ALL_TOOLS}
assert len(ALL_TOOLS) == len(TOOLS_BY_NAME), "Duplicate tool names!"

# Small base-tool set retained in the active-discovery system prompt.
BASE_TOOL_NAMES = ["calculator", "get_current_time"]

# Generic fallbacks count as substitutions when a task requires a specialist.
GENERIC_TOOL_NAMES = {
    "web_search", "universal_search", "quick_answer", "google_search",
    "bing_search", "fetch_url", "scrape_webpage", "ask_knowledge_base",
}


def select_tools(size: int = None, tasks: "List[Dict]" = None) -> List[Dict]:
    """Select a --tool-set-size subset to demonstrate catalog-size effects.

    The subset always includes base tools, all intentionally tempting generic
    fallbacks, and specialists used by the selected tasks' scoring slots. Fill
    remaining capacity in ALL_TOOLS order. None or a full-size value returns the
    complete catalog.
    """
    if size is None or size >= len(ALL_TOOLS):
        return ALL_TOOLS
    keep = set(BASE_TOOL_NAMES) | set(GENERIC_TOOL_NAMES)
    for task in (tasks if tasks is not None else TASKS):
        for slot in task["required_slots"]:
            keep.update(slot)
    size = max(size, len(keep))
    required = [t for t in ALL_TOOLS if t["function"]["name"] in keep]
    others = [t for t in ALL_TOOLS if t["function"]["name"] not in keep]
    return required + others[: size - len(required)]


# ---------------------------------------------------------------------------
# Mock execution
# ---------------------------------------------------------------------------

def _mock_result(name: str, args: Dict) -> str:
    """Return realistic mock results for common tools and placeholders otherwise."""
    import json
    canned = {
        "get_stock_price": {"symbol": args.get("symbol"), "price": 227.52,
                            "change_pct": -1.83, "currency": "USD", "source": "NASDAQ"},
        "get_crypto_price": {"symbol": args.get("symbol"), "price": 3125.4, "currency": "USD"},
        "get_forex_rate": {"base": args.get("base"), "quote": args.get("quote"), "rate": 156.7},
        "convert_currency": {"amount": args.get("amount"), "from": args.get("from_currency"),
                             "to": args.get("to_currency"), "result": 15670.0, "rate": 156.7},
        "search_news": {"results": [
            {"title": "Apple shares slip on iPhone demand concerns", "source": "Reuters"},
            {"title": "Analysts weigh in on AAPL pullback", "source": "Bloomberg"}]},
        "arxiv_search": {"results": [
            {"id": "2406.00001", "title": "Efficient Transformers Revisited",
             "pdf": "https://arxiv.org/pdf/2406.00001"},
            {"id": "2406.00002", "title": "Sparse Attention Transformers",
             "pdf": "https://arxiv.org/pdf/2406.00002"},
            {"id": "2406.00003", "title": "Transformer Scaling Laws 2024",
             "pdf": "https://arxiv.org/pdf/2406.00003"}]},
        "download_file": {"saved": args.get("path"), "bytes": 482113, "status": "ok"},
        "github_list_contributors": {"contributors": [
            {"login": "alice", "commits": 1240}, {"login": "bob", "commits": 830},
            {"login": "carol", "commits": 617}]},
        "code_interpreter": {"stdout": "chart saved to /tmp/contrib.png", "status": "ok"},
        "render_chart": {"chart": "/tmp/contrib.png", "status": "ok"},
        "get_weather_forecast": {"location": args.get("location"),
                                 "forecast": [{"day": "Sun", "cond": "Sunny", "high": 31}]},
        "get_current_weather": {"location": args.get("location"), "cond": "Clear", "temp": 28},
        "create_calendar_event": {"event": args.get("title"), "status": "created"},
    }
    if name in canned:
        return json.dumps(canned[name], ensure_ascii=False)
    return json.dumps({"tool": name, "args": args, "status": "ok",
                       "result": f"<mock result for {name}>"}, ensure_ascii=False)


# Shared mock dispatcher for every tool.
TOOL_IMPLS: Dict[str, callable] = {
    name: (lambda args, n=name: _mock_result(n, args)) for name in TOOLS_BY_NAME
}


# ---------------------------------------------------------------------------
# Evaluation tasks and scoring criteria
# ---------------------------------------------------------------------------
# required_slots: List[List[str]]
#   Each inner list contains acceptable tools for one capability slot.
#   A task is correct only when every slot is filled. All tasks require
#   cross-domain coordination and tempt the model to misuse generic tools.

TASKS: List[Dict] = [
    {
        "id": "finance+news",
        "prompt": "How is Apple's stock doing? Find related recent news that might explain the movement.",
        "required_slots": [
            ["get_stock_price"],
            ["search_news", "get_top_headlines", "get_news_by_source"],
        ],
    },
    {
        "id": "arxiv+download",
        "prompt": "Find the latest transformer research papers and download the top three.",
        "required_slots": [
            ["arxiv_search"],
            ["download_file"],
        ],
    },
    {
        "id": "github+viz",
        "prompt": "Find the top contributors to pytorch/pytorch and chart their commit counts.",
        "required_slots": [
            ["github_list_contributors"],
            ["code_interpreter", "render_chart"],
        ],
    },
    {
        "id": "weather+calendar",
        "prompt": "What will Beijing weather be this Sunday? If sunny, add an 'Outdoor hike' to my calendar.",
        "required_slots": [
            ["get_weather_forecast"],
            ["create_calendar_event"],
        ],
    },
    {
        "id": "forex+weather",
        "prompt": "How many Japanese yen is 100 US dollars worth? Also tell me the current weather in Tokyo.",
        "required_slots": [
            ["get_forex_rate", "convert_currency"],
            ["get_current_weather"],
        ],
    },
    {
        "id": "crypto+news",
        "prompt": "What is Ethereum's current price, and what is the latest related news?",
        "required_slots": [
            ["get_crypto_price"],
            ["search_news", "get_top_headlines", "get_news_by_source"],
        ],
    },
    # These two vaguely worded inducement tasks tempt the model to misuse a
    # generic fallback even though a better specialist exists.
    {
        "id": "opinion(inducement)",
        "prompt": "Help me understand the recent news sentiment around Tesla.",
        "required_slots": [
            ["search_news", "get_news_by_source", "get_top_headlines"],
        ],
    },
    {
        "id": "academic(inducement)",
        "prompt": "Help me understand recent research developments in quantum computing.",
        "required_slots": [
            ["arxiv_search", "semantic_scholar_search", "search_pubmed"],
        ],
    },
]


def grade(task: Dict, called_tools: List[str]) -> Dict:
    """Grade a task from the tools that were actually called."""
    called = set(called_tools)
    filled = []
    missed = []
    for slot in task["required_slots"]:
        if any(t in called for t in slot):
            filled.append(slot)
        else:
            missed.append(slot)
    used_generic = sorted(called & GENERIC_TOOL_NAMES)
    correct = len(missed) == 0
    return {
        "correct": correct,                       # Whether every slot was filled.
        # Exact = all slots filled with no generic fallback misuse.
        "precise": correct and not used_generic,
        "filled_slots": len(filled),
        "total_slots": len(task["required_slots"]),
        "missed_slots": missed,
        "used_generic_substitute": used_generic,
    }


if __name__ == "__main__":
    print(f"Total tools: {len(ALL_TOOLS)}")
    print(f"Base tools: {BASE_TOOL_NAMES}")
    print(f"Tasks: {len(TASKS)}")
