"""The catalog of MCP servers and A2A agents the test worker
(`app/testing/worker.py`, run with `scripts/run_worker.py`) exposes.

Design rules every entry follows, because the point is regression testing
compiled multi-tool agents, not demoing:

  - Deterministic. No clocks, randomness, or network: the same input always
    gives the same output, so a test can assert exact values.
  - Self-describing. Every MCP tool and every A2A skill carries one worked
    example (input + exact expected output). `tests/test_worker_regression.py`
    runs every one of them over the real protocol, so the catalog *is* the
    regression suite -- adding an entry adds a test.
  - Compatible with how compiled spec steps call tools
    (`app/factory/spec_agents._step_tool_function`): MCP tools take typed
    JSON arguments (a plan step passes them via `PlanStep.arguments`, and
    names the tool via `PlanStep.tool_name`); every MCP tool returns a JSON
    object, so its MCP `structuredContent` is exactly that object. A2A skills
    are text in, text out -- the A2A message model -- so any step's
    instruction text can be sent to any of them, and their outputs can be
    fed to each other.

Diversity is deliberate: text, math, statistics, JSON, dates, units,
hashing, encoding, regex extraction, CSV, URLs, geo, finance, lists, color,
validation, a stateful-looking (fixed) inventory, lookup tables, sentiment,
translation, summarization -- typed numeric args, list args, dict args,
optional args, and error-shaped results all appear somewhere.
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
import re
import statistics
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any, Callable
from urllib.parse import parse_qsl, urlencode, urlparse


@dataclass(frozen=True)
class McpToolDef:
    fn: Callable[..., dict]
    example_args: dict[str, Any]
    expected: dict[str, Any]

    @property
    def name(self) -> str:
        return self.fn.__name__


@dataclass(frozen=True)
class McpServerDef:
    name: str
    description: str
    tools: list[McpToolDef] = field(default_factory=list)


@dataclass(frozen=True)
class A2ASkillDef:
    id: str
    name: str
    description: str
    handler: Callable[[str], str]
    example_input: str
    expected: str


@dataclass(frozen=True)
class A2AAgentDef:
    name: str
    description: str
    skills: list[A2ASkillDef] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Shared helpers / fixed data
# ---------------------------------------------------------------------------

_POSITIVE = {"good", "great", "love", "excellent", "happy", "awesome"}
_NEGATIVE = {"bad", "terrible", "hate", "awful", "sad", "broken"}
_EN_ES = {
    "hello": "hola",
    "world": "mundo",
    "good": "bueno",
    "morning": "mañana",
    "friend": "amigo",
    "the": "el",
    "cat": "gato",
    "dog": "perro",
    "thanks": "gracias",
}
_ES_MARKERS = {"hola", "gracias", "mundo", "amigo", "el", "la", "perro", "gato"}
_STOPWORDS = {"the", "a", "an", "and", "or", "with", "to", "of", "in", "on", "is", "it", "for"}
_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_PHONE_RE = re.compile(r"\+?\d[\d\s().-]{8,}\d")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_INVENTORY = {
    "SKU-1": {"name": "Widget", "price": 2.5, "stock": 40},
    "SKU-2": {"name": "Gadget", "price": 10.0, "stock": 0},
    "SKU-3": {"name": "Doohickey", "price": 7.25, "stock": 12},
}
_WEATHER = {
    "london": {"condition": "rain", "high_c": 14, "low_c": 9, "uv": 2},
    "paris": {"condition": "sunny", "high_c": 19, "low_c": 11, "uv": 5},
    "tokyo": {"condition": "cloudy", "high_c": 21, "low_c": 16, "uv": 4},
    "mumbai": {"condition": "humid", "high_c": 31, "low_c": 26, "uv": 9},
}


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_SPLIT.split(text.strip()) if s.strip()]


def _words(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z']+", text.lower())


def _sentiment_score(text: str) -> int:
    words = _words(text)
    return sum(w in _POSITIVE for w in words) - sum(w in _NEGATIVE for w in words)


def _translate(text: str) -> str:
    return " ".join(_EN_ES.get(w, w) for w in text.lower().split())


# ---------------------------------------------------------------------------
# MCP tools -- every one returns a dict (its MCP structuredContent)
# ---------------------------------------------------------------------------


def word_count(text: str) -> dict:
    """Count the words and characters in a piece of text."""
    return {"words": len(text.split()), "characters": len(text)}


def slugify(text: str) -> dict:
    """Turn text into a URL-safe lowercase slug."""
    return {"slug": re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")}


def to_upper(text: str) -> dict:
    """Uppercase text."""
    return {"text": text.upper()}


def to_title(text: str) -> dict:
    """Title-case text."""
    return {"text": text.title()}


def add(a: float, b: float) -> dict:
    """Add two numbers."""
    return {"result": round(a + b, 6)}


def multiply(a: float, b: float) -> dict:
    """Multiply two numbers."""
    return {"result": round(a * b, 6)}


def mean(values: list[float]) -> dict:
    """Arithmetic mean of a list of numbers."""
    return {"mean": round(statistics.fmean(values), 4)}


def median(values: list[float]) -> dict:
    """Median of a list of numbers."""
    return {"median": statistics.median(values)}


def extract_field(json_text: str, field: str) -> dict:
    """Extract a (dot.separated) field from a JSON document."""
    try:
        node: Any = json.loads(json_text)
    except json.JSONDecodeError as exc:
        return {"found": False, "error": f"invalid JSON: {exc.msg}"}
    for key in field.split("."):
        if isinstance(node, dict) and key in node:
            node = node[key]
        else:
            return {"found": False, "value": None}
    return {"found": True, "value": node}


def is_valid_json(text: str) -> dict:
    """Check whether text parses as JSON."""
    try:
        json.loads(text)
        return {"valid": True}
    except json.JSONDecodeError:
        return {"valid": False}


def add_days(date_iso: str, days: int) -> dict:
    """Add a number of days to an ISO date (YYYY-MM-DD)."""
    return {"date": (date.fromisoformat(date_iso) + timedelta(days=days)).isoformat()}


def weekday(date_iso: str) -> dict:
    """Day of the week for an ISO date (YYYY-MM-DD)."""
    return {"weekday": date.fromisoformat(date_iso).strftime("%A")}


def celsius_to_fahrenheit(celsius: float) -> dict:
    """Convert Celsius to Fahrenheit."""
    return {"fahrenheit": round(celsius * 9 / 5 + 32, 2)}


def km_to_miles(km: float) -> dict:
    """Convert kilometers to miles."""
    return {"miles": round(km * 0.621371, 3)}


def sha256_hex(text: str) -> dict:
    """SHA-256 hex digest of text (UTF-8)."""
    return {"sha256": hashlib.sha256(text.encode("utf-8")).hexdigest()}


def md5_hex(text: str) -> dict:
    """MD5 hex digest of text (UTF-8) -- for checksums, not security."""
    return {"md5": hashlib.md5(text.encode("utf-8")).hexdigest()}  # noqa: S324 - checksum, not security


def base64_encode(text: str) -> dict:
    """Base64-encode text (UTF-8)."""
    return {"data": base64.b64encode(text.encode("utf-8")).decode("ascii")}


def base64_decode(data: str) -> dict:
    """Decode base64 back to UTF-8 text."""
    try:
        return {"text": base64.b64decode(data, validate=True).decode("utf-8")}
    except (ValueError, UnicodeDecodeError) as exc:
        return {"error": f"not valid base64 text: {exc}"}


def extract_emails(text: str) -> dict:
    """Find every email address in text."""
    return {"emails": _EMAIL_RE.findall(text)}


def extract_numbers(text: str) -> dict:
    """Find every number in text."""
    return {"numbers": [float(n) for n in re.findall(r"-?\d+(?:\.\d+)?", text)]}


def csv_row_count(csv_text: str) -> dict:
    """Number of data rows in CSV text (header excluded)."""
    lines = [line for line in csv_text.strip().splitlines() if line.strip()]
    return {"rows": max(len(lines) - 1, 0)}


def csv_column_sum(csv_text: str, column: str) -> dict:
    """Sum a numeric column of CSV text by header name."""
    lines = [line for line in csv_text.strip().splitlines() if line.strip()]
    header = [h.strip() for h in lines[0].split(",")]
    if column not in header:
        return {"column": column, "error": f"no such column; available: {header}"}
    index = header.index(column)
    return {"column": column, "sum": round(sum(float(line.split(",")[index]) for line in lines[1:]), 6)}


def parse_url(url: str) -> dict:
    """Split a URL into scheme, host, path and query parameters."""
    parsed = urlparse(url)
    return {"scheme": parsed.scheme, "host": parsed.netloc, "path": parsed.path, "query": dict(parse_qsl(parsed.query))}


def build_url(base: str, params: dict[str, str]) -> dict:
    """Append query parameters (sorted, URL-encoded) to a base URL."""
    return {"url": f"{base}?{urlencode(sorted(params.items()))}" if params else base}


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> dict:
    """Great-circle distance between two lat/lon points, in km."""
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return {"km": round(2 * r * math.asin(math.sqrt(a)), 1)}


def midpoint(lat1: float, lon1: float, lat2: float, lon2: float) -> dict:
    """Simple (planar) midpoint between two lat/lon points."""
    return {"lat": round((lat1 + lat2) / 2, 4), "lon": round((lon1 + lon2) / 2, 4)}


def compound_interest(principal: float, rate_pct: float, years: int) -> dict:
    """Amount after annual compounding."""
    return {"amount": round(principal * (1 + rate_pct / 100) ** years, 2)}


def loan_payment(principal: float, annual_rate_pct: float, months: int) -> dict:
    """Fixed monthly payment for an amortized loan."""
    monthly = annual_rate_pct / 100 / 12
    payment = principal / months if monthly == 0 else principal * monthly / (1 - (1 + monthly) ** -months)
    return {"monthly_payment": round(payment, 2)}


def sort_items(items: list[str]) -> dict:
    """Sort a list of strings."""
    return {"items": sorted(items)}


def dedupe_items(items: list[str]) -> dict:
    """Remove duplicates from a list, keeping first occurrences in order."""
    return {"items": list(dict.fromkeys(items))}


def hex_to_rgb(hex_color: str) -> dict:
    """Convert a #rrggbb color to RGB components."""
    value = hex_color.lstrip("#")
    if not re.fullmatch(r"[0-9a-fA-F]{6}", value):
        return {"error": f"not a #rrggbb color: {hex_color!r}"}
    return {"r": int(value[0:2], 16), "g": int(value[2:4], 16), "b": int(value[4:6], 16)}


def rgb_to_hex(r: int, g: int, b: int) -> dict:
    """Convert RGB components (0-255) to a #rrggbb color."""
    return {"hex": "#{:02x}{:02x}{:02x}".format(*(max(0, min(255, c)) for c in (r, g, b)))}


def validate_email(email: str) -> dict:
    """Check whether a string is a plausibly valid email address."""
    return {"email": email, "valid": bool(re.fullmatch(r"[\w.+-]+@[\w-]+(\.[\w-]+)+", email))}


def validate_phone(phone: str) -> dict:
    """Check whether a string is a plausible phone number (10-15 digits)."""
    digits = re.sub(r"\D", "", phone)
    return {"digits": digits, "valid": 10 <= len(digits) <= 15}


def stock_level(sku: str) -> dict:
    """Current stock level for a SKU."""
    item = _INVENTORY.get(sku)
    if item is None:
        return {"sku": sku, "error": "unknown sku"}
    return {"sku": sku, "name": item["name"], "stock": item["stock"], "in_stock": item["stock"] > 0}


def price_quote(sku: str, quantity: int) -> dict:
    """Price quote for a quantity of a SKU."""
    item = _INVENTORY.get(sku)
    if item is None:
        return {"sku": sku, "error": "unknown sku"}
    return {"sku": sku, "quantity": quantity, "unit_price": item["price"], "total": round(item["price"] * quantity, 2)}


def forecast(city: str) -> dict:
    """Today's forecast for a city (fixed demo data)."""
    item = _WEATHER.get(city.lower())
    if item is None:
        return {"city": city, "error": f"no forecast for {city}; known: {sorted(_WEATHER)}"}
    return {"city": city.title(), "condition": item["condition"], "high_c": item["high_c"], "low_c": item["low_c"]}


def uv_index(city: str) -> dict:
    """UV index and risk band for a city (fixed demo data)."""
    item = _WEATHER.get(city.lower())
    if item is None:
        return {"city": city, "error": f"no UV data for {city}"}
    uv = item["uv"]
    risk = "low" if uv <= 2 else "moderate" if uv <= 5 else "high" if uv <= 7 else "very high"
    return {"city": city.title(), "uv_index": uv, "risk": risk}


def sentiment(text: str) -> dict:
    """Lexicon-based sentiment of text."""
    score = _sentiment_score(text)
    return {"label": "positive" if score > 0 else "negative" if score < 0 else "neutral", "score": score}


def keyword_density(text: str, keyword: str) -> dict:
    """How often a keyword appears, relative to word count."""
    words = _words(text)
    count = words.count(keyword.lower())
    return {"keyword": keyword, "count": count, "density": round(count / len(words), 4) if words else 0.0}


def translate_en_es(text: str) -> dict:
    """Word-by-word English -> Spanish (small fixed dictionary)."""
    return {"translation": _translate(text)}


def detect_language(text: str) -> dict:
    """Guess whether text is English or Spanish."""
    return {"language": "es" if set(_words(text)) & _ES_MARKERS else "en"}


def first_sentences(text: str, n: int = 1) -> dict:
    """The first n sentences of text."""
    return {"summary": " ".join(_sentences(text)[: max(n, 0)])}


def headline(text: str) -> dict:
    """A title-cased headline from the first eight words of text."""
    words = re.sub(r"[^\w\s]", "", text).split()[:8]
    return {"headline": " ".join(w.capitalize() for w in words)}


MCP_SERVERS: list[McpServerDef] = [
    McpServerDef("text-tools", "Word counts and slugs for text.", [
        McpToolDef(word_count, {"text": "the quick brown fox"}, {"words": 4, "characters": 19}),
        McpToolDef(slugify, {"text": "Hello, World! 2026"}, {"slug": "hello-world-2026"}),
    ]),
    McpServerDef("case-tools", "Change the case of text.", [
        McpToolDef(to_upper, {"text": "abc Def"}, {"text": "ABC DEF"}),
        McpToolDef(to_title, {"text": "hello big world"}, {"text": "Hello Big World"}),
    ]),
    McpServerDef("math-tools", "Basic arithmetic.", [
        McpToolDef(add, {"a": 2, "b": 3}, {"result": 5}),
        McpToolDef(multiply, {"a": 4, "b": 2.5}, {"result": 10}),
    ]),
    McpServerDef("stats-tools", "Descriptive statistics over lists of numbers.", [
        McpToolDef(mean, {"values": [1, 2, 3, 4]}, {"mean": 2.5}),
        McpToolDef(median, {"values": [3, 1, 2]}, {"median": 2}),
    ]),
    McpServerDef("json-tools", "Parse and query JSON documents.", [
        McpToolDef(extract_field, {"json_text": '{"user": {"name": "Ada"}}', "field": "user.name"}, {"found": True, "value": "Ada"}),
        McpToolDef(is_valid_json, {"text": '{"a": 1}'}, {"valid": True}),
    ]),
    McpServerDef("date-tools", "Calendar arithmetic on ISO dates.", [
        McpToolDef(add_days, {"date_iso": "2026-01-30", "days": 3}, {"date": "2026-02-02"}),
        McpToolDef(weekday, {"date_iso": "2026-10-16"}, {"weekday": "Friday"}),
    ]),
    McpServerDef("unit-tools", "Unit conversions.", [
        McpToolDef(celsius_to_fahrenheit, {"celsius": 100}, {"fahrenheit": 212}),
        McpToolDef(km_to_miles, {"km": 10}, {"miles": 6.214}),
    ]),
    McpServerDef("hash-tools", "Content digests.", [
        McpToolDef(sha256_hex, {"text": "abc"}, {"sha256": "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"}),
        McpToolDef(md5_hex, {"text": "abc"}, {"md5": "900150983cd24fb0d6963f7d28e17f72"}),
    ]),
    McpServerDef("encoding-tools", "Base64 encoding and decoding.", [
        McpToolDef(base64_encode, {"text": "hello"}, {"data": "aGVsbG8="}),
        McpToolDef(base64_decode, {"data": "aGVsbG8="}, {"text": "hello"}),
    ]),
    McpServerDef("regex-tools", "Pull structured values out of free text.", [
        McpToolDef(extract_emails, {"text": "contact a@b.com or c.d@e.org"}, {"emails": ["a@b.com", "c.d@e.org"]}),
        McpToolDef(extract_numbers, {"text": "3 apples and 4.5 pears"}, {"numbers": [3.0, 4.5]}),
    ]),
    McpServerDef("csv-tools", "Small CSV computations.", [
        McpToolDef(csv_row_count, {"csv_text": "a,b\n1,2\n3,4"}, {"rows": 2}),
        McpToolDef(csv_column_sum, {"csv_text": "a,b\n1,2\n3,4", "column": "b"}, {"column": "b", "sum": 6}),
    ]),
    McpServerDef("url-tools", "Parse and build URLs.", [
        McpToolDef(parse_url, {"url": "https://ex.com/p/q?x=1&y=2"}, {"scheme": "https", "host": "ex.com", "path": "/p/q", "query": {"x": "1", "y": "2"}}),
        McpToolDef(build_url, {"base": "https://ex.com/s", "params": {"q": "adk", "page": "2"}}, {"url": "https://ex.com/s?page=2&q=adk"}),
    ]),
    McpServerDef("geo-tools", "Distances and midpoints between coordinates.", [
        McpToolDef(haversine_km, {"lat1": 48.8566, "lon1": 2.3522, "lat2": 51.5074, "lon2": -0.1278}, {"km": 343.6}),
        McpToolDef(midpoint, {"lat1": 0, "lon1": 0, "lat2": 10, "lon2": 20}, {"lat": 5, "lon": 10}),
    ]),
    McpServerDef("finance-tools", "Interest and loan calculations.", [
        McpToolDef(compound_interest, {"principal": 1000, "rate_pct": 5, "years": 2}, {"amount": 1102.5}),
        McpToolDef(loan_payment, {"principal": 10000, "annual_rate_pct": 12, "months": 12}, {"monthly_payment": 888.49}),
    ]),
    McpServerDef("list-tools", "Sort and de-duplicate lists.", [
        McpToolDef(sort_items, {"items": ["b", "a", "c"]}, {"items": ["a", "b", "c"]}),
        McpToolDef(dedupe_items, {"items": ["a", "b", "a"]}, {"items": ["a", "b"]}),
    ]),
    McpServerDef("color-tools", "Convert between hex and RGB colors.", [
        McpToolDef(hex_to_rgb, {"hex_color": "#ff8000"}, {"r": 255, "g": 128, "b": 0}),
        McpToolDef(rgb_to_hex, {"r": 255, "g": 128, "b": 0}, {"hex": "#ff8000"}),
    ]),
    McpServerDef("validation-tools", "Validate emails and phone numbers.", [
        McpToolDef(validate_email, {"email": "ada@example.com"}, {"email": "ada@example.com", "valid": True}),
        McpToolDef(validate_phone, {"phone": "+1 (555) 123-4567"}, {"digits": "15551234567", "valid": True}),
    ]),
    McpServerDef("inventory-tools", "Stock levels and price quotes (fixed demo catalog).", [
        McpToolDef(stock_level, {"sku": "SKU-1"}, {"sku": "SKU-1", "name": "Widget", "stock": 40, "in_stock": True}),
        McpToolDef(price_quote, {"sku": "SKU-3", "quantity": 4}, {"sku": "SKU-3", "quantity": 4, "unit_price": 7.25, "total": 29.0}),
    ]),
    McpServerDef("weather-tools", "Forecasts and UV index (fixed demo data).", [
        McpToolDef(forecast, {"city": "London"}, {"city": "London", "condition": "rain", "high_c": 14, "low_c": 9}),
        McpToolDef(uv_index, {"city": "paris"}, {"city": "Paris", "uv_index": 5, "risk": "moderate"}),
    ]),
    McpServerDef("sentiment-tools", "Lexicon-based sentiment and keyword density.", [
        McpToolDef(sentiment, {"text": "I love this great product"}, {"label": "positive", "score": 2}),
        McpToolDef(keyword_density, {"text": "the cat and the hat", "keyword": "the"}, {"keyword": "the", "count": 2, "density": 0.4}),
    ]),
    McpServerDef("translate-tools", "Tiny English/Spanish translation and language detection.", [
        McpToolDef(translate_en_es, {"text": "hello world friend"}, {"translation": "hola mundo amigo"}),
        McpToolDef(detect_language, {"text": "hola amigo"}, {"language": "es"}),
    ]),
    McpServerDef("summarize-tools", "Extractive summaries and headlines.", [
        McpToolDef(first_sentences, {"text": "One. Two. Three.", "n": 2}, {"summary": "One. Two."}),
        McpToolDef(headline, {"text": "the quick brown fox jumps over the lazy dog today"}, {"headline": "The Quick Brown Fox Jumps Over The Lazy"}),
    ]),
]


# ---------------------------------------------------------------------------
# A2A skills -- text in, text out
# ---------------------------------------------------------------------------


def _triage(text: str) -> str:
    words = set(_words(text))
    if words & {"outage", "down", "urgent", "security", "breach"}:
        return "priority=high"
    if words & {"error", "bug", "slow", "fail", "failing"}:
        return "priority=medium"
    return "priority=low"


def _route(text: str) -> str:
    words = set(_words(text))
    if words & {"billing", "payment", "invoice", "refund"}:
        return "team=billing"
    if words & {"login", "password", "sso"}:
        return "team=identity"
    return "team=general"


def _classify_sentiment(text: str) -> str:
    score = _sentiment_score(text)
    return "positive" if score > 0 else "negative" if score < 0 else "neutral"


def _summarize(text: str) -> str:
    sentences = _sentences(text)
    return sentences[0] if sentences else ""


def _summarize_bullets(text: str) -> str:
    return "\n".join(f"- {s}" for s in _sentences(text))


def _review_code(text: str) -> str:
    return f"todo={text.count('TODO')} fixme={text.count('FIXME')} print_calls={text.count('print(')}"


_FAQ = {
    "refund": "Refunds are processed within 5 business days.",
    "shipping": "Standard shipping takes 3-7 days.",
    "password": "Reset your password from Settings > Security.",
}


def _answer_faq(text: str) -> str:
    words = set(_words(text))
    return next((answer for key, answer in _FAQ.items() if key in words), "I don't have an answer for that yet.")


_WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]


def _propose_slot(text: str) -> str:
    words = set(_words(text))
    day = next((d for d in _WEEKDAYS if d in words), "monday")
    return f"Proposed slot: {day.capitalize()} 10:00-10:30"


_DOLLAR_AMOUNT_RE = re.compile(r"\$(\d+(?:\.\d+)?)")


def _total_invoice(text: str) -> str:
    total = sum(float(x) for x in _DOLLAR_AMOUNT_RE.findall(text))
    return f"total={total:.2f}"


_CLAUSES = ["confidentiality", "indemnification", "termination", "governing law", "liability"]


def _detect_clauses(text: str) -> str:
    lowered = text.lower()
    found = [c for c in _CLAUSES if c in lowered or (c == "indemnification" and "indemnif" in lowered)]
    return f"clauses={','.join(found) if found else 'none'}"


def _meta_description(text: str) -> str:
    collapsed = " ".join(text.split())
    return collapsed if len(collapsed) <= 155 else collapsed[:152].rstrip() + "..."


def _draft_email(text: str) -> str:
    return f"Hi there,\n\n{text.strip()}\n\nBest regards,\nThe Team"


def _check_text(text: str) -> str:
    double_spaces = text.count("  ")
    lowercase_starts = sum(1 for s in _sentences(text) if s[:1].islower())
    return f"double_spaces={double_spaces} lowercase_sentence_starts={lowercase_starts}"


_RISK_WEIGHTS = {"fraud": 5, "chargeback": 4, "overdue": 3, "dispute": 2, "late": 1}


def _score_risk(text: str) -> str:
    words = _words(text)
    score = sum(_RISK_WEIGHTS.get(w, 0) for w in words)
    level = "high" if score >= 7 else "medium" if score >= 3 else "low"
    return f"risk_score={score} level={level}"


_RECOMMENDATIONS = {"camera": ["tripod", "sd card"], "laptop": ["mouse", "laptop sleeve"], "running": ["water bottle", "running socks"]}


def _recommend(text: str) -> str:
    words = set(_words(text))
    picks = next((items for key, items in _RECOMMENDATIONS.items() if key in words), ["gift card"])
    return f"recommendations={', '.join(picks)}"


def _normalize_text(text: str) -> str:
    return " ".join(text.split())


def _generate_title(text: str) -> str:
    return " ".join(w.capitalize() for w in re.sub(r"[^\w\s]", "", text).split()[:6])


def _extract_keywords(text: str) -> str:
    counts: dict[str, int] = {}
    for word in _words(text):
        if word not in _STOPWORDS and len(word) > 2:
            counts[word] = counts.get(word, 0) + 1
    top = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:3]
    return f"keywords={', '.join(word for word, _ in top)}"


def _check_pii(text: str) -> str:
    emails = len(_EMAIL_RE.findall(text))
    phones = len(_PHONE_RE.findall(text))
    return f"pii_found={'true' if emails or phones else 'false'} emails={emails} phones={phones}"


def _quote_price(text: str) -> str:
    match = re.search(r"(\d+)\s*x\s*\$?(\d+(?:\.\d+)?)", text)
    if not match:
        return "quote=unparseable (expected e.g. '3 x $4.50')"
    quantity, unit = int(match.group(1)), float(match.group(2))
    discount = 0.1 if quantity >= 10 else 0.0
    total = quantity * unit * (1 - discount)
    return f"quote=quantity:{quantity} unit:{unit:.2f} discount:{int(discount * 100)}% total:{total:.2f}"


_CHECKLISTS = {
    "engineer": "1. Set up laptop 2. Get repo access 3. Run the test suite",
    "sales": "1. Set up laptop 2. Get CRM access 3. Shadow a demo call",
}


def _onboarding_checklist(text: str) -> str:
    words = set(_words(text))
    return next((c for role, c in _CHECKLISTS.items() if role in words), "1. Set up laptop 2. Meet your manager 3. Read the handbook")


def _skill(skill_id: str, name: str, description: str, handler: Callable[[str], str], example: str, expected: str) -> A2ASkillDef:
    return A2ASkillDef(skill_id, name, description, handler, example, expected)


A2A_AGENTS: list[A2AAgentDef] = [
    A2AAgentDef("ticket-triage-agent", "Triages and routes support tickets.", [
        _skill("triage_ticket", "Triage ticket", "Assign a priority to a support ticket.", _triage,
               "Production is down for all users", "priority=high"),
        _skill("route_ticket", "Route ticket", "Pick the team a ticket belongs to.", _route,
               "I can't login with SSO", "team=identity"),
    ]),
    A2AAgentDef("sentiment-agent", "Classifies the sentiment of text.", [
        _skill("classify_sentiment", "Classify sentiment", "positive / negative / neutral.", _classify_sentiment,
               "This is terrible and I hate it", "negative"),
    ]),
    A2AAgentDef("summarizer-agent", "Summarizes text.", [
        _skill("summarize", "Summarize", "One-sentence summary.", _summarize,
               "ADK builds agents. It is flexible.", "ADK builds agents."),
        _skill("summarize_bullets", "Summarize as bullets", "One bullet per sentence.", _summarize_bullets,
               "First point. Second point.", "- First point.\n- Second point."),
    ]),
    A2AAgentDef("translator-agent", "Translates English to Spanish.", [
        _skill("translate_to_spanish", "Translate to Spanish", "Word-by-word translation.", _translate,
               "hello friend", "hola amigo"),
    ]),
    A2AAgentDef("code-review-agent", "Flags TODOs, FIXMEs and stray prints.", [
        _skill("review_code", "Review code", "Count review markers in code.", _review_code,
               "x = 1  # TODO\nprint(x)", "todo=1 fixme=0 print_calls=1"),
    ]),
    A2AAgentDef("faq-agent", "Answers frequently asked questions.", [
        _skill("answer_faq", "Answer FAQ", "Answer from the FAQ.", _answer_faq,
               "How long does shipping take?", "Standard shipping takes 3-7 days."),
    ]),
    A2AAgentDef("scheduler-agent", "Proposes meeting slots.", [
        _skill("propose_slot", "Propose slot", "Suggest a meeting slot.", _propose_slot,
               "Can we meet on Thursday?", "Proposed slot: Thursday 10:00-10:30"),
    ]),
    A2AAgentDef("invoice-agent", "Totals invoice amounts.", [
        _skill("total_invoice", "Total invoice", "Sum every $ amount in the text.", _total_invoice,
               "Items: $10.50, $4.50 and $5", "total=20.00"),
    ]),
    A2AAgentDef("legal-clause-agent", "Detects common contract clauses.", [
        _skill("detect_clauses", "Detect clauses", "List the standard clauses present.", _detect_clauses,
               "This Agreement covers confidentiality and termination rights.", "clauses=confidentiality,termination"),
    ]),
    A2AAgentDef("seo-agent", "Writes meta descriptions.", [
        _skill("meta_description", "Meta description", "A <=155 character meta description.", _meta_description,
               "Build   agents with ADK.", "Build agents with ADK."),
    ]),
    A2AAgentDef("email-drafter-agent", "Drafts short emails.", [
        _skill("draft_email", "Draft email", "Wrap a message in an email.", _draft_email,
               "Your order shipped.", "Hi there,\n\nYour order shipped.\n\nBest regards,\nThe Team"),
    ]),
    A2AAgentDef("qa-agent", "Basic writing quality checks.", [
        _skill("check_text", "Check text", "Count double spaces and lowercase sentence starts.", _check_text,
               "Hello  world. this is bad.", "double_spaces=1 lowercase_sentence_starts=1"),
    ]),
    A2AAgentDef("risk-scorer-agent", "Scores account risk from notes.", [
        _skill("score_risk", "Score risk", "Weighted keyword risk score.", _score_risk,
               "customer filed a chargeback and is overdue", "risk_score=7 level=high"),
    ]),
    A2AAgentDef("recommender-agent", "Recommends related products.", [
        _skill("recommend_products", "Recommend products", "Accessories for what was bought.", _recommend,
               "I just bought a camera", "recommendations=tripod, sd card"),
    ]),
    A2AAgentDef("data-cleaner-agent", "Normalizes messy text.", [
        _skill("normalize_text", "Normalize text", "Collapse whitespace.", _normalize_text,
               "  too   many    spaces  ", "too many spaces"),
    ]),
    A2AAgentDef("title-generator-agent", "Generates titles.", [
        _skill("generate_title", "Generate title", "Six-word title-cased title.", _generate_title,
               "how to build multi agent systems with adk", "How To Build Multi Agent Systems"),
    ]),
    A2AAgentDef("keyword-extractor-agent", "Extracts keywords.", [
        _skill("extract_keywords", "Extract keywords", "Top three keywords.", _extract_keywords,
               "ADK agents use tools. Agents call tools and agents plan with ADK.", "keywords=agents, adk, tools"),
    ]),
    A2AAgentDef("compliance-agent", "Flags personal data.", [
        _skill("check_pii", "Check PII", "Count emails and phone numbers.", _check_pii,
               "Mail jo@x.com or call +1 555 123 4567", "pii_found=true emails=1 phones=1"),
    ]),
    A2AAgentDef("pricing-agent", "Quotes prices with volume discounts.", [
        _skill("quote_price", "Quote price", "Parse 'N x $P' and quote (10% off at 10+).", _quote_price,
               "3 x $4.50", "quote=quantity:3 unit:4.50 discount:0% total:13.50"),
    ]),
    A2AAgentDef("onboarding-agent", "Builds onboarding checklists.", [
        _skill("onboarding_checklist", "Onboarding checklist", "Checklist for a new hire's role.", _onboarding_checklist,
               "new engineer joining Monday", "1. Set up laptop 2. Get repo access 3. Run the test suite"),
    ]),
]
