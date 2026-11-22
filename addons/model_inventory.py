#!/usr/bin/env python3
"""Validated OpenRouter model metadata and price snapshot helpers.

The bundled JSON remains readable as the original ``{models: [...]}`` format.
Refreshes only update provider metadata. User-managed model membership is stored
separately by Argus.
"""
from __future__ import annotations

import copy
import json
import math
import os
import re
import tempfile
import threading
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"
DEFAULT_TIMEOUT_SECONDS = 12
DEFAULT_MAX_RESPONSE_BYTES = 8 * 1024 * 1024
_refresh_lock = threading.Lock()
_MAX_MODELS = 100_000
_MODEL_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+:/-]{0,239}$")


class InventoryError(ValueError):
    """The provider response or local inventory is unusable."""


def _ensure_json_safe(value):
    try:
        json.dumps(value, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError, OverflowError) as exc:
        raise InventoryError("Model inventory contains data that cannot be stored as JSON") from exc


def _reject_json_constant(value):
    raise InventoryError(f"Model inventory contains non-finite JSON number: {value}")


def models_path() -> Path:
    configured = os.getenv("ARGUS_MODELS_CONFIG")
    if configured:
        return Path(configured)
    return Path(__file__).resolve().parent.parent / "data" / "openrouter_models_cache.json"


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def is_stale(fetched_at, *, now=None, max_age=timedelta(hours=24)):
    if not isinstance(fetched_at, str) or not fetched_at.strip():
        return True
    try:
        parsed = datetime.fromisoformat(fetched_at.strip().replace("Z", "+00:00"))
    except ValueError:
        return True
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    current = now or datetime.now(timezone.utc)
    if current.tzinfo is None:
        current = current.replace(tzinfo=timezone.utc)
    age = current.astimezone(timezone.utc) - parsed.astimezone(timezone.utc)
    return age > max_age or age < timedelta(0)


def _number(value):
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        raise InventoryError("Model price must be numeric, not a boolean")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise InventoryError("Model price must be numeric") from exc
    if result == -1:
        return None
    if not math.isfinite(result) or result < 0:
        raise InventoryError("Model price must be finite and non-negative")
    return result


def _tokens_per_million(price):
    if price is None or price == "":
        return None
    if isinstance(price, bool):
        raise InventoryError("OpenRouter token rate must be numeric, not a boolean")
    try:
        parsed = float(price)
    except (TypeError, ValueError, OverflowError) as exc:
        raise InventoryError("OpenRouter token rate must be numeric") from exc
    if parsed == -1:
        # OpenRouter uses -1 for dynamic/router pricing without a fixed token rate.
        return None
    if not math.isfinite(parsed) or parsed < 0 or parsed > float.fromhex("0x1.fffffffffffffp+1023") / 1_000_000:
        raise InventoryError("OpenRouter token rate is outside the supported numeric range")
    return round(parsed * 1_000_000, 12)


def _model_id(value):
    if not isinstance(value, str):
        return None
    model_id = value.strip()
    if not _MODEL_ID_RE.fullmatch(model_id):
        return None
    parts = model_id.split("/")
    if len(parts) != 2 or any(part in {"", ".", ".."} for part in parts):
        return None
    return model_id


def validate_model_id(value):
    """Return a safe exact provider model ID, or None for an invalid value."""
    return _model_id(value)


_LEGACY_ASSIGNMENT_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+\-/]{0,119}$")


def validate_assignment_id(value):
    """Accept current catalog IDs plus safe IDs allowed by older seat configs."""
    model_id = _model_id(value)
    if model_id:
        return model_id
    if not isinstance(value, str):
        return None
    model_id = value.strip()
    if (len(model_id) <= 120 and _LEGACY_ASSIGNMENT_ID_RE.fullmatch(model_id)
            and ".." not in model_id and not model_id.startswith("//")):
        return model_id
    return None


def _text_list(value):
    if not isinstance(value, list):
        return []
    return sorted({item.strip() for item in value if isinstance(item, str) and item.strip()})


def _tier_rows(raw):
    """Normalize supplied prompt-token tiers without inferring missing rates."""
    if not isinstance(raw, list):
        if raw is not None:
            raise InventoryError("OpenRouter price tiers must be a list")
        return []
    output = []
    for tier in raw:
        if not isinstance(tier, dict):
            raise InventoryError("OpenRouter price tier must be an object")
        _validate_override_rates(tier)
        time_conditioned = any(key in tier for key in ("utc_start", "utc_end", "utc_days"))
        threshold = tier.get("min_prompt_tokens", tier.get("context_size"))
        if time_conditioned:
            if threshold is not None:
                _validate_prompt_threshold(threshold)
            continue
        threshold = _validate_prompt_threshold(threshold)
        if "usd_per_mtok_in" in tier or "usd_per_mtok_out" in tier:
            pin = _number(tier.get("usd_per_mtok_in"))
            pout = _number(tier.get("usd_per_mtok_out"))
        else:
            pin = _tokens_per_million(tier.get("prompt"))
            pout = _tokens_per_million(tier.get("completion"))
        output.append({"min_prompt_tokens": threshold,
                       "usd_per_mtok_in": pin, "usd_per_mtok_out": pout})
    return sorted(output, key=lambda row: row["min_prompt_tokens"])


_PRICING_DIMENSIONS = (
    "prompt", "completion", "input_cache_read", "input_cache_write",
    "cache_read", "cache_write", "image", "audio", "web_search", "request",
    "input_audio_cache", "internal_reasoning",
)


def _validate_override_rates(override):
    for key in _PRICING_DIMENSIONS:
        if key not in override:
            continue
        if key in {"prompt", "completion"}:
            _tokens_per_million(override[key])
        else:
            _number(override[key])


def _validate_prompt_threshold(value):
    if isinstance(value, bool):
        raise InventoryError("OpenRouter price tier threshold must be an integer")
    try:
        threshold = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise InventoryError("OpenRouter price tier threshold must be an integer") from exc
    if threshold < 0:
        raise InventoryError("OpenRouter price tier threshold cannot be negative")
    return threshold


def normalize_row(row, *, source="OpenRouter", fetched_at=None):
    if not isinstance(row, dict):
        return None
    model_id = _model_id(row.get("id"))
    if not model_id:
        return None
    pricing = row.get("pricing")
    if pricing is not None and not isinstance(pricing, dict):
        raise InventoryError("OpenRouter pricing must be an object")
    pricing = pricing or {}
    supplied_tiers = row.get("price_tiers")
    if supplied_tiers is None:
        supplied_tiers = pricing.get("overrides")
    price_tiers = _tier_rows(supplied_tiers)
    overrides = pricing.get("overrides")
    if overrides is not None and not isinstance(overrides, list):
        raise InventoryError("OpenRouter price overrides must be a list")
    has_conditional_pricing = any(
        any(key in override for key in ("utc_start", "utc_end", "utc_days"))
        for override in (overrides or []) if isinstance(override, dict)
    )
    _validate_override_rates(pricing)
    input_rate = row.get("usd_per_mtok_in")
    output_rate = row.get("usd_per_mtok_out")
    if input_rate is None:
        input_rate = _tokens_per_million(pricing.get("prompt"))
    else:
        input_rate = _number(input_rate)
    if output_rate is None:
        output_rate = _tokens_per_million(pricing.get("completion"))
    else:
        output_rate = _number(output_rate)
    name = row.get("name") or row.get("label")
    name = name.strip()[:240] if isinstance(name, str) else model_id
    description = row.get("description")
    description = description.strip()[:4000] if isinstance(description, str) else ""
    context = row.get("context_length") or row.get("context_size")
    try:
        context = max(0, int(context)) if context is not None else None
    except (TypeError, ValueError):
        context = None
    top_provider = row.get("top_provider") if isinstance(row.get("top_provider"), dict) else {}
    output_limit = row.get("max_output_tokens", top_provider.get("max_completion_tokens"))
    try:
        output_limit = max(0, int(output_limit)) if output_limit is not None else None
    except (TypeError, ValueError):
        output_limit = None
    parameters = _text_list(row.get("supported_parameters"))
    architecture = row.get("architecture") if isinstance(row.get("architecture"), dict) else {}
    modality_list = _text_list(row.get("input_modalities", architecture.get("input_modalities", [])))
    output_modalities = _text_list(row.get("output_modalities", architecture.get("output_modalities", [])))
    preview = bool(row.get("preview") or row.get("is_preview") or "preview" in model_id.lower())
    normalized_pricing = copy.deepcopy(pricing)
    for key in _PRICING_DIMENSIONS:
        if key in normalized_pricing:
            if key in {"prompt", "completion"}:
                if _tokens_per_million(normalized_pricing[key]) is None:
                    normalized_pricing[key] = None
            elif _number(normalized_pricing[key]) is None:
                normalized_pricing[key] = None
    for override in normalized_pricing.get("overrides", []):
        if not isinstance(override, dict):
            continue
        for key in _PRICING_DIMENSIONS:
            if key in override:
                if key in {"prompt", "completion"}:
                    if _tokens_per_million(override[key]) is None:
                        override[key] = None
                elif _number(override[key]) is None:
                    override[key] = None
    normalized = {
        "id": model_id,
        "label": name,
        "name": name,
        "publisher": model_id.split("/", 1)[0],
        "description": description,
        "context_length": context,
        "max_output_tokens": output_limit,
        "input_modalities": modality_list,
        "output_modalities": output_modalities,
        "supported_parameters": parameters,
        "preview": preview,
        "available": row.get("available", True) is not False,
        "availability_status": row.get("availability_status") or
        ("available" if row.get("available", True) is not False else "unavailable"),
        "effort": "reasoning" in parameters or bool(row.get("effort")),
        "usd_per_mtok_in": input_rate,
        "usd_per_mtok_out": output_rate,
        "price_tiers": price_tiers,
        "has_conditional_pricing": has_conditional_pricing,
        "legacy_ids": _text_list(row.get("legacy_ids")),
        "pricing": normalized_pricing,
        "price_source": row.get("price_source") or source,
        "price_fetched_at": row.get("price_fetched_at") or fetched_at,
        "price_status": row.get("price_status") or "current",
        "compatibility_unknown": not bool(modality_list and parameters),
    }
    # Keep provider-specific price dimensions for clear UI details, while the
    # canonical token rates above remain the only rates consumed by estimates.
    for key in ("cache_read", "cache_write", "input_cache_read", "input_cache_write",
                "request", "image", "audio", "web_search"):
        value = _number(pricing.get(key))
        if value is not None:
            normalized["pricing"][key] = pricing[key]
    return normalized


def normalize_openrouter(payload, *, fetched_at=None):
    _ensure_json_safe(payload)
    if not isinstance(payload, dict) or not isinstance(payload.get("data"), list):
        raise InventoryError("OpenRouter response must contain a data list")
    raw_rows = payload["data"]
    if not raw_rows or len(raw_rows) > _MAX_MODELS:
        raise InventoryError("OpenRouter returned an empty or oversized model list")
    timestamp = fetched_at or _utc_now()
    models = [normalized for row in raw_rows
              if (normalized := normalize_row(row, source="OpenRouter", fetched_at=timestamp))]
    if not models:
        raise InventoryError("OpenRouter response contained no valid model IDs")
    unique = {row["id"]: row for row in models}
    return {"schema_version": 2,
            "pricing_note": "OpenRouter public catalog rates. Routed prices, cache, tools, and usage dimensions may differ.",
            "source": "OpenRouter public models catalog",
            "fetched_at": timestamp,
            "models": [unique[key] for key in sorted(unique)]}


def _validate_snapshot(data):
    _ensure_json_safe(data)
    if not isinstance(data, dict):
        raise InventoryError("Model inventory must be an object")
    rows = data.get("models")
    if not isinstance(rows, list):
        raise InventoryError("Model inventory must contain a models list")
    normalized = []
    seen = set()
    for row in rows:
        model = normalize_row(row, source=data.get("source") or "Local model inventory",
                              fetched_at=data.get("fetched_at"))
        if not model or model["id"] in seen:
            continue
        seen.add(model["id"])
        normalized.append(model)
    if rows and not normalized:
        raise InventoryError("Model inventory contains no valid models")
    return {**data, "models": normalized,
            "pricing_note": str(data.get("pricing_note") or ""),
            "source": str(data.get("source") or "Bundled OpenRouter inventory"),
            "fetched_at": data.get("fetched_at"),
            "stale": bool(data.get("stale", False)),
            "unavailable": bool(data.get("unavailable", False))}


def load_snapshot(path=None):
    file = Path(path) if path is not None else models_path()
    if not file.is_file():
        default_cache = models_path()
        if (not os.getenv("ARGUS_MODELS_CONFIG") and file.resolve() == default_cache.resolve()):
            seed = Path(__file__).resolve().parent / "openrouter_models.json"
            if seed.is_file():
                try:
                    seed_data = json.loads(seed.read_text(encoding="utf-8"),
                                           parse_constant=_reject_json_constant)
                except (OSError, json.JSONDecodeError) as exc:
                    raise InventoryError(f"Could not read bundled model seed: {exc}") from exc
                snapshot = _validate_snapshot(seed_data)
                snapshot.update({"source": snapshot.get("source") or "Bundled OpenRouter inventory",
                                 "fetched_at": snapshot.get("fetched_at"), "stale": True,
                                 "unavailable": False})
                return snapshot
        return {"models": [], "pricing_note": "", "source": "Unavailable",
                "fetched_at": None, "stale": True, "unavailable": True}
    try:
        data = json.loads(file.read_text(encoding="utf-8"), parse_constant=_reject_json_constant)
    except (OSError, json.JSONDecodeError) as exc:
        raise InventoryError(f"Could not read model inventory: {exc}") from exc
    # Legacy files sometimes used a raw array, and the original bundled form is
    # an object with only pricing_note and models.
    if isinstance(data, list):
        data = {"models": data}
    snapshot = _validate_snapshot(data)
    snapshot["stale"] = is_stale(snapshot.get("fetched_at"))
    return snapshot


def _read_existing(path):
    try:
        data = json.loads(path.read_text(encoding="utf-8"), parse_constant=_reject_json_constant)
    except (OSError, json.JSONDecodeError) as exc:
        raise InventoryError(f"Could not read existing model inventory: {exc}") from exc
    if isinstance(data, list):
        data = {"models": data}
    return _validate_snapshot(data)


def _merge_snapshot(candidate, previous, seed):
    """Keep last-known-good rates and retired rows when publishing a refresh."""
    incoming = {row["id"]: copy.deepcopy(row) for row in candidate["models"]}
    older = {}
    for snapshot in (seed, previous):
        for row in snapshot.get("models", []):
            older[row["id"]] = row
    for model_id, current in incoming.items():
        old = older.get(model_id)
        if old is None:
            continue
        retained_rates = False
        complete_current_rates = (current.get("usd_per_mtok_in") is not None
                                  and current.get("usd_per_mtok_out") is not None)
        for key in ("usd_per_mtok_in", "usd_per_mtok_out"):
            if current.get(key) is None and old.get(key) is not None:
                current[key] = old[key]
                retained_rates = True
        if (not complete_current_rates and not current.get("price_tiers")
                and old.get("price_tiers")):
            current["price_tiers"] = copy.deepcopy(old["price_tiers"])
            retained_rates = True
        if not current.get("pricing") and old.get("pricing"):
            current["pricing"] = copy.deepcopy(old["pricing"])
            retained_rates = True
        if retained_rates:
            current["price_source"] = old.get("price_source") or old.get("source") or "Last known OpenRouter rate"
            current["price_fetched_at"] = old.get("price_fetched_at")
            current["price_status"] = "stale"
        current["available"] = True
        current["availability_status"] = "available"
    for model_id, old in older.items():
        if model_id in incoming:
            continue
        retired = copy.deepcopy(old)
        retired["available"] = False
        retired["availability_status"] = "unavailable"
        retired["price_status"] = "stale"
        incoming[model_id] = retired
    return {**candidate, "models": [incoming[key] for key in sorted(incoming)]}


def write_snapshot(path, snapshot):
    """Atomically replace the cache after validating the complete candidate."""
    file = Path(path)
    validated = _validate_snapshot(snapshot)
    if not validated["models"]:
        raise InventoryError("Refusing to replace a usable catalog with an empty one")
    previous = _read_existing(file) if file.is_file() else {"models": []}
    seed_path = Path(__file__).resolve().parent / "openrouter_models.json"
    seed = _read_existing(seed_path) if seed_path.is_file() and seed_path.resolve() != file.resolve() else {"models": []}
    validated = _validate_snapshot(_merge_snapshot(validated, previous, seed))
    file.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(prefix=f".{file.name}.", suffix=".tmp", dir=file.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(validated, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_name, file)
    except Exception:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise
    return validated


def fetch_openrouter(*, timeout=DEFAULT_TIMEOUT_SECONDS, max_bytes=DEFAULT_MAX_RESPONSE_BYTES):
    request = Request(OPENROUTER_MODELS_URL, headers={"Accept": "application/json",
                                                      "User-Agent": "Argus model catalog"})
    try:
        with urlopen(request, timeout=timeout) as response:
            raw = response.read(max_bytes + 1)
    except (HTTPError, URLError, TimeoutError, OSError) as exc:
        raise InventoryError(f"OpenRouter catalog refresh failed: {exc}") from exc
    if len(raw) > max_bytes:
        raise InventoryError("OpenRouter catalog response exceeded the size limit")
    try:
        payload = json.loads(raw, parse_constant=_reject_json_constant)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise InventoryError("OpenRouter catalog returned invalid JSON") from exc
    return normalize_openrouter(payload)


def refresh(path=None, *, fetcher=None):
    """Fetch and publish one complete inventory, serializing local refreshes."""
    target = Path(path) if path is not None else models_path()
    with _refresh_lock:
        candidate = (fetcher or fetch_openrouter)()
        return write_snapshot(target, candidate)


@contextmanager
def refresh_guard():
    """Serialize provider refresh attempts within the dashboard process."""
    with _refresh_lock:
        yield


def compatible_seats(row):
    """Return agent seat IDs whose synchronous calls can use this model.

    Missing provider capability metadata is treated as unknown rather than a
    false incompatibility. Explicit media-only and batch offerings are excluded.
    """
    model = normalize_row(row)
    if not model or not model["available"]:
        return []
    model_id = model["id"].lower()
    modalities = {value.lower() for value in model["input_modalities"]}
    output_modalities = {value.lower() for value in model["output_modalities"]}
    if not modalities:
        return []
    if modalities and "text" not in modalities:
        return []
    if output_modalities and "text" not in output_modalities:
        return []
    if any(label in model_id for label in ("-batch", "/batch", "image-gen", "video-gen", "embedding", "embed/")):
        return []
    if any(token in model_id for token in ("live-audio", "realtime-audio")):
        return []
    params = {value.lower() for value in model["supported_parameters"]}
    all_seats = ("theme_scout", "market", "sentiment", "news", "fundamentals", "bull", "bear",
                 "research_manager", "trader", "risk_aggressive", "risk_conservative", "risk_neutral",
                 "portfolio_manager", "memory", "fm_manager", "fm_aggressive", "fm_conservative", "fm_neutral")
    if not params:
        return []
    # These exact call paths bind tools or structured output. Other native
    # agents call the model as plain text and need no additional capability.
    structured_native = {"research_manager", "trader", "portfolio_manager"}
    tools = {"tools", "tool_choice"} & params
    structured_tools = bool(tools or "structured_outputs" in params)
    schema_output = {"json_schema", "structured_outputs"} & params
    fund_json = schema_output | ({"response_format"} & params)
    tool_seats = {"market", "news", "fundamentals"}
    plain_text = {"sentiment", "bull", "bear", "risk_aggressive", "risk_conservative", "risk_neutral", "memory"}
    return [seat for seat in all_seats
            if (seat in plain_text
                or (seat in tool_seats and bool(tools))
                or (seat in structured_native and structured_tools)
                or (seat == "theme_scout" and bool(schema_output))
                or (seat in {"fm_manager", "fm_aggressive", "fm_conservative", "fm_neutral"}
                    and bool(fund_json)))]


def effective_rates(model_ids, snapshot=None):
    """Return stable execution-relevant price/capability rows for a run digest."""
    inventory = snapshot if snapshot is not None else load_snapshot()
    by_id = {row.get("id"): row for row in inventory.get("models", []) if isinstance(row, dict)}
    projection = []
    for model_id in sorted({str(value) for value in model_ids if value}):
        row = by_id.get(model_id)
        if row is None:
            projection.append({"id": model_id, "unpriced": True})
            continue
        projection.append({key: row.get(key) for key in (
            "id", "available", "usd_per_mtok_in", "usd_per_mtok_out", "price_tiers",
            "input_modalities", "output_modalities", "supported_parameters",
            "has_conditional_pricing")})
        projection[-1]["pricing"] = {
            key: _tokens_per_million((row.get("pricing") or {}).get(key))
            for key in ("prompt", "completion", "input_cache_read", "input_cache_write")
            if key in (row.get("pricing") or {})
        }
    return projection


def token_rates(model, prompt_tokens=0):
    """Resolve token rates for one call, including a matching prompt tier."""
    if not isinstance(model, dict):
        return None, None
    input_rate = _number(model.get("usd_per_mtok_in"))
    output_rate = _number(model.get("usd_per_mtok_out"))
    try:
        prompt_tokens = int(prompt_tokens or 0)
    except (TypeError, ValueError):
        prompt_tokens = 0
    tiers = _tier_rows(model.get("price_tiers"))
    selected = [tier for tier in tiers if prompt_tokens >= tier["min_prompt_tokens"]]
    if selected:
        tier = selected[-1]
        if tier["usd_per_mtok_in"] is not None:
            input_rate = tier["usd_per_mtok_in"]
        if tier["usd_per_mtok_out"] is not None:
            output_rate = tier["usd_per_mtok_out"]
    return input_rate, output_rate
