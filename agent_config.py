#!/usr/bin/env python3
"""Seat configuration for Argus and the add-on runners.

TradingAgents upstream is not edited. run_native.py calls install_native_seats()
before the CLI starts, and the fund manager and theme scout read the same file.

agent_config.json stores only overrides. Defaults come from this module and from
the role text already in the add-on and upstream sources (read, not modified).
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

LOCKED_MODEL = "anthropic/claude-opus-5.5"
CONSERVATIVE_SEATS = ("risk_conservative", "fm_conservative")
EFFORTS = ("low", "medium", "high")
_MODEL_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+\-/]{0,119}$")
_MAX_PROMPT = 60000

# Token totals from fund-manager run 2026-11-06 172558_251bfd (one debate round).
# Draft + final are one fund-manager seat. Native seats have no per-agent usage.
_EST_TOKENS = {
    "fm_manager": (50599 + 69839, 14476 + 13872),
    "fm_aggressive": (63846, 1975),
    "fm_conservative": (98436, 3658),
    "fm_neutral": (66984, 4229),
}

_TABLE = (
    " Make sure to append a Markdown table at the end of the report to organize "
    "key points in the report, organized and easy to read."
)


class AgentConfigError(ValueError):
    pass


def addons_dir() -> Path:
    env = os.getenv("ARGUS_ADDONS_DIR")
    if env:
        return Path(env)
    here = Path(__file__).resolve().parent
    if (here / "agent_config.py").is_file():
        return here
    docker = Path("/home/appuser/addons")
    if (docker / "agent_config.py").is_file():
        return docker
    return here


def config_path() -> Path:
    env = os.getenv("ARGUS_AGENT_CONFIG")
    if env:
        return Path(env)
    return addons_dir() / "agent_config.json"


def models_path() -> Path:
    env = os.getenv("ARGUS_MODELS_CONFIG")
    if env:
        return Path(env)
    return addons_dir() / "openrouter_models.json"


def ta_root() -> Path:
    env = os.getenv("TRADINGAGENTS_ROOT")
    if env:
        return Path(env)
    root = addons_dir().parent
    if (root / "tradingagents").is_dir():
        return root
    return root


def _read_source(rel: str) -> str:
    path = ta_root() / rel
    if path.is_file():
        return path.read_text(encoding="utf-8")
    return ""


def _between(text: str, pattern: str, end: str = '"""') -> str:
    match = re.search(pattern, text, re.S)
    if not match:
        return ""
    rest = text[match.end():]
    stop = rest.find(end)
    if stop < 0:
        return ""
    return rest[:stop].strip()


def _static_block(rel: str, start: str, end: str) -> str:
    text = _read_source(rel)
    i = text.find(start)
    if i < 0:
        return start
    j = text.find(end, i)
    block = text[i:j if j > i else i + len(start)]
    return block.strip()


def _market_prompt() -> str:
    body = _between(_read_source("tradingagents/agents/analysts/market_analyst.py"), r"system_message = \(\s*\"\"\"")
    if not body:
        return "You are a trading assistant tasked with analyzing financial markets."
    if _TABLE.strip() not in body:
        body += _TABLE
    return body


def _fundamentals_prompt() -> str:
    text = _read_source("tradingagents/agents/analysts/fundamentals_analyst.py")
    first = _between(text, r"system_message = \(\s*\"", end='"')
    tools = _between(text, r"\+\s*\" Use the available tools:", end='"')
    if not first:
        return "You are a researcher tasked with analyzing fundamental information over the past week about a company."
    tools = (" Use the available tools:" + tools) if tools else ""
    return first + _TABLE + tools


def _news_prompt() -> str:
    text = _read_source("tradingagents/agents/analysts/news_analyst.py")
    first = _between(text, r"system_message = \(\s*f\"", end='"')
    if not first:
        return "You are a news researcher tasked with analyzing recent news and trends over the past week."
    return first.replace("{asset_label}", "company") + _TABLE


def _sentiment_prompt() -> str:
    text = _read_source("tradingagents/agents/analysts/sentiment_analyst.py")
    start = "## How to analyze this data (best practices)"
    block = _static_block("tradingagents/agents/analysts/sentiment_analyst.py", start, "{get_language_instruction()")
    if start not in text:
        return "You are a financial market sentiment analyst."
    return block


def _role_prompt(rel: str, start: str, end: str, fallback: str) -> str:
    text = _read_source(rel)
    if start not in text:
        return fallback
    return _static_block(rel, start, end)


def _fm_piece(marker: str, fallback: str) -> str:
    text = _read_source("addons/fund_manager/fund_manager.py")
    body = _between(text, re.escape(marker) + r"\s*\"\"\"")
    return body or fallback


def _scout_prompt() -> str:
    text = _read_source("addons/theme_scout/theme_scout.py")
    body = _between(text, r"SCOUT_SYSTEM = \"\"\"")
    return body or "You are the theme scout for a concentrated US equity book."


def _seat_defs() -> list[dict]:
    """Static seat metadata. system/anchor are filled by catalog()."""
    quick = "openai/gpt-6-luna"
    deep = "openai/gpt-6.1-sol"
    native = "Per-ticker desk"
    fund = "Fund manager"
    return [
        {"id": "theme_scout", "label": "Theme scout", "group": "Theme scout", "pipeline": "scout",
         "model": deep, "effort": "high", "max_tokens": 6000, "tier": "deep"},
        {"id": "market", "label": "Market analyst", "group": native, "pipeline": "native",
         "model": quick, "effort": None, "max_tokens": None, "tier": "quick"},
        {"id": "sentiment", "label": "Sentiment analyst", "group": native, "pipeline": "native",
         "model": quick, "effort": None, "max_tokens": None, "tier": "quick"},
        {"id": "news", "label": "News analyst", "group": native, "pipeline": "native",
         "model": quick, "effort": None, "max_tokens": None, "tier": "quick"},
        {"id": "fundamentals", "label": "Fundamentals analyst", "group": native, "pipeline": "native",
         "model": quick, "effort": None, "max_tokens": None, "tier": "quick"},
        {"id": "bull", "label": "Bull researcher", "group": native, "pipeline": "native",
         "model": quick, "effort": None, "max_tokens": None, "tier": "quick"},
        {"id": "bear", "label": "Bear researcher", "group": native, "pipeline": "native",
         "model": quick, "effort": None, "max_tokens": None, "tier": "quick"},
        {"id": "research_manager", "label": "Research manager", "group": native, "pipeline": "native",
         "model": deep, "effort": None, "max_tokens": None, "tier": "deep"},
        {"id": "trader", "label": "Trader", "group": native, "pipeline": "native",
         "model": quick, "effort": None, "max_tokens": None, "tier": "quick"},
        {"id": "risk_aggressive", "label": "Risk aggressive", "group": native, "pipeline": "native",
         "model": quick, "effort": None, "max_tokens": None, "tier": "quick"},
        {"id": "risk_conservative", "label": "Risk conservative", "group": native, "pipeline": "native",
         "model": LOCKED_MODEL, "effort": None, "max_tokens": None, "tier": "quick"},
        {"id": "risk_neutral", "label": "Risk neutral", "group": native, "pipeline": "native",
         "model": quick, "effort": None, "max_tokens": None, "tier": "quick"},
        {"id": "portfolio_manager", "label": "Portfolio manager", "group": native, "pipeline": "native",
         "model": deep, "effort": None, "max_tokens": None, "tier": "deep"},
        {"id": "memory", "label": "Memory reflection", "group": native, "pipeline": "native",
         "model": quick, "effort": None, "max_tokens": None, "tier": "quick"},
        {"id": "fm_manager", "label": "Fund manager", "group": fund, "pipeline": "fund",
         "model": deep, "effort": "high", "max_tokens": 24000, "tier": "deep"},
        {"id": "fm_aggressive", "label": "Risk aggressive", "group": fund, "pipeline": "fund",
         "model": deep, "effort": "medium", "max_tokens": 12000, "tier": "deep"},
        {"id": "fm_conservative", "label": "Risk conservative", "group": fund, "pipeline": "fund",
         "model": LOCKED_MODEL, "effort": "medium", "max_tokens": 12000, "tier": "deep"},
        {"id": "fm_neutral", "label": "Risk neutral", "group": fund, "pipeline": "fund",
         "model": quick, "effort": "medium", "max_tokens": 12000, "tier": "quick"},
    ]


def _prompts() -> dict[str, str]:
    bull = _role_prompt(
        "tradingagents/agents/researchers/bull_researcher.py",
        "Key points to focus on:", "Resources available:",
        "Key points to focus on:\n- Growth Potential\n- Competitive Advantages\n- Positive Indicators",
    )
    bear = _role_prompt(
        "tradingagents/agents/researchers/bear_researcher.py",
        "Key points to focus on:", "Resources available:",
        "Key points to focus on:\n- Risks and Challenges\n- Competitive Weaknesses\n- Negative Indicators",
    )
    return {
        "theme_scout": _scout_prompt(),
        "market": _market_prompt(),
        "sentiment": _sentiment_prompt(),
        "news": _news_prompt(),
        "fundamentals": _fundamentals_prompt(),
        "bull": bull,
        "bear": bear,
        "research_manager": _role_prompt(
            "tradingagents/agents/managers/research_manager.py",
            "As the Research Manager and debate facilitator", "{instrument_context}",
            "As the Research Manager and debate facilitator, your role is to critically evaluate this round of debate and deliver a clear, actionable investment plan for the trader.",
        ),
        "trader": "You are a trading agent analyzing market data to make investment decisions. Based on your analysis, provide a specific recommendation to buy, sell, or hold.",
        "risk_aggressive": _role_prompt(
            "tradingagents/agents/risk_mgmt/aggressive_debator.py",
            "As the Aggressive Risk Analyst", "Here is the trader's decision:",
            "As the Aggressive Risk Analyst, your role is to actively champion high-reward, high-risk opportunities.",
        ),
        "risk_conservative": _role_prompt(
            "tradingagents/agents/risk_mgmt/conservative_debator.py",
            "As the Conservative Risk Analyst", "Here is the trader's decision:",
            "As the Conservative Risk Analyst, your primary objective is to protect assets, minimize volatility, and ensure steady, reliable growth.",
        ),
        "risk_neutral": _role_prompt(
            "tradingagents/agents/risk_mgmt/neutral_debator.py",
            "As the Neutral Risk Analyst", "Here is the trader's decision:",
            "As the Neutral Risk Analyst, your role is to provide a balanced perspective, weighing both the potential benefits and risks of the trader's decision or plan.",
        ),
        "portfolio_manager": _role_prompt(
            "tradingagents/agents/managers/portfolio_manager.py",
            "As the Portfolio Manager", "{instrument_context}",
            "As the Portfolio Manager, synthesize the risk analysts' debate and deliver the final trading decision.",
        ),
        "memory": (
            "You are a trading analyst reviewing your own past decision now that the outcome is known.\n"
            "Write exactly 2-4 sentences of plain prose (no bullets, no headers, no markdown)."
        ),
        "fm_manager": _fm_piece(
            "FM_SYSTEM =",
            "You are the senior Fund Manager / CIO sitting on top of a TradingAgents desk.",
        ),
        "fm_aggressive": _fm_piece(
            '"aggressive":',
            "You are the AGGRESSIVE portfolio risk analyst on the fund's risk committee.",
        ),
        "fm_conservative": _fm_piece(
            '"conservative":',
            "You are the CONSERVATIVE portfolio risk analyst on the fund's risk committee.",
        ),
        "fm_neutral": _fm_piece(
            '"neutral":',
            "You are the NEUTRAL portfolio risk analyst on the fund's risk committee.",
        ),
    }


_CATALOG: dict[str, dict] | None = None


def catalog() -> dict[str, dict]:
    global _CATALOG
    if _CATALOG is not None:
        return _CATALOG
    prompts = _prompts()
    out = {}
    for row in _seat_defs():
        seat = dict(row)
        seat["system"] = prompts[seat["id"]]
        seat["anchor"] = seat["system"]
        seat["temperature"] = None
        seat["supports_effort"] = True
        seat["supports_temperature"] = seat["pipeline"] == "native"
        seat["supports_max_tokens"] = True
        tokens = _EST_TOKENS.get(seat["id"])
        seat["est_input_tokens"] = tokens[0] if tokens else None
        seat["est_output_tokens"] = tokens[1] if tokens else None
        seat["locked"] = False
        out[seat["id"]] = seat
    _CATALOG = out
    return out


def load_models() -> dict:
    path = models_path()
    if not path.is_file():
        return {"models": [], "pricing_note": ""}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise AgentConfigError("openrouter_models.json must be an object")
    return data


def load_overrides() -> dict:
    path = config_path()
    if not path.is_file():
        return {"seats": {}}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise AgentConfigError("agent_config.json must be an object")
    seats = data.get("seats") or {}
    if not isinstance(seats, dict):
        raise AgentConfigError("agent_config.json seats must be an object")
    return {"seats": seats}


def _model_row(model_id: str, models: list[dict]) -> dict | None:
    for row in models:
        if isinstance(row, dict) and row.get("id") == model_id:
            return row
    return None


def estimate_cost(seat: dict, model_row: dict | None) -> float | None:
    inn = seat.get("est_input_tokens")
    out = seat.get("est_output_tokens")
    if model_row is None or inn is None or out is None:
        return None
    pin = model_row.get("usd_per_mtok_in")
    pout = model_row.get("usd_per_mtok_out")
    if not isinstance(pin, (int, float)) or not isinstance(pout, (int, float)):
        return None
    return round(float(inn) / 1_000_000 * float(pin) + float(out) / 1_000_000 * float(pout), 4)


def opus_warning(model: str, seat_id: str) -> str | None:
    if seat_id in CONSERVATIVE_SEATS or "opus" not in (model or "").lower():
        return None
    return f"{seat_id} uses {model}. Opus on this seat is allowed and is not blocked."


def resolved_seat(seat_id: str) -> dict:
    base = catalog().get(seat_id)
    if base is None:
        raise AgentConfigError(f"Unknown seat: {seat_id}")
    saved = load_overrides()
    over = saved["seats"].get(seat_id) or {}
    if not isinstance(over, dict):
        over = {}
    seat = dict(base)
    for key in ("model", "effort", "temperature", "max_tokens", "system"):
        if key in over:
            seat[key] = over[key]
    seat["default_system"] = base["system"]
    seat["default_model"] = base["model"]
    seat["default_effort"] = base["effort"]
    seat["default_max_tokens"] = base["max_tokens"]
    seat["unlocked"] = False
    seat["locked"] = False
    models = load_models().get("models") or []
    seat["opus_warning"] = opus_warning(str(seat.get("model") or ""), seat_id)
    seat["est_cost_usd"] = estimate_cost(seat, _model_row(str(seat.get("model") or ""), models))
    seat["customized"] = _is_custom(base, seat)
    return seat


def _is_custom(base: dict, seat: dict) -> bool:
    for key in ("model", "effort", "temperature", "max_tokens", "system"):
        if seat.get(key) != base.get(key):
            return True
    return False


def resolved_seats() -> list[dict]:
    return [resolved_seat(row["id"]) for row in _seat_defs()]


def public_payload() -> dict:
    models = load_models()
    seats = []
    for seat in resolved_seats():
        seats.append({k: seat.get(k) for k in (
            "id", "label", "group", "pipeline", "model", "effort", "temperature", "max_tokens",
            "system", "default_system", "default_model", "default_effort", "default_max_tokens",
            "supports_effort", "supports_temperature", "supports_max_tokens",
            "locked", "unlocked", "opus_warning", "est_cost_usd", "est_input_tokens",
            "est_output_tokens", "customized", "tier",
        )})
    warnings = [s["opus_warning"] for s in seats if s.get("opus_warning")]
    return {
        "models": models.get("models") or [],
        "pricing_note": models.get("pricing_note") or "",
        "locked_model": None,
        "lock_note": "",
        "seats": seats,
        "warnings": warnings,
        "path": str(config_path()),
    }


def _clean_model(value: str) -> str:
    model = str(value or "").strip()
    if "://" in model or model.startswith("//") or "\\" in model or ".." in model:
        raise AgentConfigError(f"Model id cannot be a URL or path: {model!r}")
    if not _MODEL_RE.fullmatch(model):
        raise AgentConfigError(f"Invalid model id: {model!r}")
    return model


def _clean_effort(value) -> str | None:
    if value is None or value == "" or value == "default":
        return None
    effort = str(value).strip().lower()
    if effort not in EFFORTS:
        raise AgentConfigError(f"Reasoning effort must be low, medium, or high (got {value!r})")
    return effort


def _clean_temperature(value, allowed: bool):
    if value is None or value == "":
        return None
    if not allowed:
        raise AgentConfigError("This seat does not send temperature")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise AgentConfigError(f"Temperature must be a number (got {value!r})") from exc
    if not 0 <= number <= 2:
        raise AgentConfigError("Temperature must be from 0 to 2")
    return number


def _clean_max_tokens(value, allowed: bool):
    if value is None or value == "":
        return None
    if not allowed:
        raise AgentConfigError("This seat does not send max tokens")
    if isinstance(value, bool):
        raise AgentConfigError("max_tokens must be an integer")
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise AgentConfigError(f"max_tokens must be an integer (got {value!r})") from exc
    if not 1 <= number <= 200000:
        raise AgentConfigError("max_tokens must be from 1 to 200000")
    return number


def _clean_system(value: str) -> str:
    if not isinstance(value, str):
        raise AgentConfigError("System prompt must be text")
    if "\x00" in value:
        raise AgentConfigError("System prompt contains a null byte")
    text = value.replace("\r\n", "\n").replace("\r", "\n")
    if len(text) > _MAX_PROMPT:
        raise AgentConfigError(f"System prompt is longer than {_MAX_PROMPT} characters")
    return text


def validate_and_save(incoming: dict) -> dict:
    """Persist overrides. Every seat, including conservative, keeps the model it is given."""
    if not isinstance(incoming, dict):
        raise AgentConfigError("Body must be an object")
    seats_in = incoming.get("seats")
    if not isinstance(seats_in, dict):
        raise AgentConfigError("seats must be an object")
    known = catalog()
    extra = [k for k in seats_in if k not in known]
    if extra:
        raise AgentConfigError("Unknown seat: " + ", ".join(sorted(extra)))
    overrides: dict[str, dict] = {}
    for seat_id, base in known.items():
        raw = seats_in.get(seat_id)
        if raw is None:
            continue
        if not isinstance(raw, dict):
            raise AgentConfigError(f"{seat_id} must be an object")
        model = _clean_model(raw.get("model") or base["model"])
        effort = _clean_effort(raw.get("effort"))
        temperature = _clean_temperature(raw.get("temperature"), base["supports_temperature"])
        max_tokens = _clean_max_tokens(raw.get("max_tokens"), base["supports_max_tokens"])
        # An empty number means the seat default. An empty effort is explicit:
        # the runner does not send reasoning.effort.
        if temperature is None:
            temperature = base["temperature"]
        if max_tokens is None:
            max_tokens = base["max_tokens"]
        system = _clean_system(raw.get("system") if raw.get("system") is not None else base["system"])
        diff = {}
        if model != base["model"]:
            diff["model"] = model
        if effort != base["effort"]:
            diff["effort"] = effort
        if temperature != base["temperature"]:
            diff["temperature"] = temperature
        if max_tokens != base["max_tokens"]:
            diff["max_tokens"] = max_tokens
        if system != base["system"]:
            diff["system"] = system
        if diff:
            overrides[seat_id] = diff
    payload = {"seats": overrides}
    path = config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)
    return public_payload()


def apply_system_prompt(seat_id: str, text: str) -> str:
    """Swap a customized role into a prompt. An unchanged role is left alone."""
    seat = resolved_seat(seat_id)
    custom = str(seat.get("system") or "").strip()
    default = str(seat.get("default_system") or "").strip()
    anchor = str(seat.get("anchor") or default).strip()
    if not isinstance(text, str) or not custom or custom == default:
        return text
    if anchor and anchor in text:
        return text.replace(anchor, custom, 1)
    return custom + "\n\n" + text


def apply_messages(seat_id: str, payload):
    """Rewrite a string prompt or a chat-message list for one seat."""
    if isinstance(payload, str):
        return apply_system_prompt(seat_id, payload)
    if isinstance(payload, list):
        out = []
        for message in payload:
            if isinstance(message, str):
                out.append(apply_system_prompt(seat_id, message))
                continue
            if isinstance(message, tuple) and len(message) == 2 and isinstance(message[1], str):
                role, content = message
                if role in ("system", "human", "user"):
                    out.append((role, apply_system_prompt(seat_id, content)))
                else:
                    out.append(message)
                continue
            content = getattr(message, "content", None)
            if isinstance(content, str):
                try:
                    message.content = apply_system_prompt(seat_id, content)
                except Exception:
                    pass
            elif isinstance(message, dict) and isinstance(message.get("content"), str):
                message = dict(message)
                message["content"] = apply_system_prompt(seat_id, message["content"])
            out.append(message)
        return out
    return payload


def dry_run_text() -> str:
    """What the next run would load. Does not construct a client or call a model."""
    lines = ["agent_config dry-run (no LLM calls)", f"file: {config_path()}"]
    for seat in resolved_seats():
        cost = seat.get("est_cost_usd")
        cost_s = f"${cost:.4f}" if isinstance(cost, float) else "omitted"
        lines.append(
            f"{seat['id']}: model={seat['model']} effort={seat['effort']} "
            f"temperature={seat['temperature']} max_tokens={seat['max_tokens']} "
            f"prompt_chars={len(seat['system'] or '')} est_cost={cost_s}"
        )
        if seat.get("opus_warning"):
            lines.append("  warning: " + seat["opus_warning"])
    lines.append("fund_manager.py and theme_scout.py read resolved_seat() at start.")
    lines.append("run_native.py calls install_native_seats() before the CLI. Seats: "
                 + ", ".join(s["id"] for s in resolved_seats() if s["pipeline"] == "native"))
    return "\n".join(lines)


# --------------------------------------------------------------------------- native hook
_NATIVE_FACTORIES = {
    "market": ("tradingagents.graph.setup", "create_market_analyst"),
    "sentiment": ("tradingagents.graph.setup", "create_sentiment_analyst"),
    "news": ("tradingagents.graph.setup", "create_news_analyst"),
    "fundamentals": ("tradingagents.graph.setup", "create_fundamentals_analyst"),
    "bull": ("tradingagents.graph.setup", "create_bull_researcher"),
    "bear": ("tradingagents.graph.setup", "create_bear_researcher"),
    "research_manager": ("tradingagents.graph.setup", "create_research_manager"),
    "trader": ("tradingagents.graph.setup", "create_trader"),
    "risk_aggressive": ("tradingagents.graph.setup", "create_aggressive_debator"),
    "risk_conservative": ("tradingagents.graph.setup", "create_conservative_debator"),
    "risk_neutral": ("tradingagents.graph.setup", "create_neutral_debator"),
    "portfolio_manager": ("tradingagents.graph.setup", "create_portfolio_manager"),
}


class _SeatCall:
    """Delegates to a seat client and rewrites the role text on each call."""

    def __init__(self, seat_id: str, inner):
        self.seat_id = seat_id
        self.inner = inner

    def invoke(self, payload, *args, **kwargs):
        return self.inner.invoke(apply_messages(self.seat_id, payload), *args, **kwargs)

    def bind_tools(self, tools, **kwargs):
        return _SeatCall(self.seat_id, self.inner.bind_tools(tools, **kwargs))

    def bind(self, **kwargs):
        return _SeatCall(self.seat_id, self.inner.bind(**kwargs))

    def __getattr__(self, name):
        return getattr(self.inner, name)


def _fallback_model_name(llm) -> str:
    for attr in ("model_name", "model"):
        value = getattr(llm, attr, None)
        if isinstance(value, str) and value:
            return value
    return ""


def _needs_own_client(seat: dict, fallback_name: str) -> bool:
    if seat.get("model") and seat["model"] != fallback_name:
        return True
    if seat.get("max_tokens") is not None or seat.get("temperature") is not None:
        return True
    if seat.get("effort"):
        return True
    return False


def _build_client(seat: dict, config: dict):
    from tradingagents.llm_clients import create_llm_client

    kwargs = {}
    if seat.get("max_tokens") is not None:
        kwargs["max_tokens"] = int(seat["max_tokens"])
    if seat.get("temperature") is not None:
        kwargs["temperature"] = float(seat["temperature"])
    if seat.get("effort"):
        kwargs["reasoning_effort"] = seat["effort"]
    base = (config or {}).get("backend_url") or "https://openrouter.ai/api/v1"
    client = create_llm_client("openrouter", seat["model"], base_url=base, **kwargs)
    llm = client.get_llm()
    if seat.get("effort"):
        try:
            llm = llm.bind(extra_body={"reasoning": {"effort": seat["effort"]}})
        except Exception:
            pass
    return llm


def _wrap_factory(seat_id: str, original, config: dict):
    def wrapped(llm, _orig=original, _seat=seat_id, _config=config):
        seat = resolved_seat(_seat)
        name = _fallback_model_name(llm)
        try:
            inner = _build_client(seat, _config) if _needs_own_client(seat, name) else llm
        except Exception as exc:  # noqa: BLE001
            print(f"[agent_config] {_seat}: keeping the tier client ({exc})", flush=True)
            inner = llm
        return _orig(_SeatCall(_seat, inner))

    return wrapped


def install_native_seats(config: dict | None = None) -> list[str]:
    """Patch the graph factories. Importing this does not call a model."""
    import importlib

    config = config or {}
    installed = []
    for seat_id, (module_name, fn_name) in _NATIVE_FACTORIES.items():
        module = importlib.import_module(module_name)
        original = getattr(module, fn_name)
        if getattr(original, "_argus_seat", None) == seat_id:
            installed.append(seat_id)
            continue
        wrapped = _wrap_factory(seat_id, original, config)
        wrapped._argus_seat = seat_id  # type: ignore[attr-defined]
        setattr(module, fn_name, wrapped)
        installed.append(seat_id)
    from tradingagents.memory.reflection import Reflector

    if not getattr(Reflector.__init__, "_argus_seat", None):
        orig_init = Reflector.__init__

        def _init(self, llm, _orig=orig_init, _config=config):
            seat = resolved_seat("memory")
            name = _fallback_model_name(llm)
            try:
                inner = _build_client(seat, _config) if _needs_own_client(seat, name) else llm
            except Exception as exc:  # noqa: BLE001
                print(f"[agent_config] memory: keeping the tier client ({exc})", flush=True)
                inner = llm
            _orig(self, _SeatCall("memory", inner))

        _init._argus_seat = "memory"  # type: ignore[attr-defined]
        Reflector.__init__ = _init  # type: ignore[method-assign]
        installed.append("memory")
    print("[agent_config] native seats: " + ", ".join(installed), flush=True)
    return installed


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv not in ([], ["--dry-run"]):
        print("usage: agent_config.py [--dry-run]", file=sys.stderr)
        return 2
    print(dry_run_text())
    # Prove the runners import the same resolver. Import is local so a missing
    # optional dependency cannot break Argus, which only needs this module.
    for mod_name, fn_name in (
        ("fund_manager.fund_manager", "agent_plan_from_config"),
        ("theme_scout.theme_scout", "agent_plan_from_config"),
    ):
        try:
            if mod_name.startswith("fund_manager"):
                sys.path.insert(0, str(addons_dir() / "fund_manager"))
                import fund_manager as fm  # type: ignore
                plan = fm.agent_plan_from_config()
            else:
                sys.path.insert(0, str(addons_dir() / "theme_scout"))
                import theme_scout as ts  # type: ignore
                plan = ts.agent_plan_from_config()
        except Exception as exc:  # noqa: BLE001
            print(f"{mod_name}: not imported ({exc})")
            continue
        bits = ", ".join(f"{k}={v.get('model')}" for k, v in plan.items())
        print(f"{mod_name} loaded: {bits}")
    fm_model = resolved_seat("fm_conservative")["model"]
    risk_model = resolved_seat("risk_conservative")["model"]
    print(f"conservative models: fund={fm_model} per-ticker={risk_model}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
