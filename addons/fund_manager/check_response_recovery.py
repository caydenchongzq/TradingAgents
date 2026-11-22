"""Offline checks for durable model responses and safe recovery. No API calls."""
from __future__ import annotations
import json
import sys
import tempfile
import types
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fund_manager as fm
from reprocess_response import reprocess

SCHEMA = {"type": "object", "required": ["name", "items"], "additionalProperties": False,
          "properties": {"name": {"type": "string"}, "items": {"type": "array", "items": {"type": "object", "required": ["value"], "properties": {"value": {"type": ["number", "null"]}}}}}}
assert fm.parse_structured_response('```json\n{"name":"ok","items":[{"value":2}]}\n```', SCHEMA)["name"] == "ok"
for bad in ('{"name":"x","items":[]} {}', '{"name":"x","items":[{"value":"wrong"}]}', '{"name":"truncated"', '{"outer": {"name":"x","items":[]}', '[{"name":"x","items":[]}]', '[{"broken": {"name":"x","items":[]}', '{"name":"x","items":[{"value":NaN}]}', '{"name":"x","items":[{"value":Infinity}]}', '{"name":"x","items":[{"value":1e999}]}'):
    try:
        fm.parse_structured_response(bad, SCHEMA)
        raise AssertionError("invalid response accepted")
    except ValueError:
        pass

class FakeUsage:
    def model_dump(self): return {"prompt_tokens": 5, "completion_tokens": 7, "cost": 0.01}
class FakeMessage:
    def __init__(self, content): self.content = content
class FakeChoice:
    def __init__(self, content, reason="stop"):
        self.message, self.finish_reason = FakeMessage(content), reason
class FakeResponse:
    def __init__(self, content, ident):
        self.choices, self.usage, self.model, self.id = [FakeChoice(content)], FakeUsage(), "test-model", ident

responses = [RuntimeError("provider rejected schema"), FakeResponse('{"name":"partial","items":[]} {"name":"second","items":[]}', "bad-json"), FakeResponse("# Draft prose", "text")]
created = []
class Completions:
    def create(self, **kwargs):
        item = responses[len(created)]
        created.append(kwargs)
        if isinstance(item, Exception): raise item
        return item
fake_openai = types.ModuleType("openai")
fake_openai.OpenAI = lambda **kwargs: types.SimpleNamespace(chat=types.SimpleNamespace(completions=Completions()))
cost_rows=[]
fake_cost = types.ModuleType("cost_ledger")
fake_cost.write_chat_cost = lambda *args, **kwargs: cost_rows.append(kwargs)
with tempfile.TemporaryDirectory() as tmp, patch.dict(sys.modules, {"openai": fake_openai, "cost_ledger": fake_cost}), patch.object(fm, "_addons_root", lambda: None), patch.object(fm.os, "getenv", lambda key, default=None: "fixture-key" if key == "OPENROUTER_API_KEY" else default):
    journal = Path(tmp) / "response_attempts.jsonl"
    result = fm.call_llm("test-model", "system", "user", SCHEMA, "test_schema", 100, None, "fm/draft", attempt_path=journal)
    rows = [json.loads(row) for row in journal.read_text().splitlines()]
    assert result["parsed"] is None and len(created) == 3
    assert len([row for row in rows if row["event"] == "response"]) == 2
    assert all(row["usage"]["completion_tokens"] == 7 for row in rows if row["event"] == "response")
    assert len(cost_rows) == 2, cost_rows
    report = Path(tmp) / "portfolio_report.json"
    report.write_text(json.dumps({"raw_text_fallbacks": ["# Draft prose"]}))
    before = report.read_bytes()
    recovered = reprocess(report, 1)
    saved = json.loads(recovered.read_text())
    assert saved["status"] == "unavailable" and "does not" not in saved.get("reason", "")
    assert report.read_bytes() == before
    assert len(cost_rows) == 2, "offline recovery must not replay ledger costs"
    for label in ("fm/draft", "fm/final"):
        validation_journal = Path(tmp) / f"{label.split('/')[-1]}.jsonl"
        fm._append_attempt(validation_journal, {"event":"response", "attempt":1,"attempt_id":"candidate-"+label,"label":label,"schema_name":"fund_manager_draft" if label.endswith("draft") else "fund_manager_final","content":"raw candidate", "mode":"json_schema"})
        fm._append_attempt(validation_journal, {"event":"parse_result", "attempt":1,"label":label,"status":"valid"})
        with patch.object(fm, "validate_trades", side_effect=ValueError("fixture trade rejection")):
            try:
                fm.validate_candidate({"trades": []}, {}, label, validation_journal, "candidate-"+label)
                raise AssertionError("trade validation failure accepted")
            except ValueError:
                pass
        outputs = fm._unvalidated_outputs(validation_journal)
        assert outputs and outputs[0]["content"] == "raw candidate" and "fixture trade rejection" in outputs[0]["reason"]
    validation_journal = Path(tmp) / "same_label.jsonl"
    fm._append_attempt(validation_journal, {"event":"response", "attempt":1, "attempt_id":"first-response",
                "label":"fm/draft", "schema_name":"fund_manager_draft", "content":"earlier accepted candidate"})
    fm._append_attempt(validation_journal, {"event":"parse_result", "attempt":1, "label":"fm/draft", "status":"valid"})
    fm._append_attempt(validation_journal, {"event":"trade_validation_result", "label":"fm/draft",
                "attempt_id":"first-response", "status":"valid"})
    fm._append_attempt(validation_journal, {"event":"response", "attempt":2, "attempt_id":"second-response",
                "label":"fm/draft", "schema_name":"fund_manager_draft", "content":"later rejected candidate"})
    fm._append_attempt(validation_journal, {"event":"parse_result", "attempt":2, "label":"fm/draft", "status":"valid"})
    fm._append_attempt(validation_journal, {"event":"trade_validation_result", "label":"fm/draft",
                "attempt_id":"second-response", "status":"invalid", "error":"second candidate rejected"})
    outputs = fm._unvalidated_outputs(validation_journal)
    assert len(outputs) == 1 and outputs[0]["content"] == "later rejected candidate"
    assert outputs[0]["reason"] == "second candidate rejected"
    blocker = Path(tmp) / "blocker"
    blocker.write_text("x")
    called=[]
    class OneCall:
        def create(self, **kwargs):
            called.append(1); return FakeResponse('{"name":"ok","items":[]}', "persist-fail")
    fake_openai.OpenAI = lambda **kwargs: types.SimpleNamespace(chat=types.SimpleNamespace(completions=OneCall()))
    try:
        fm.call_llm("test-model", "system", "user", SCHEMA, "test_schema", 100, None, "fm/draft", attempt_path=blocker / "journal")
        raise AssertionError("journal failure did not stop call")
    except OSError:
        assert len(called) == 1
    fake_cost.write_chat_cost = lambda *args, **kwargs: (_ for _ in ()).throw(OSError("ledger unavailable"))
    journal2 = Path(tmp) / "cost_failure.jsonl"
    try:
        fm.call_llm("test-model", "system", "user", SCHEMA, "test_schema", 100, None, "fm/draft", attempt_path=journal2)
        raise AssertionError("ledger failure did not stop downstream work")
    except RuntimeError as exc:
        assert "cost accounting failed" in str(exc)
    persisted = [json.loads(line) for line in journal2.read_text().splitlines()]
    assert any(row["event"] == "response" and row["content"] for row in persisted)
    assert any(row["event"] == "parse_result" and row["cost_error"] for row in persisted)

with tempfile.TemporaryDirectory() as tmp:
    import cost_ledger
    root = Path(tmp) / "data"
    folder = root / "fund_manager" / "2026-11-17" / "fixture_run"
    folder.mkdir(parents=True)
    report = folder / "portfolio_report.json"
    report.write_text(json.dumps({"date":"2026-11-17", "raw_text_fallbacks": []}))
    journal = folder / "response_attempts.jsonl"
    response = {"event":"response", "attempt":2, "attempt_id":"stable-attempt-id", "label":"fm/draft",
                "schema_name":"unknown_test_schema", "mode":"json_object", "model_returned":"test-model",
                "response_id":"fixture-response", "created_utc":"2026-11-17T00:00:00Z",
                "usage":{"prompt_tokens":50,"completion_tokens":30}, "content":"raw malformed response"}
    fm._append_attempt(journal, response)
    original_journal_hash = __import__("hashlib").sha256(journal.read_bytes()).hexdigest()
    recovery1 = reprocess(report, 1)
    reprocess(report, 1)
    saved = json.loads(recovery1.read_text())
    assert saved["response_journal_sha256"] == original_journal_hash
    assert saved["selected_response_sha256"] == __import__("hashlib").sha256(response["content"].encode()).hexdigest()
    costs = list((root / "fund_manager" / "run_costs").glob("attempt_*.json"))
    assert len(costs) == 1, "repeat recovery must reconcile known usage idempotently"
    assert report.read_text().find("raw_text_fallbacks") >= 0
    assert len(journal.read_text().splitlines()) == 1, "reprocessing must not rewrite the source journal"
    orphan_dir = root / "fund_manager" / "2026-11-17" / "orphan"
    orphan_dir.mkdir(parents=True)
    orphan_journal = orphan_dir / "response_attempts.jsonl"
    candidate = {"headline":"Keep", "diagnosis":"No change", "cash_decision":{
        "target_cash_pct":10, "regime":"neutral", "regime_signals":[], "rationale":"Fixture",
        "deploy_conditions":[], "raise_conditions":[]}, "key_risks":[], "correlation_clusters":[],
        "decision_summary":"Hold", "trades":[], "swaps_considered":[], "priority_order":[],
        "cash_plan":{"summary":"Hold", "deployments":[]}, "levels":[],
        "what_would_change_my_mind":[], "data_gaps":[], "confidence":"low"}
    fm._append_attempt(orphan_journal, {"event":"response", "attempt":1, "attempt_id":"orphan-attempt",
        "label":"fm/draft", "schema_name":"fund_manager_draft", "mode":"json_object",
        "created_utc":"2026-11-17T00:00:00Z", "usage":{"prompt_tokens":1,"completion_tokens":1},
        "model_returned":"test-model", "content":json.dumps(candidate)})
    orphan_hash = __import__("hashlib").sha256(orphan_journal.read_bytes()).hexdigest()
    recovered_orphan = json.loads(reprocess(orphan_journal, 1).read_text())
    assert recovered_orphan["source_artifact_type"] == "response_journal"
    assert recovered_orphan["source_sha256"] == orphan_hash
    assert recovered_orphan["status"] == "schema_valid"
    assert recovered_orphan["trade_validation_status"] == "not_run_missing_report_stats"
    v2_candidate = {**candidate, "decision_contract_version": 2,
                    "candidate_coverage": [{"ticker": "NEW", "disposition": "watch", "rationale": "Wait for evidence.",
                                            "research_status": "missing", "source_profile": "missing",
                                            "evidence_date": None}]}
    v2_dir = orphan_dir / "v2"
    v2_dir.mkdir()
    v2_journal = v2_dir / "response_attempts.jsonl"
    fm._append_attempt(v2_journal, {"event":"response", "attempt":1, "attempt_id":"v2-attempt",
        "label":"fm/draft", "schema_name":"fund_manager_draft_v2", "mode":"json_object",
        "created_utc":"2026-11-17T00:00:00Z", "usage":{"prompt_tokens":1,"completion_tokens":1},
        "model_returned":"test-model", "content":json.dumps(v2_candidate)})
    recovered_v2 = json.loads(reprocess(v2_journal, 1).read_text())
    assert recovered_v2["status"] == "schema_valid" and recovered_v2["schema_name"] == "fund_manager_draft_v2", recovered_v2
print("check_response_recovery: ok")
