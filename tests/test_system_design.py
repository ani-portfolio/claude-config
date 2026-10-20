"""Unit checks for ml-system-design: the entry rule, every block and warning, the build, and the document.

Run: python3 tests/test_system_design.py
"""

import copy
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "ml-system-design", "scripts"))

import design as d  # noqa: E402
import ml_common as m  # noqa: E402
import system_doc  # noqa: E402

EXAMPLE = m.load(os.path.join(HERE, "..", "ml-system-design", "templates", "system_answers.example.yaml"))
INPUTS = ["temp_max_lag_1", "temp_max_roll_7", "day_of_year"]


def _spec(mode="standard", decision="Go: ship V1."):
    s = m.template("weather")
    s["project"]["kernel"] = "k"
    s["business"].update({"mode": mode, "status": "confirmed", "business_goal": "g", "decision_supported": "d",
                          "learning_question": "q", "outcome": "o",
                          "answer_actions": [{"answer": "Matches persistence.", "action": "Build the system and serve persistence."}]})
    s["framing"].update({"status": "confirmed", "task_type": "forecasting"})
    s["framing"]["success"]["primary_metric"] = "MAE"
    s["framing"]["prediction_time"]["cadence"] = "daily batch"
    s["features"] = {"inputs": [{"name": n} for n in INPUTS]}
    s["training"] = {"champion": {"name": "lgbm", "std": 0.1}}
    s["evaluation"] = {"champion": "lgbm", "run_at": "2026-10-01T10:00:00"}
    s["notebooks"] = {"evaluation": {"run_at": "2026-10-01T10:00:00"}}
    if decision:
        s["decisions"] = [{"date": "2026-10-01", "stage": "evaluation", "decision": decision}]
    return s


def _answers(**changes):
    a = copy.deepcopy(EXAMPLE)
    for path, value in changes.items():
        node = a
        keys = path.split(".")
        for k in keys[:-1]:
            node = node.setdefault(k, {})
        node[keys[-1]] = value
    return a


def _ids(findings):
    return {f["id"] for f in findings}


# ---------------------------------------------------------------- entry rule

def test_entry_needs_an_evaluation_and_a_decision():
    s = _spec(decision=None)
    assert d.entry_state(s)["rule"] == "refuse"
    s["evaluation"] = {}
    assert "has not run" in d.entry_state(s)["detail"]


def test_entry_go_and_unclear():
    assert d.entry_state(_spec())["rule"] == "go"
    assert d.entry_state(_spec(decision="Looks fine to me."))["rule"] == "unclear"


def test_entry_no_go_refuses_standard_and_defers_exploratory():
    assert d.entry_state(_spec(decision="No go: worse than persistence."))["rule"] == "refuse"
    st = d.entry_state(_spec("exploratory", "No go: only matches persistence."))
    assert st["rule"] == "exploratory_no_go" and st["answer_actions"]


def test_entry_outcome_must_match_the_rule():
    s = _spec(decision="No go: worse.")
    assert "entry" in _ids(d.check(s, _answers())[0])
    assert "entry" not in _ids(d.check(s, _answers(**{"entry.outcome": "override", "entry.reason": "Learning goal."}))[0])
    assert "entry" in _ids(d.check(s, _answers(**{"entry.outcome": "override"}))[0])
    ex = _spec("exploratory", "No go: only matches persistence.")
    assert "entry" in _ids(d.check(ex, _answers(**{"entry.outcome": "no_go_build", "entry.answer": "Something else."}))[0])
    assert "entry" not in _ids(d.check(ex, _answers(**{"entry.outcome": "no_go_build", "entry.answer": "Matches persistence."}))[0])


# ---------------------------------------------------------------- blocks and warnings

def test_example_answers_pass():
    blocks, warns = d.check(_spec(), _answers())
    assert not blocks, blocks
    assert _ids(warns) == {"online for a batch cadence"} and all(w["accepted"] for w in warns), warns


def test_gate_metric_must_be_the_primary_metric():
    assert "gate metric" in _ids(d.check(_spec(), _answers(**{"components.registry.settings.gate.metric": "RMSE"}))[0])


def test_gate_metric_falls_back_to_the_baseline_metric():
    s = _spec()
    s["framing"]["success"]["primary_metric"] = None
    s["baseline"] = {"metrics": {"primary": {"key": "wape"}}}
    assert "gate metric" in _ids(d.check(s, _answers())[0])
    assert "gate metric" not in _ids(d.check(s, _answers(**{"components.registry.settings.gate.metric": "WAPE"}))[0])


def test_gate_margin_below_cv_spread_is_blocked():
    assert "gate margin" in _ids(d.check(_spec(), _answers(**{"components.registry.settings.gate.margin": 0.05}))[0])
    assert "gate margin" in _ids(d.check(_spec(), _answers(**{"components.registry.settings.gate.margin": "small"}))[0])


def test_every_input_must_be_served():
    blocks = d.check(_spec(), _answers(**{"components.features.settings.served_inputs": INPUTS[:2]}))[0]
    assert any(b["id"] == "served inputs" and "day_of_year" in b["detail"] for b in blocks), blocks


def test_batch_faster_than_the_data_is_blocked():
    a = _answers(**{"components.inference.settings": {"mode": "batch", "schedule": "hourly"}})
    assert "schedule" in _ids(d.check(_spec(), a)[0])
    a = _answers(**{"components.inference.settings": {"mode": "batch", "schedule": "daily"}})
    blocks, warns = d.check(_spec(), a)
    assert "schedule" not in _ids(blocks) and "online for a batch cadence" not in _ids(warns)


def test_platform_registry_needs_a_decision():
    a = _answers(**{"components.registry.settings.decides_champion": None})
    assert "registry" in _ids(d.check(_spec(), a)[0])
    a = _answers(**{"platform": {"kind": "default"}, "components.registry.settings.decides_champion": None})
    assert "registry" not in _ids(d.check(_spec(), a)[0])


def test_not_needed_needs_a_reason_and_confirmed_needs_all():
    a = _answers(**{"components.data_validation": {"needed": False}})
    assert "not needed" in _ids(d.check(_spec(), a)[0])
    a = _answers()
    del a["components"]["testing"]
    assert "components" in _ids(d.check(_spec(), a)[0])
    a["status"] = "draft"
    assert "components" not in _ids(d.check(_spec(), a)[0])


def test_judgement_calls_warn():
    a = _answers(**{"components.failure_handling": {"needed": False, "reason": "r"},
                    "components.label_feedback": {"needed": False, "reason": "r"},
                    "components.testing": {"needed": False, "reason": "r"},
                    "components.registry.settings.gate.promotion": "auto"})
    warns = d.check(_spec(), a)[1]
    assert {"no fallback", "monitoring without labels", "no tests", "auto promotion"} <= _ids(warns), warns
    assert not any(w["accepted"] for w in warns if w["id"] != "online for a batch cadence")


# ---------------------------------------------------------------- build and document

def _project(spec):
    root = tempfile.mkdtemp()
    path = os.path.join(root, m.SPEC_NAME)
    m.save(path, spec)
    return root, path


def test_build_writes_section_and_document():
    root, path = _project(_spec())
    r = d.build(path, _answers())
    assert r["ok"], r
    s = m.load(path)
    assert s["system"]["champion"] == "lgbm" and s["system"]["doc_hash"] and s["system"]["status"] == "confirmed"
    assert any(x.get("stage") == "system" and "online for a batch cadence" in x["decision"] for x in s["decisions"])
    text = open(os.path.join(root, s["system"]["doc"]), encoding="utf-8").read()
    assert "```mermaid" in text and "flowchart LR" in text
    for title in system_doc.TITLES.values():
        assert f"| {title}" in text, title
    assert m.consistency_checks(s) == [] or all(c["check"] != "system champion" for c in m.consistency_checks(s))


def test_build_refuses_blocks_and_unaccepted_warnings():
    root, path = _project(_spec())
    assert not d.build(path, _answers(**{"components.registry.settings.gate.metric": "RMSE"}))["ok"]
    assert not d.build(path, _answers(warnings_accepted=[]))["ok"]
    assert not m.load(path).get("system")


def test_hand_edits_are_noticed():
    root, path = _project(_spec())
    d.build(path, _answers())
    doc = os.path.join(root, m.load(path)["system"]["doc"])
    with open(doc, "a", encoding="utf-8") as f:
        f.write("\nA note by hand.\n")
    assert d.build(path, _answers()).get("edited")
    assert d.build(path, _answers(), overwrite=True)["ok"]


def test_a_new_champion_makes_the_design_out_of_date():
    root, path = _project(_spec())
    d.build(path, _answers())
    s = m.load(path)
    s["training"]["champion"]["name"] = "ridge"
    assert any(c["check"] == "system champion" for c in m.consistency_checks(s))
    assert d.stale_parts(s) == d.CHAMPION_DEPENDENT


def test_not_needed_components_are_dashed_in_the_diagram():
    s = _spec()
    s["system"] = {"components": {"testing": {"needed": False, "reason": "r"}, "serving": {"needed": True, "tool": 'A "q"'}}}
    text = system_doc.render(s)
    assert "class " in text and "testing" in text.split("class ")[-1]
    assert 'A \'q\'' in text


def main():
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"ok    {name}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FAIL  {name}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed} of {len(tests)} passed.")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
