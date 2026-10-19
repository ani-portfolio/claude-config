#!/usr/bin/env python3
"""Unit checks for ml-autopilot's next action, checkpoints, backlog, and limits (ml-autopilot/scripts/autopilot.py).

Plain asserts, no test framework and no project packages needed:
    python3 tests/test_autopilot.py
Exit code 0 when every check passed, 1 otherwise.
"""

from __future__ import annotations

import os
import sys
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "ml-autopilot", "scripts"))

import autopilot as a  # noqa: E402
import ml_common as m  # noqa: E402


def _spec(**business):
    s = m.template("t")
    s["project"]["kernel"] = "k"
    s["business"].update({"mode": "standard", "business_goal": "g", "decision_supported": "d", "status": "draft",
                          **business})
    s["autopilot"] = {"settings": dict(m.AUTOPILOT_DEFAULTS)}
    return s


def _modelled(s):
    """A spec where every step up to training is done, with a confirmed business and ML framing."""
    s["business"].update(status="confirmed", outcome="o")
    s["discovery"] = {"sources": ["x"]}
    s["framing"]["status"] = "confirmed"
    s["eda"] = {"mode": "targeted", "target": "y"}
    for k in ("split", "features", "baseline", "training"):
        s[k] = {"done": True}
    return s


def _evaluated(s):
    s = _modelled(s)
    s["evaluation"] = {"test_opened": [{"at": "2026-10-01T10:00:00"}]}
    return s


def test_no_settings_asks_the_goal():
    s = _spec()
    s["autopilot"] = {}
    assert a.next_action(s)["action"] == "ask_goal"


def test_no_anchors_asks_the_goal():
    s = _spec(business_goal=None)
    assert a.next_action(s)["action"] == "ask_goal"


def test_discovery_runs_unattended():
    act = a.next_action(_spec())
    assert act["action"] == "run_stage" and act["stage"] == "discovery" and not act["attended"], act


def test_drafted_business_stops_at_checkpoint():
    s = _spec(outcome="o")
    s["discovery"] = {"sources": ["x"]}
    act = a.next_action(s)
    assert act["action"] == "checkpoint" and act["checkpoint"] == "business", act


def test_undrafted_business_is_drafted():
    s = _spec()
    s["discovery"] = {"sources": ["x"]}
    act = a.next_action(s)
    assert act["action"] == "run_stage" and act["stage"] == "business" and act["mode"] == "draft", act


def test_open_checkpoint_means_review():
    s = _spec()
    s["autopilot"]["checkpoints"] = [{"kind": "business", "status": "open", "file": "docs/checkpoint.md"}]
    assert a.next_action(s)["action"] == "review"


def test_training_done_stops_before_the_test_set():
    act = a.next_action(_modelled(_spec()))
    assert act["action"] == "checkpoint" and act["checkpoint"] == "test_v1", act


def test_cleared_test_checkpoint_runs_evaluation():
    s = _modelled(_spec())
    s["autopilot"]["checkpoints"] = [{"kind": "test_v1", "status": "cleared"}]
    act = a.next_action(s)
    assert act["action"] == "run_stage" and act["stage"] == "evaluation" and act["attended"], act


def test_evaluated_opens_a_round():
    assert a.next_action(_evaluated(_spec()))["action"] == "open_round"


def test_round_waits_for_experiments_then_closes():
    s = _evaluated(_spec())
    s["iterations"]["current"] = 1
    s["iterations"]["entries"] = [{"id": 1, "status": "open", "experiments": [{"id": "a", "status": "running"},
                                                                              {"id": "b", "status": "passed"}]}]
    assert a.next_action(s)["action"] == "await_experiments"
    s["iterations"]["entries"][0]["experiments"][0]["status"] = "error"
    assert a.next_action(s)["action"] == "close_round"


def test_manual_iteration_is_left_to_ml_iterate():
    s = _evaluated(_spec())
    s["iterations"]["current"] = 1
    s["iterations"]["entries"] = [{"id": 1, "status": "open", "changes": []}]
    assert a.next_action(s)["action"] == "manual_iteration"


def _rounds(s, outcomes):
    s["iterations"]["entries"] = [{"id": i + 1, "status": "closed", "outcome": o, "closed_at": f"2026-10-0{i + 2}T00:00:00",
                                   "experiments": [{"id": "a"}]} for i, o in enumerate(outcomes)]
    return s


def test_limit_on_rounds():
    s = _rounds(_evaluated(_spec()), ["kept", "kept", "kept"])
    lim = a.limits(s)
    assert not lim["another_round"] and "3 of 3" in lim["reason"], lim


def test_limit_on_failures_in_a_row():
    s = _rounds(_evaluated(_spec()), ["rolled_back", "rolled_back"])
    lim = a.limits(s)
    assert not lim["another_round"] and lim["fails_in_a_row"] == 2, lim
    s = _rounds(_evaluated(_spec()), ["rolled_back", "kept"])
    assert a.limits(s)["another_round"]


def test_max_rounds_setting_never_exceeds_the_bar():
    s = _evaluated(_spec())
    s["autopilot"]["settings"]["max_rounds"] = 10
    assert a.limits(s)["max_rounds"] == m.ITERATION_BAR_DEFAULTS["max_iterations"]


def test_kept_round_needs_a_final_test_look():
    s = _rounds(_evaluated(_spec()), ["kept", "rolled_back", "rolled_back"])
    act = a.next_action(s)
    assert act["action"] == "checkpoint" and act["checkpoint"] == "test_final", act
    s["autopilot"]["checkpoints"] = [{"kind": "test_final", "status": "cleared", "cleared_at": "2026-10-09T00:00:00"}]
    assert a.next_action(s)["action"] == "run_stage"
    s["evaluation"]["test_opened"].append({"at": "2026-10-09T01:00:00"})
    assert a.next_action(s)["action"] == "done"


def test_nothing_kept_is_done_without_a_test_look():
    s = _rounds(_evaluated(_spec()), ["rolled_back", "rolled_back"])
    assert a.next_action(s)["action"] == "done"


def test_stop_ends_the_rounds():
    s = _evaluated(_spec())
    s["autopilot"]["stopped"] = {"reason": "nothing worth trying"}
    assert a.next_action(s)["action"] == "done"


def test_pick_prefers_cheap_and_different_stages():
    s = _spec()
    s["autopilot"]["backlog"] = [
        {"id": "B1", "stage": "features", "change": "x", "cost": 2, "status": "open"},
        {"id": "B2", "stage": "features", "change": "y", "cost": 1, "status": "open"},
        {"id": "B3", "stage": "split", "change": "z", "cost": 3, "status": "open"},
        {"id": "B4", "stage": "training", "change": "w", "cost": 1, "status": "tried"}]
    assert [b["id"] for b in a._pick(s, 2)] == ["B2", "B3"]
    assert [b["id"] for b in a._pick(s, 3)] == ["B2", "B3", "B1"]


def test_checkpoint_kinds_validate():
    s = _spec()
    s["autopilot"]["checkpoints"] = [{"kind": "nope", "status": "open"}]
    assert any("autopilot.checkpoints" in e for e in m.validate(s)[0])


def test_confirmed_while_checkpoint_open_is_flagged():
    s = _spec(status="confirmed", outcome="o")
    s["autopilot"]["checkpoints"] = [{"kind": "business", "status": "open"}]
    assert any(c["check"] == "checkpoint" for c in m.consistency_checks(s))


def test_autopilot_mode():
    assert m.autopilot_mode("autopilot") and m.autopilot_mode("--json autopilot")
    assert not m.autopilot_mode("") and not m.autopilot_mode("change the folds")


def main():
    tests = [(k, v) for k, v in globals().items() if k.startswith("test_") and callable(v)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"ok    {name}")
        except Exception:
            failed += 1
            print(f"FAIL  {name}")
            traceback.print_exc()
    print(f"\n{len(tests) - failed} of {len(tests)} passed.")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
