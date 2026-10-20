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


def test_round_launches_waits_then_advances():
    s = _evaluated(_spec())
    s["autopilot"]["settings"]["max_parallel"] = 2
    s["iterations"]["current"] = 1
    s["iterations"]["entries"] = [{"id": 1, "status": "open", "experiments": [
        {"id": "a", "status": "running"}, {"id": "b", "status": "planned"}, {"id": "c", "status": "planned"}]}]
    act = a.next_action(s)
    assert act["action"] == "launch_experiments" and act["experiments"] == ["b"], act
    s["iterations"]["entries"][0]["experiments"][1]["status"] = "running"
    assert a.next_action(s)["action"] == "await_experiments"
    for x in s["iterations"]["entries"][0]["experiments"]:
        x["status"] = "passed"
    assert a.next_action(s)["action"] == "advance_round"
    s["autopilot"]["rounds"] = [{"iteration": 1, "decided": True, "log": ["settled"]}]
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


def test_a_round_that_adopts_nothing_stops_when_the_backlog_is_empty():
    s = _rounds(_evaluated(_spec()), ["rolled_back"])
    lim = a.limits(s)
    assert not lim["another_round"] and "adopted nothing" in lim["reason"], lim
    s["autopilot"]["backlog"] = [{"id": "B1", "stage": "features", "change": "x", "status": "open"}]
    assert a.limits(s)["another_round"]   # an item upstream of the split had a round of its own: the rest wait
    assert a.limits(_rounds(_evaluated(_spec()), ["kept"]))["another_round"]


def test_old_settings_are_brought_up_to_date():
    s = _spec()
    s["autopilot"]["settings"] = {"experiments_per_round": 4, "max_rounds": 9, "unattended": True}
    assert m.autopilot_settings(s) == {"max_parallel": 4, "unattended": True}
    assert m.migrate(s)["autopilot"]["settings"] == {"max_parallel": 4, "unattended": True}


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


def test_round_items_take_the_whole_backlog_or_one_upstream_item():
    s = _spec()
    s["autopilot"]["backlog"] = [
        {"id": "B1", "stage": "features", "change": "x", "cost": 2, "status": "open"},
        {"id": "B2", "stage": "features", "change": "y", "cost": 1, "status": "open"},
        {"id": "B3", "stage": "training", "change": "z", "cost": 3, "status": "open"},
        {"id": "B4", "stage": "training", "change": "w", "cost": 1, "status": "tried"}]
    assert [b["id"] for b in a._round_items(s)] == ["B2", "B1", "B3"]
    s["autopilot"]["backlog"] += [{"id": "B5", "stage": "split", "change": "v", "cost": 2, "status": "open"},
                                  {"id": "B6", "stage": "split", "change": "u", "cost": 1, "status": "open"}]
    assert [b["id"] for b in a._round_items(s)] == ["B6"]


# ---------------------------------------------------------------- the phases of a round

GOOD, BASE = [0.80, 0.81, 0.82, 0.83], [0.70, 0.71, 0.72, 0.73]


def _x(id_, status="passed", folds=None, phase="screen", stage="features", gain=None, carried=None):
    """An experiment record as record-experiment leaves it: its bar against the carried champion, with per fold
    scores of the model it is judged by and of the carried one."""
    folds = folds or GOOD
    carried = carried or BASE
    d = sum(f - c for f, c in zip(folds, carried)) / len(folds)
    return {"id": id_, "experiment": f"V2-{id_}", "status": status, "phase": phase, "hypothesis": f"h {id_}",
            "changes": [{"stage": stage, "change": f"change {id_}"}], "backlog": f"B{id_}", "branch": f"exp/v2-{id_}",
            "bar": {"status": "pass" if status == "passed" else "fail", "new": "m", "carried": "old",
                    "mean_diff": gain if gain is not None else d, "per_fold": {"m": folds, "old": carried},
                    "champion_model": {"name": "m", "type": "lightgbm", "params": {"n": 1}, "tuning": "none"}}}


def _round(*xs):
    s = _evaluated(_spec())
    s["training"] = {"primary_label": "auc"}
    entry = {"id": 1, "status": "open", "version": "V2", "experiments": list(xs)}
    st = {"iteration": 1, "phase": "screen", "status": "open", "log": []}
    return s, entry, st


def _run(s, entry, st):
    plans, problems = a._advance(s, entry, st)
    while not plans and not problems and not st.get("decided"):
        plans, problems = a._advance(s, entry, st)
    assert not problems, problems
    return plans


def _report(entry, plans, results):
    """Add the planned experiments to the round with their results, as the agents and record-experiment would."""
    for p in plans:
        entry["experiments"].append({**p, **results[p["id"]], "id": p["id"], "phase": p["phase"]})


def test_screen_none_passes():
    s, entry, st = _round(_x("a", "failed"), _x("b", "error"))
    assert _run(s, entry, st) == [] and st["decided"] and st["final"] is None


def test_screen_one_passes_is_adopted_alone():
    s, entry, st = _round(_x("a"), _x("b", "failed"))
    assert _run(s, entry, st) == [] and st["final"] == "a"


def test_two_pass_combine_then_reuse_the_screen_for_ablation():
    s, entry, st = _round(_x("a", gain=0.05), _x("b", gain=0.1))
    plans = _run(s, entry, st)
    assert [p["id"] for p in plans] == ["comb"] and st["passers"] == ["b", "a"], (plans, st)
    assert {c["change"] for c in plans[0]["changes"]} == {"change a", "change b"}
    assert sorted(plans[0]["backlog"]) == ["Ba", "Bb"] and sorted(plans[0]["refs"]) == ["exp/v2-a", "exp/v2-b"]
    _report(entry, plans, {"comb": _x("comb", folds=[0.85, 0.86, 0.87, 0.88])})
    assert _run(s, entry, st) == [] and st["final"] == "comb", st   # a and b alone are worse than together


def test_ablation_drops_a_change_that_hurts_then_verifies():
    s, entry, st = _round(_x("a", gain=0.1), _x("b", gain=0.08), _x("c", gain=0.05, stage="training"))
    _report(entry, _run(s, entry, st), {"comb": _x("comb", folds=[0.85, 0.86, 0.87, 0.88])})
    plans = _run(s, entry, st)
    assert st["phase"] == "ablate" and [p["id"] for p in plans] == ["abl1", "abl2", "abl3"], plans
    by = {p["id"]: p for p in plans}
    # taking out a features change keeps the combination's tuned model fixed; taking out a training change cannot
    assert by["abl1"]["frozen_model"]["name"] == "m" and "frozen_model" not in by["abl3"], by
    assert all(p["compare_to"] == "comb" for p in plans)
    assert {c["change"] for c in by["abl2"]["changes"]} == {"change a", "change c"}
    _report(entry, plans, {"abl1": _x("abl1", folds=[0.75, 0.76, 0.77, 0.78]),   # without a: worse, a stays
                           "abl2": _x("abl2", folds=[0.90, 0.91, 0.925, 0.93]),  # without b: better, b goes
                           "abl3": _x("abl3", folds=[0.84, 0.86, 0.87, 0.87])})  # without c: no clear change
    plans = _run(s, entry, st)
    assert st["dropped"] == ["b"] and [p["id"] for p in plans] == ["ver"], st
    assert sorted(plans[0]["backlog"]) == ["Ba", "Bc"]
    _report(entry, plans, {"ver": _x("ver", "failed")})
    assert _run(s, entry, st) == [] and st["final"] == "comb", st   # the verify failed: keep the full combination


def test_failed_combination_falls_back_to_greedy():
    s, entry, st = _round(_x("a", gain=0.1), _x("b", gain=0.08), _x("c", gain=0.05))
    _report(entry, _run(s, entry, st), {"comb": _x("comb", "failed", folds=BASE)})
    plans = _run(s, entry, st)
    assert st["phase"] == "greedy" and [p["id"] for p in plans] == ["g1"] and plans[0]["compare_to"] == "a", plans
    _report(entry, plans, {"g1": _x("g1", folds=[0.86, 0.87, 0.88, 0.89])})   # a + b beats a
    plans = _run(s, entry, st)
    assert [p["id"] for p in plans] == ["g2"] and plans[0]["compare_to"] == "g1", plans
    assert sorted(plans[0]["backlog"]) == ["Ba", "Bb", "Bc"]
    _report(entry, plans, {"g2": _x("g2", folds=[0.85, 0.87, 0.88, 0.88])})   # adding c does not help
    assert _run(s, entry, st) == [] and st["final"] == "g1" and st["greedy"]["set"] == ["a", "b"], st


def test_autopilot_drafts_are_text_and_shown_at_the_checkpoint():
    assert m.assumption_text({"assumption": "x", "source": "autopilot draft"}) == "Autopilot draft: x"
    assert m.assumption_text({"assumption": "Autopilot draft: x", "source": "autopilot draft"}) == "Autopilot draft: x"
    assert m.assumption_text({"assumption": "y", "source": "user"}) == "y" and m.assumption_text(" z ") == "z"
    s = _spec(outcome="o")
    s["assumptions"] = ["Autopilot draft: capacity is 10%", "a person's assumption",
                        {"assumption": "an old mapping", "source": "autopilot draft"}]
    page = a.checkpoint_text(s, "business")
    assert "* capacity is 10%" in page and "* an old mapping" in page and "a person's" not in page, page


def test_write_stage_keeps_assumptions_as_text():
    import tempfile
    sys.path.insert(0, os.path.join(os.path.dirname(HERE), "_ml_shared"))
    import nb_common
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, m.SPEC_NAME)
        s = _spec()
        s["assumptions"] = ["kept"]
        m.save(p, s)
        nb_common.write_stage(p, "eda", {"mode": "targeted"}, "ml-eda-report",
                              assumptions=[{"assumption": "x", "source": "autopilot draft"}, "kept", "new"])
        assert m.load(p)["assumptions"] == ["kept", "Autopilot draft: x", "new"], m.load(p)["assumptions"]


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
