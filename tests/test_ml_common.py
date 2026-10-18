#!/usr/bin/env python3
"""Unit checks for the step order, the business modes, and the iteration bar in _ml_shared/ml_common.py.

Plain asserts, no test framework and no project packages needed:
    python3 tests/test_ml_common.py
Exit code 0 when every check passed, 1 otherwise.
"""

from __future__ import annotations

import os
import sys
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "_ml_shared"))

import ml_common as m  # noqa: E402


def _spec(**business):
    s = m.template("t")
    s["project"]["kernel"] = "k"
    s["business"].update(business)
    return s


def test_step_order():
    order = [s[0] for s in m.STEPS]
    assert order[:5] == ["setup", "business_anchors", "discovery", "business", "framing"], order
    assert m.STEP_STAGE["business_anchors"] == "business"
    assert [w[0] for w in m.WORKFLOW][1] == "business", "WORKFLOW keeps one business stage"


def test_new_project_starts_with_anchors():
    assert m.next_step(_spec())[0] == "business_anchors"


def test_standard_anchors_then_discovery():
    s = _spec(mode="standard", business_goal="g", decision_supported="d", status="draft")
    assert m.step_done(s, "business_anchors")
    assert not m.step_done(s, "business")
    assert m.next_step(s)[0] == "discovery"


def test_exploratory_anchors():
    s = _spec(mode="exploratory", learning_question="q", status="draft")
    assert not m.step_done(s, "business_anchors"), "answer_actions is the second anchor"
    s["business"]["answer_actions"] = [{"answer": "yes", "action": "build"}]
    assert m.step_done(s, "business_anchors")


def test_full_business_after_discovery():
    s = _spec(mode="standard", business_goal="g", decision_supported="d", status="draft")
    s["discovery"] = {"main_source": "x"}
    assert m.next_step(s)[0] == "business"
    s["business"]["status"] = "confirmed"
    assert m.next_step(s)[0] == "framing"


def test_old_spec_without_mode():
    s = _spec(status="confirmed", business_goal="g", decision_supported="d", outcome="o")
    assert m.business_mode(s) == "standard"
    assert m.step_done(s, "business_anchors"), "a confirmed old framing is not sent back to the anchors"
    assert m.validate(s)[0] == []


def test_validate_by_mode():
    s = _spec(mode="exploratory", status="confirmed", learning_question="q",
              answer_actions=[{"answer": "yes", "action": "build"}], outcome="o")
    errors, req, _ = m.validate(s)
    assert errors == [], errors
    assert not [r for r in req if r.startswith("business.")], req
    s["business"]["answer_actions"] = [{"answer": "yes"}]
    assert any("answer_actions[0]" in e for e in m.validate(s)[0])
    s["business"]["mode"] = "maybe"
    assert any("business.mode" in e for e in m.validate(s)[0])


def test_bar_pass():
    r = m.iteration_bar_evaluate([0.10, 0.11, 0.09], [0.12, 0.13, 0.12], lower_is_better=True)
    assert r["status"] == "pass" and r["wins"] == 3, r


def test_bar_fail_on_wins():
    r = m.iteration_bar_evaluate([0.10, 0.14, 0.13, 0.09], [0.12, 0.12, 0.12, 0.12], lower_is_better=True)
    assert r["status"] == "fail" and r["wins"] == 2, r


def test_bar_fail_on_spread():
    # Wins on 3 of 4 folds, but one large loss makes the spread bigger than the mean improvement.
    r = m.iteration_bar_evaluate([0.80, 0.81, 0.82, 0.60], [0.79, 0.80, 0.81, 0.75], lower_is_better=False)
    assert r["wins"] == 3 and r["status"] == "fail", r


def test_bar_higher_is_better():
    r = m.iteration_bar_evaluate([0.80, 0.81, 0.82], [0.70, 0.72, 0.71], lower_is_better=False)
    assert r["status"] == "pass", r


def test_bar_incomplete():
    assert m.iteration_bar_evaluate([0.1, 0.2], [0.2, 0.3], True)["status"] == "incomplete"
    assert m.iteration_bar_evaluate([], [0.2, 0.3, 0.4], True)["status"] == "incomplete"
    assert m.iteration_bar_evaluate([0.1, 0.2, 0.3], [0.2, 0.3], True)["status"] == "incomplete"


def test_bar_settings():
    s = _spec()
    assert m.iteration_bar(s)["min_folds"] == 3
    s["iterations"]["bar"] = {"min_folds": 5}
    assert m.iteration_bar(s)["min_folds"] == 5 and m.iteration_bar(s)["spread_multiple"] == 1.0


def test_kept_against_failed_bar_is_flagged():
    s = _spec()
    s["iterations"]["entries"] = [{"id": 1, "outcome": "kept", "bar_result": {"status": "fail"}}]
    assert any(c["check"] == "iteration bar" for c in m.consistency_checks(s))
    s["decisions"] = [{"decision": "Kept iteration 1 although the bar failed", "reason": "simpler model"}]
    assert not any(c["check"] == "iteration bar" for c in m.consistency_checks(s))


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
