# Checkpoints

A checkpoint is where the agent stops for a person. It writes `docs/checkpoint.md`, records the checkpoint as
`open` in the spec's `autopilot.checkpoints`, sends a push notification, and ends its turn. Starting
`/ml-autopilot` again finds the open checkpoint and reviews it with the person first.

| Kind | When | The page shows | Resuming |
|---|---|---|---|
| `business` | The full business framing is drafted from the data | Every field of the `business` section, the assumptions the agent made, and the red flags (empty required fields, notes that disagree, open questions) | Disputed rows one question at a time, then ml-business-framing runs with the reviewed answers and confirms the section |
| `framing` | The ML framing is drafted | The same for the `framing` section: target, unit, prediction time, window, metric, V1 bar and its criteria | The same, with ml-problem-framing; the modelling gate opens |
| `test_v1` | Training is done | The CV leaderboard, the V1 bar on CV, and how many times the test set was opened | Asks whether to open the test set; yes runs ml-offline-evaluation |
| `test_final` | The rounds are over and at least one was kept | The same, plus every experiment of every round with its bar result | The same; this look is optimistic, and the page says so |

Rules:

* The agent never confirms a framing itself. `/ml-status` reports a framing that is confirmed while its checkpoint
  is still open.
* A checkpoint is cleared only after the review: `autopilot.py checkpoint clear <kind>`.
* Writing a new checkpoint of a kind clears an older open one of the same kind.
* When the person declines to open the test set, the run stops (`autopilot.py stop`) and the CV result is the
  outcome.
