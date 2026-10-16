# How the Data Dictionary Is Drafted

`scripts/discover.py` drafts every entry from the column name and the data. It does not guess beyond what the name and the data support: when it cannot work something out, it leaves it empty and flags the column for review.

## Meaning

1. The name is split into words (`days_since_last_order` becomes "days since last order", `txnCnt7d` becomes "transaction count over the last 7 days").
2. Common abbreviations are expanded, for example `amt` to amount, `txn` to transaction, `cnt` to count, `mcc` to merchant category code, `acct` to account, and `pct` to percent. The full list is `ABBREV` in `discover.py`.
3. A window such as `90d` becomes "over the last 90 days". In a column that looks like an outcome, it becomes "over a 90 day window", because the name does not say whether the window runs before or after a date.
4. Flags such as `is_active` or `has_card` become "Whether ...".
5. Facts from the data are added: the number of distinct values for categories, and the date range for dates.
6. The meaning is left empty and flagged `unclear_meaning` when the name is generic (`value`, `col3`, `feature_12`, `Unnamed: 0`) or made only of codes (`c14`, `v23`, `f_3`, hashes).

## Timing

Timing says when a column is filled in relative to the prediction. It decides whether the column can be a feature. The draft follows these rules, in order:

| Timing | When it is chosen |
|---|---|
| `identifier` | The column is an ID, by role or by a name such as `*_id`, `*_key`, `*_no`. |
| `possibly_after_outcome` | The column looks like an outcome (a candidate target), or its name contains words such as refund, chargeback, cancel, closed, resolved, settled, final, outcome, result, after, next, writeoff, or default. |
| `static` | A date set once (sign up, created, opened, birth), or a stable attribute such as region, country, plan, segment, type, category, channel, or merchant category code. |
| `event_time` | Any other date or timestamp. A candidate for the prediction time column. |
| `before_prediction` | The name describes history up to a point: since, last, previous, prior, rolling, a window like `30d`, average, count, total, minimum, maximum, recency, or tenure. |
| `at_event` | A plain attribute in a table with one row per event and an event time, such as the amount of a transaction. This is an assumption, stated in `timing_basis`. |
| `unknown` | None of the above. Flagged `timing_unknown`. |

## Flags and review

`needs_review` is true when the meaning is unclear, the timing is unknown, or the column may be recorded after the outcome (including candidate targets). Other flags (`personal_data`, `free_text`, `mostly_missing`, `constant`) are information and do not by themselves need review.

## Keeping reviewed entries

`meaning_source` records where the meaning came from: `name`, `unclear`, `reviewed` (Claude improved the wording from context the user gave), or `user` (the user confirmed it). Entries marked `reviewed` or `user` keep their meaning, timing, and notes when `discover.py` runs again, so a rerun after new data arrives does not undo the review. Statistics such as missing % are always refreshed.

## What Claude may change

Claude may improve the wording of a drafted meaning when the name is clear, and mark it `reviewed`. Claude may fill in an unclear meaning or an unknown timing only from something the user or a document said, and then records the source in `notes`. Claude never fills a flagged entry with a guess.
