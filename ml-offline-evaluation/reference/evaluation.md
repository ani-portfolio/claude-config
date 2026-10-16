# Offline Evaluation Guide

This guide explains what the evaluation notebook measures and how to read each part.

## What is scored

The champion is fixed in model training, before the test set is opened. Evaluation scores it, every other ML model (the best trial of each on the current data), and every baseline, all on the same test rows. The champion is marked "(champion)" in every table and drawn as a thick red line in every chart; the reference is grey.

The champion is loaded from its saved file. The other models and the baselines are refitted on the whole training set with their best settings, which is the same treatment the champion had.

Evaluation never picks a new champion. If another model scores better on the test set, the notebook says so, but switching to it would choose the model with the test set, and its test score would then be optimistic. The honest path is to go back to model training, change the selection there for a reason that does not depend on the test results, and accept that the next test score is less clean.

## Bootstrap intervals

Each metric has a 95% interval from bootstrap draws of the test rows. When rows are related, they are drawn together:

* Time split: rows with the same date are drawn together, because one unusual day moves every score of that day.
* Group split: rows of the same group are drawn together.
* Random split: single rows.

The comparisons against the reference and the best baseline are paired: both models are scored on the same draw, so the interval of the difference is narrower than the two separate intervals suggest. A difference whose interval is above zero is a real improvement on this test set.

At k metrics with many tied scores (rules and prior baselines often give the same score to many rows) are averaged over 20 random tie breaks, so a baseline is not lucky or unlucky by the order of the rows.

## Charts

* **Gain curve** (classification): the share of all positives found when acting on the top x% of rows. The capacity line shows the operating point from the framing.
* **Primary metric as k changes** (at k metrics): whether the champion stays ahead at other capacities.
* **ROC and precision recall curves** (classification).
* **Predicted against actual and residuals** (regression and forecasting).
* **Primary metric over the test period**: whether the gain holds every period or comes from a few.
* **Calibration and scores by class** (classification): whether the scores can be read as probabilities, and how well they separate the classes.
* **Slice dot plot**: the champion's difference from the reference in each slice, with its interval.

## Slices

A slice is a group of test rows with the same value of a column. The skill proposes:

* the categorical features the business uses to split work,
* up to three of the most important numeric features, cut into quarters on the training rows,
* the time period,
* whether the entity was seen in training (new customers, stores, or products),
* columns with more than 5% missing values among the top features,
* columns named in a fairness constraint.

A slice with fewer than `min_rows` rows or `min_positives` positives is shown but marked too small: its interval is too wide to read. The flags are:

* **worse than reference**: the whole interval of the difference is below zero.
* **far below overall**: the champion's score in the slice is far below its overall score. This flag is not used for at k metrics, where the score of a slice depends on how many of its rows are in the top k.

For at k metrics the table also shows how many rows of the slice each model acts on, because a slice can look weak simply because the model sends its capacity elsewhere.

## Other checks

* **CV to test**: a test score far below the CV score means the tuning fitted the folds, or the data changed after the training period.
* **Permutation importance on test against training**: a feature that matters much less on the test set may have changed.
* **Score drift (PSI)**: the champion's test scores against its out of fold scores. Under 0.1 is stable, 0.1 to 0.25 is moderate, above 0.25 is large.
* **Test reuse**: every run records when the test set was opened. A later run with another champion or other data warns that the test score may be optimistic.

## The decision

The notebook ends with an evidence table: one question and one answer per row. It does not decide. The user records the decision (go, iterate, or no go) and the reason in the `decisions` cell.

## Model card

`docs/model_card.md` is written from the spec: the purpose, the data and the split, the features, the champion and its settings, the test results with intervals, the comparisons, the slices that need attention, the limitations, and the open questions. Edit it freely; it is rewritten on the next run, so keep edits you want to keep in the spec or in a copy.
