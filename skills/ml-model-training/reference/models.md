# Models and Tuning

## The model menu

| Type | Library | Preprocessor | Good for | Watch out for |
|---|---|---|---|---|
| `logistic` | scikit-learn | linear | The first model for classification. Shows whether the features carry signal; coefficients explain each score. | Misses interactions and curved effects unless features capture them. |
| `ridge` | scikit-learn | linear | The first model for regression and forecasting. | Same as logistic. |
| `ridge_nonneg` | SciPy | linear | Ridge with the coefficients of the inputs listed in `params.nonneg` kept at zero or above, for effects that cannot be negative, such as media in a marketing mix model. Alpha is searched up to 1000. | A bounded coefficient can sit at exactly zero, which means no measurable effect, not a proven one. |
| `tree` | scikit-learn | tree | A readable set of rules to compare with the business rule. | Rarely the most accurate. |
| `random_forest` | scikit-learn | tree | A robust model with little tuning. | Large files; slower to score; probabilities often poorly calibrated. |
| `hist_gbm` | scikit-learn | tree | Gradient boosting with no extra package. | Fewer options than the libraries below. |
| `lightgbm` | lightgbm | tree | Fast gradient boosting, usually the strongest on tables. | Overfits small data without regularisation. |
| `xgboost` | xgboost | tree | Gradient boosting with strong regularisation. | Slower than LightGBM on large data. |
| `catboost` | catboost | tree | Gradient boosting that copes well with many categories. | Slowest to train of the three. |

All use `src.features.make_preprocessor` (linear or tree) inside the same pipeline, fitted inside each fold. Classification types output probabilities.

## Questions worth asking per model

* **Tuning budget:** `none` (the default settings, one trial), `quick` (20 trials, at most 10 minutes), or `thorough` (60 trials, at most 30 minutes). `--suggest` estimates the time of one trial from a real fit. Linear models need little tuning; gradient boosting gains the most from it.
* **Class weights** (classification, rare positive class): `balanced` makes errors on the rare class cost more. It can help threshold decisions, rarely changes a ranking metric much, and makes the probabilities too high, so do not use it when the output is read as a probability.
* **Fixed settings:** anything the user wants held fixed (for example `max_depth: 3` for a model that must stay simple) goes in `params` and is never tuned.
* **Inputs subset:** a model can use only some of the feature inputs, listed in `inputs` (for example a reference model with trend and seasonality but no media). Every other input is excluded from that model's preprocessor, the subset is part of the run log key, and permutation importance shows zero for the inputs it does not use. Without `inputs`, a model uses every input.
* **Interpretability:** when the business framing asks for reasons per prediction, recommend logistic regression first, and point out that the feature importance in the notebook explains the model as a whole, not each score.

## Tuning

The default settings are always scored first (trial 0), so tuning can never end worse than the defaults. Optuna's TPE sampler then proposes settings from each type's search space, learning from earlier trials. Every trial is scored on all CV folds, with the preprocessor fitted inside each fold. The search stops at `n_trials` or `timeout_minutes`, whichever comes first.

Choosing the best of many trials on the same folds makes the best CV score slightly optimistic. The test set in `/ml-offline-evaluation` gives the honest score; a large drop there means the tuning overfitted the folds.

## The run log

`training_runs.parquet` holds one row per trial: the model, its type, the settings, the mean and spread of the primary metric, every metric, the score per fold, the seconds it took, the data version, and when it ran. It is only appended to.

A tuning run (a study) is identified by the model type, fixed settings, class weights, tuning budget, metrics, and the data version (a hash of the feature settings and the split). When the notebook runs, a study already in the log is read back instead of trained again. Changing any of those, or rerunning the features or the split, makes a new study. Runs on older data stay in the log but are left out of the leaderboard.

## Adding models later

Run `/ml-model-training` again, or add a model to `MODELS` in `08_training.ipynb` and choose Run All. Only the new model is trained. Names must be unique; to retrain a model with new settings, give it a new name (to keep both) or replace it.

## The champion

The model with the best mean CV score on the primary metric. It is refitted on every training row, saved to `models/champion_<name>.joblib` (a fitted pipeline, the one file that is not parquet), and evaluated once on the test set by `/ml-offline-evaluation`. When two models are close, the leaderboard shows it; the user can still prefer the simpler one, by keeping only it in `MODELS` or recording the choice in `decisions`.

## Checks

| Check | Problem when |
|---|---|
| Leakage | AUC above 0.97: a feature probably leaks the outcome. |
| Against the baselines | The champion does not beat the reference or the best baseline. |
| V1 bar | The improvement over the reference is below the percentage in `framing.success.v1_bar` (when it states one). |
| Overfitting | The score on the training rows is more than 30% away from the CV score. Not judged when TargetEncoder columns are used, because their training rows are encoded with their own targets. |
| Stability | The spread across folds is more than 20% of the mean. |

The notebook also shows the tuning history (is the curve still rising?), permutation importance of the champion, and for classification a calibration table and chart.
