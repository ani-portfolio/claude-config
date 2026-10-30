# Keyword Signals by Section

Grep for these terms (case insensitive) to find where each section lives. A hit is a lead, not evidence. Open the file and confirm what the code does before citing it.

## Problem formulation and labels
`target`, `label`, `y_true`, `y_col`, `ltv`, `churn`, `horizon`, `lookback`, `window`, `cutoff`, `as_of`, `prediction_date`, `objective=`, `task_type`

## Raw data
`read_sql`, `read_parquet`, `read_csv`, `spark.read`, `spark.table`, `bigquery`, `snowflake`, `redshift`, `databricks`, `s3://`, `gs://`, `abfss://`, `delta`, `kafka`, `kinesis`, `pubsub`, `FROM `, `JOIN `

## Data quality
`great_expectations`, `pandera`, `pydantic`, `deequ`, `tfdv`, `soda`, `dbt test`, `assert_schema`, `isnull`, `validate`, `expectation`

## Feature pipeline and features
`feature`, `transform`, `ColumnTransformer`, `Pipeline(`, `OneHotEncoder`, `TargetEncoder`, `StandardScaler`, `rolling(`, `groupby(`, `agg(`, `point_in_time`, `merge_asof`, `as_of_join`, `on_demand`

## Feature store
`feast`, `FeatureView`, `feature_store.yaml`, `get_historical_features`, `get_online_features`, `materialize`, `tecton`, `hopsworks`, `FeatureStoreClient`, `FeatureEngineeringClient`, `vertex_ai.Featurestore`, `sagemaker.feature_store`, `online_store`, `offline_store`

## Training
`fit(`, `train(`, `xgboost`, `lightgbm`, `catboost`, `sklearn`, `torch`, `tensorflow`, `keras`, `prophet`, `statsmodels`, `optuna`, `hyperopt`, `GridSearchCV`, `RandomizedSearchCV`, `KFold`, `StratifiedKFold`, `GroupKFold`, `TimeSeriesSplit`, `train_test_split`, `early_stopping`, `sample_weight`, `class_weight`

## Evaluation
`rmse`, `mae`, `mape`, `wape`, `r2`, `auc`, `roc_auc`, `pr_auc`, `logloss`, `ndcg`, `precision`, `recall`, `calibration`, `decile`, `lift`, `gini`, `backtest`, `baseline`, `threshold`

## Model registry
`mlflow`, `log_model`, `register_model`, `transition_model_version_stage`, `set_registered_model_alias`, `champion`, `challenger`, `wandb.Artifact`, `ModelRegistry`, `aiplatform.Model.upload`, `model_package`, `model_version`

## Orchestration and scheduling
`airflow`, `DAG(`, `@dag`, `@task`, `kfp`, `kubeflow`, `prefect`, `@flow`, `dagster`, `@asset`, `argo`, `step functions`, `sagemaker.workflow`, `vertex pipelines`, `databricks jobs`, `cron`, `schedule_interval`, `schedule=`

## Deployment
`Dockerfile`, `kserve`, `seldon`, `bentoml`, `torchserve`, `triton`, `ray serve`, `sagemaker endpoint`, `Endpoint(`, `deploy(`, `canary`, `shadow`, `blue_green`, `traffic_split`, `traffic_percentage`, `rollout`, `helm`, `kubernetes`

## Experimentation
`ab_test`, `a/b`, `experiment`, `treatment`, `control`, `holdout`, `variant`, `bucket`, `assignment`, `randomiz`, `split_ratio`, `statsig`, `optimizely`, `growthbook`, `launchdarkly`, `eppo`, `explore`, `epsilon`

## Inference
`predict(`, `predict_proba(`, `FastAPI`, `@app.post`, `flask`, `grpc`, `lambda_handler`, `batch_predict`, `score(`, `inference`, `serving`, `load_model`, `joblib.load`, `pickle.load`

## Outputs and fallback
`fallback`, `default_value`, `except`, `clip(`, `np.clip`, `quantile`, `percentile`, `cap`, `floor`, `write_to`, `to_parquet`, `to_sql`, `publish`, `upload`, `conversions api`, `offline conversion`

## Monitoring and retraining
`evidently`, `whylogs`, `arize`, `fiddler`, `nannyml`, `alibi_detect`, `drift`, `ks_2samp`, `kstest`, `jensenshannon`, `psi`, `population_stability`, `prometheus`, `datadog`, `grafana`, `cloudwatch`, `latency`, `throughput`, `alert`, `retrain`, `trigger`
