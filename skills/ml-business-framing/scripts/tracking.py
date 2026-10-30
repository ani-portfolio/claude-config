"""MLflow tracking for the ML project skills.

Every project logs to its own experiment on one shared MLflow server (see tools/mlflow_server.py in the
skills folder). The spec stays the record of decisions; MLflow records each run of a stage notebook so runs
can be browsed and compared. Tracking never blocks the work: when MLflow is not installed or the server cannot be
reached, the notebook prints one line and carries on.

The tracking address comes from, in order: the MLFLOW_TRACKING_URI environment variable, the spec's
tracking.uri, then the local server at http://127.0.0.1:5000.

Inside a stage notebook:
    tracking.capture_figures()                          # once, before any chart (the setup cell does this)
    tracking.log_stage(SPEC_PATH, STAGE, NOTEBOOK, SKILL)   # after the spec is written

Command line (with the project's Python, from the project root):
    tracking.py setup       create or find the project's experiment and write the spec's tracking section
    tracking.py status      the address, whether it is reachable, the experiment, and the current run per stage
    tracking.py export [folder]   copy the project's runs (metrics, parameters, tags, artifacts) into a folder,
                                  default mlflow_export/ in the project, to archive or hand over the project
    tracking.py start-experiment <experiment> [hypothesis]   create the parent run of an ml-iterate experiment
                                  and print its id
    tracking.py promote <experiment>    make an experiment's stage runs current and move the champion alias to it
    tracking.py reject <experiment> [reason]   mark an experiment as rejected (tags only)
    tracking.py restore-current   make the runs the spec names current again (after a rollback)
    tracking.py tracing [hook command]   trace Claude Code sessions in this project into its experiment

Candidate mode. When ML_ITERATE_EXPERIMENT is set (ml-iterate sets it while it runs an experiment in a workspace),
every stage run is logged as a child of the experiment's parent run (ML_ITERATE_PARENT_RUN), tagged candidate=true
with the experiment, iteration (ML_ITERATE_ITERATION) and agent session (ML_AGENT_SESSION). It never becomes the
current run of its stage, and the champion it registers never gets the alias 'champion'. Only promote does that.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import re
import sys
import tempfile
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
# The spec library: ab_common in the A/B testing skills (ab_spec.yaml), ml_common in the ML project skills.
if os.path.exists(os.path.join(HERE, "ab_common.py")):
    import ab_common as ml_common  # noqa: E402
    KIND = "ab"
else:
    import ml_common  # noqa: E402
    KIND = "ml"

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
os.environ.setdefault("MLFLOW_DISABLE_TELEMETRY", "true")   # MLflow otherwise sends usage data to its makers
os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "2")
os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "30")
os.environ.setdefault("MLFLOW_SUPPRESS_PRINTING_URL_TO_STDOUT", "true")
os.environ.setdefault("MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR", "false")

DEFAULT_URI = "http://127.0.0.1:5000"
ML_STAGES = ("split", "features", "baseline", "training", "evaluation", "review")
AB_STAGES = ("framing", "design", "validation", "analysis", "segments", "readout")
LOGGED_STAGES = AB_STAGES if KIND == "ab" else ML_STAGES
SERVER_HELP = ("Start the shared server with `python3 ~/Projects/claude-config/skills/tools/mlflow_server.py start` "
               "(or install-agent to start it at every login).")
_FIGURES = []
_WARNED = set()


def _warn(msg):
    if msg not in _WARNED:
        _WARNED.add(msg)
        print("MLflow: " + msg)


# ---------------------------------------------------------------- address and experiment

def tracking_uri(spec=None):
    return os.environ.get("MLFLOW_TRACKING_URI") or ((spec or {}).get("tracking") or {}).get("uri") or DEFAULT_URI


def reachable(uri, timeout=1.5):
    if not uri.startswith(("http://", "https://")):
        return True
    try:
        with urllib.request.urlopen(uri.rstrip("/") + "/health", timeout=timeout) as r:
            return r.status == 200
    except OSError:
        return False


def _mlflow():
    import warnings
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            import mlflow
        return mlflow
    except ImportError:
        return None


def ready(spec):
    """(mlflow module, uri) when logging can work, else (None, reason)."""
    mlflow = _mlflow()
    if mlflow is None:
        return None, "not installed in this project, so nothing was logged. Add the client with `uv add 'mlflow-skinny>=3.17,<4'`."
    uri = tracking_uri(spec)
    if not reachable(uri):
        return None, f"the server at {uri} is not reachable, so nothing was logged. {SERVER_HELP}"
    mlflow.set_tracking_uri(uri)
    return mlflow, uri


def slug(text):
    return re.sub(r"[^a-z0-9]+", "-", str(text).lower()).strip("-") or "project"


def experiment_name(spec, prefix=KIND):
    name = (spec.get("project") or {}).get("name") or "project"
    return f"{prefix}/{slug(name)}"


def registered_model_name(spec):
    return f"{slug((spec.get('project') or {}).get('name'))}-champion"


def ui_link(uri, experiment_id, run_id=None):
    if not uri.startswith("http"):
        return None
    base = uri.rstrip("/") + f"/#/experiments/{experiment_id}"
    return base + (f"/runs/{run_id}" if run_id else "")


def ensure_experiment(spec, root, mlflow, uri):
    """The project's experiment id, created when missing. A name already used by a project in another folder gets
    a suffix, so two projects never share an experiment."""
    from mlflow.tracking import MlflowClient
    client = MlflowClient()
    root = os.path.abspath(root)
    tr = spec.get("tracking") or {}
    name = tr.get("experiment") if tr.get("uri") == uri and tr.get("experiment") else experiment_name(spec)
    exp = client.get_experiment_by_name(name)
    if exp is not None and exp.lifecycle_stage == "deleted":
        client.restore_experiment(exp.experiment_id)
    if exp is not None and exp.tags.get("project_root") not in (None, root) and name != tr.get("experiment"):
        name = f"{name}-{hashlib.sha256(root.encode()).hexdigest()[:6]}"
        exp = client.get_experiment_by_name(name)
    if exp is None:
        exp_id = client.create_experiment(name, tags={"project_root": root,
                                                      "project": str((spec.get("project") or {}).get("name")),
                                                      "mlflow.note.content": str((spec.get("project") or {}).get("description") or "")})
    else:
        exp_id = exp.experiment_id
        if not exp.tags.get("project_root"):
            client.set_experiment_tag(exp_id, "project_root", root)
    return name, exp_id


def write_tracking_section(spec_path, uri, name, exp_id, skill=f"{KIND}-project-setup"):
    spec = ml_common.load(spec_path)
    if os.environ.get("MLFLOW_TRACKING_URI") and (spec.get("tracking") or {}).get("uri") not in (None, uri):
        return  # an address set for this shell only does not replace the project's own
    old = spec.get("tracking") or {}
    spec["tracking"] = {"uri": uri, "experiment": name, "experiment_id": str(exp_id), "ui": ui_link(uri, exp_id)}
    if KIND == "ml":   # a name the project already chose is kept
        spec["tracking"]["registered_model"] = old.get("registered_model") or registered_model_name(spec)
    ml_common.log(spec, skill, f"tracking: experiment {name} on {uri}")
    ml_common.save(spec_path, spec)


# ---------------------------------------------------------------- figures

def _save_open_figures(folder):
    import matplotlib.pyplot as plt
    for num in plt.get_fignums():
        fig = plt.figure(num)
        if getattr(fig, "_tracking_saved", False):
            continue
        title = next((a.get_title() for a in fig.axes if a.get_title()), "") or (
            fig._suptitle.get_text() if getattr(fig, "_suptitle", None) else "chart")
        path = os.path.join(folder, f"{len(_FIGURES) + 1:02d}_{slug(title)[:50]}.png")
        try:
            fig.savefig(path, dpi=110, bbox_inches="tight")
            fig._tracking_saved = True
            _FIGURES.append(path)
        except Exception:  # noqa: BLE001  a chart that cannot be saved is simply not logged
            pass


def capture_figures():
    """Keep a copy of every matplotlib chart the notebook shows, so log_stage can attach the charts. Works for
    plt.show() and for charts Jupyter shows at the end of a cell."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    if getattr(plt.show, "_tracking", False):
        return
    folder = tempfile.mkdtemp(prefix="mlflow_figs_")
    original = plt.show

    def show(*args, **kwargs):
        _save_open_figures(folder)
        return original(*args, **kwargs)

    show._tracking = True
    plt.show = show
    try:  # charts shown by Jupyter at the end of a cell: save them just before the inline backend closes them
        ip = get_ipython()  # noqa: F821  defined inside IPython
    except NameError:
        return
    hooks = ip.events.callbacks.get("post_execute", [])
    for cb in list(hooks):
        if getattr(cb, "__name__", "") == "flush_figures":
            def flush(_cb=cb):
                _save_open_figures(folder)
                return _cb()
            flush.__name__ = "flush_figures_tracked"
            hooks[hooks.index(cb)] = flush


# ---------------------------------------------------------------- helpers for loggers

_KEY = re.compile(r"[^A-Za-z0-9_\-. /]")


def key(name):
    return _KEY.sub("_", str(name).replace("%", "pct")).strip()[:250] or "value"


def _num(v):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if f != f or f in (float("inf"), float("-inf")) else f


def flat(d, prefix=""):
    """A nested dict as flat MLflow parameters: lists joined with commas, long values cut."""
    out = {}
    for k, v in (d or {}).items():
        name = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flat(v, name + "."))
        elif isinstance(v, (list, tuple)):
            out[key(name)] = ", ".join(json.dumps(x, default=str) if isinstance(x, dict) else str(x) for x in v)[:6000]
        elif v is not None:
            out[key(name)] = str(v)[:6000]
    return out


class Logger:
    """Logging into one run. Every call is safe: a failure is reported once and skipped."""

    def __init__(self, mlflow, run_id, root):
        self.mlflow, self.run_id, self.root = mlflow, run_id, root
        from mlflow.tracking import MlflowClient
        self.client = MlflowClient()

    def _try(self, what, fn):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001
            _warn(f"could not log {what}: {type(e).__name__}: {str(e)[:200]}")

    def params(self, d):
        from mlflow.entities import Param
        items = list(flat(d).items())
        for i in range(0, len(items), 100):
            chunk = [Param(k, v) for k, v in items[i:i + 100]]
            self._try("parameters", lambda: self.client.log_batch(self.run_id, params=chunk))

    def metrics(self, d, step=0):
        import time
        from mlflow.entities import Metric
        now = int(time.time() * 1000)
        vals = [Metric(key(k), _num(v), now, step) for k, v in (d or {}).items() if _num(v) is not None]
        for i in range(0, len(vals), 500):
            chunk = vals[i:i + 500]
            self._try("metrics", lambda: self.client.log_batch(self.run_id, metrics=chunk))

    def series(self, name, values):
        import time
        from mlflow.entities import Metric
        now = int(time.time() * 1000)
        vals = [Metric(key(name), _num(v), now, i) for i, v in enumerate(values or []) if _num(v) is not None]
        if vals:
            self._try(name, lambda: self.client.log_batch(self.run_id, metrics=vals))

    def tags(self, d):
        from mlflow.entities import RunTag
        vals = [RunTag(key(k), str(v)[:5000]) for k, v in (d or {}).items() if v is not None]
        if vals:
            self._try("tags", lambda: self.client.log_batch(self.run_id, tags=vals))

    def table(self, rows, name, folder="tables"):
        """A small table as CSV, so it can be read in the MLflow UI. The project's own copy stays parquet."""
        import pandas as pd
        df = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows or [])
        if df.empty:
            return
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, f"{slug(name)}.csv")
            df.to_csv(p, index=False)
            self._try(name, lambda: self.client.log_artifact(self.run_id, p, folder))

    def file(self, path, folder=None):
        p = path if os.path.isabs(path) else os.path.join(self.root, path)
        if p and os.path.isfile(p):
            self._try(os.path.basename(p), lambda: self.client.log_artifact(self.run_id, p, folder))

    def parquet(self, rel, name, folder="tables", max_rows=50000):
        import pandas as pd
        p = os.path.join(self.root, rel) if rel else None
        if p and os.path.exists(p):
            df = pd.read_parquet(p)
            if len(df) <= max_rows:
                self.table(df, name, folder)

    def child(self, name, tags=None):
        run = self._try(f"run {name}", lambda: self.client.create_run(
            self.client.get_run(self.run_id).info.experiment_id, run_name=name,
            tags={"mlflow.parentRunId": self.run_id, **{key(k): str(v) for k, v in (tags or {}).items()}}))
        return Logger(self.mlflow, run.info.run_id, self.root) if run else None

    def end(self, status="FINISHED"):
        self._try("run end", lambda: self.client.set_terminated(self.run_id, status))


def _read(root, rel):
    import pandas as pd
    p = os.path.join(root, rel) if rel else None
    return pd.read_parquet(p) if p and os.path.exists(p) else None


def _md5(root, rel):
    p = os.path.join(root, rel) if rel else None
    if not p or not os.path.isfile(p):
        return None
    h = hashlib.md5()
    with open(p, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()[:12]


def _higher(spec):
    try:
        import model_core
        k = (((spec.get("baseline") or {}).get("metrics") or {}).get("primary") or {}).get("key")
        return model_core.HIGHER_IS_BETTER.get(k) is not False
    except Exception:  # noqa: BLE001
        return True


# ---------------------------------------------------------------- what each stage logs

def _log_split(L, spec, root):
    sp = spec.get("split") or {}
    cv = sp.get("cv") or {}
    L.params({k: sp.get(k) for k in ("strategy", "target", "task_type", "date_col", "group_col", "row_id_col",
                                     "gap_days", "seed", "stratify")})
    L.params({f"cv.{k}": cv.get(k) for k in ("method", "n_folds", "window", "valid_days", "train_days")})
    L.params({f"test.{k}": (sp.get("test") or {}).get(k) for k in ("start", "end", "days", "size", "share")})
    L.params({"duplicates.handling": (sp.get("duplicates") or {}).get("handling")})
    rows = sp.get("rows") or {}
    L.metrics({f"rows_{k}": v for k, v in rows.items()})
    L.metrics({"test_share": (sp.get("test") or {}).get("share"),
               **{f"dropped_{k}": v for k, v in (sp.get("dropped") or {}).items()},
               "test_rows_with_copy_in_train": (sp.get("checks") or {}).get("test_rows_with_copy_in_train")})
    files = sp.get("files") or {}
    L.tags({f"md5.{k}": _md5(root, v) for k, v in files.items()})
    target = sp.get("target")
    mlflow = L.mlflow
    for part, context in (("train", "training"), ("test", "test")):
        df = _read(root, files.get(part))
        if df is None:
            continue
        if target in df and df[target].nunique() <= 2:
            L.metrics({f"positive_share_{part}": float((df[target] == df[target].max()).mean())})
        def log_ds():
            ds = mlflow.data.from_pandas(df, source=os.path.join(root, files[part]), name=part,
                                         targets=target if target in df else None)
            with mlflow.start_run(run_id=L.run_id):
                mlflow.log_input(ds, context=context)
        L._try(f"the {part} dataset", log_ds)
    L.table(cv.get("folds") or [], "cv_folds")


def _log_features(L, spec, root):
    fe = spec.get("features") or {}
    cfg = fe.get("config") or {}
    L.params(flat(cfg, "config."))
    L.params({"n_inputs": fe.get("n_inputs"), **{f"outputs.{k}": v for k, v in (fe.get("outputs") or {}).items()}})
    chk = fe.get("checks") or {}
    L.metrics({"n_inputs": fe.get("n_inputs"), "n_excluded": len(fe.get("excluded") or []),
               "n_flagged_leakage": len(chk.get("flagged") or []), "n_drift_large": len(chk.get("drift_large") or []),
               "leakage_threshold": chk.get("leakage_threshold"),
               **{f"outputs_{k}": v for k, v in (fe.get("outputs") or {}).items()}})
    L.tags({"features_hash": (fe.get("code") or {}).get("features_hash")})
    L.table(fe.get("inputs") or [], "inputs")
    L.table(fe.get("excluded") or [], "excluded")
    L.table(chk.get("strongest") or [], "strongest_features")
    for k in ("features", "custom"):
        L.file((fe.get("code") or {}).get(k), "code")


def _log_baseline(L, spec, root):
    bl = spec.get("baseline") or {}
    label = bl.get("primary_label")
    L.params({"primary_metric": label, "secondary_metrics": ((bl.get("metrics") or {}).get("secondary")),
              "reference": (bl.get("reference") or {}).get("name"), "best": (bl.get("best_non_ml") or {}).get("name")})
    L.metrics({"reference_" + label: (bl.get("reference") or {}).get(label),
               "best_" + label: (bl.get("best_non_ml") or {}).get(label),
               "best_vs_reference_pct": bl.get("best_vs_reference_pct")})
    results = {r.get("model"): r for r in bl.get("results") or []}
    for m in bl.get("models") or []:
        C = L.child(m.get("name"), {"stage": "baseline", "kind": "baseline", "model_type": m.get("type")})
        if not C:
            continue
        C.params({"type": m.get("type"), **flat(m.get("params") or {}, "params.")})
        C.tags({"reason": m.get("reason")})
        C.metrics({k: v for k, v in (results.get(m.get("name")) or {}).items() if k not in ("model", "kind", "type")})
        C.end()
    L.table(bl.get("results") or [], "cv_results")


def _best_trials(spec, root):
    import pandas as pd
    tr = spec.get("training") or {}
    runs = _read(root, (tr.get("files") or {}).get("runs"))
    if runs is None or not len(runs):
        return {}, runs
    cur = runs[runs["data_version"] == tr.get("data_version")] if "data_version" in runs else runs
    higher = _higher(spec)
    best = {}
    for name, g in cur.groupby("model"):
        g = g.dropna(subset=["primary_mean"])
        if len(g):
            best[name] = (g.sort_values("primary_mean", ascending=not higher).iloc[0], g)
    return best, pd.DataFrame(runs)


def _register_champion(C, spec, root):
    """Log the champion pipeline as an MLflow model and give it the alias 'champion', unless the same run is
    already registered."""
    tr = spec.get("training") or {}
    champ = tr.get("champion") or {}
    path = os.path.join(root, champ.get("path") or "")
    if not champ.get("path") or not os.path.exists(path):
        return
    mlflow, client = C.mlflow, C.client
    name = (spec.get("tracking") or {}).get("registered_model") or registered_model_name(spec)
    try:
        versions = client.search_model_versions(f"name='{name}'")
    except Exception:  # noqa: BLE001
        versions = []
    same = [v for v in versions if v.tags.get("run_log_id") == str(champ.get("run_id"))
            and v.tags.get("data_version") == str(tr.get("data_version"))]
    reused = bool(same)
    if same:
        version = same[0].version
    else:
        import joblib
        pipe = joblib.load(path)
        example = None
        try:
            sys.path.insert(0, root)
            import importlib
            F = importlib.import_module("src.features")
            cfg = F.load_config(spec, root) if "root" in F.load_config.__code__.co_varnames else F.load_config(spec)
            X = _read(root, ((spec.get("features") or {}).get("files") or {}).get("train"))
            wanted = champ.get("inputs") or F.feature_columns(cfg)   # a model may use a subset of the inputs
            cols = [c for c in wanted if X is not None and c in X]
            example = X[cols].head(5) if cols else None
        except Exception:  # noqa: BLE001  the model is logged without an example
            example = None
        # The project's src code and the skills' model_core.py (home of estimators such as NonNegRidge) go with the
        # model, so it loads on a machine without the skills.
        src = os.path.join(root, "src")
        extra_code = [p for p in (src, os.path.join(HERE, "model_core.py")) if os.path.exists(p)]
        with mlflow.start_run(run_id=C.run_id):
            # cloudpickle, like the project's own joblib file: the pipeline holds the project's own steps, which
            # the newer skops format refuses unless every type is listed as trusted.
            info = mlflow.sklearn.log_model(pipe, name="model", input_example=example,
                                            code_paths=extra_code or None,
                                            serialization_format="cloudpickle",
                                            pyfunc_predict_fn="predict_proba" if hasattr(pipe, "predict_proba") else "predict",
                                            registered_model_name=name)
        version = info.registered_model_version
        for k, v in {"run_log_id": champ.get("run_id"), "data_version": tr.get("data_version"),
                     "project_root": root, "model_type": champ.get("type")}.items():
            client.set_model_version_tag(name, version, k, str(v))
    cand = candidate(spec, root)
    ver = current_version(spec)
    C.tags({"registered_model": name, "registered_version": version})
    # Every version's champion keeps an alias of its own (champion-v1, champion-v2, ...), so versions can be
    # loaded and compared. Experiments of one version run side by side in workspaces, so each one's champion gets
    # an alias of its own instead (candidate-v2-a); promote gives the adopted one the version's alias.
    workspace = bool(cand and not cand["in_place"])
    alias = candidate_alias(cand["experiment"]) if workspace else version_alias(ver)
    client.set_registered_model_alias(name, alias, version)
    client.set_model_version_tag(name, version, "version", ver)
    if cand:   # a candidate champion is registered but never takes the alias until its experiment is promoted
        if not reused:   # a version that already existed (the same model) keeps its own tags
            client.set_model_version_tag(name, version, "candidate", "true")
            client.set_model_version_tag(name, version, "experiment", cand["experiment"])
        print(f"MLflow: candidate champion registered as {name} version {version} (alias {alias}) for "
              f"{cand['experiment']}; the alias 'champion' is unchanged until the version is promoted.")
        return
    client.set_registered_model_alias(name, "champion", version)
    print(f"MLflow: the champion is {name} version {version}, aliases 'champion' and '{version_alias(ver)}'.")


def _log_training(L, spec, root):
    tr = spec.get("training") or {}
    champ = tr.get("champion") or {}
    label = tr.get("primary_label")
    L.params({**{f"settings.{k}": v for k, v in (tr.get("settings") or {}).items()},
              "data_version": tr.get("data_version"), "primary_metric": label, "champion": champ.get("name"),
              "champion.type": champ.get("type"), "champion.class_weight": champ.get("class_weight"),
              **flat(champ.get("params") or {}, "champion.params.")})
    L.metrics({"champion_" + label: champ.get(label), "champion_std": champ.get("std"),
               "champion_vs_reference_pct": champ.get("vs_reference_pct"),
               "train_score": (tr.get("checks") or {}).get("train_score"),
               "cv_score": (tr.get("checks") or {}).get("cv_score"),
               "v1_met": {"met": 1, "not_met": 0}.get((champ.get("v1") or {}).get("status"))})
    L.tags({"data_version": tr.get("data_version"), "v1_bar_met": champ.get("v1_bar_met"),
            "v1_status": (champ.get("v1") or {}).get("status"),
            "beats_best_baseline": champ.get("beats_best_baseline")})
    best, runs = _best_trials(spec, root)
    for m in tr.get("models") or []:
        name = m.get("name")
        row, trials = best.get(name, (None, None))
        C = L.child(name, {"stage": "training", "kind": "champion" if name == champ.get("name") else "ml",
                           "model_type": m.get("type")})
        if not C:
            continue
        C.params({"type": m.get("type"), "class_weight": m.get("class_weight"), "inputs": m.get("inputs"),
                  **flat(m.get("params") or {}, "fixed."), **flat(m.get("tuning") or {}, "tuning.")})
        C.tags({"reason": m.get("reason")})
        if row is not None:
            try:
                C.params(flat(json.loads(row["params"] or "{}"), "best."))
                C.metrics(json.loads(row["metrics"] or "{}"))
                C.series("fold_" + label, json.loads(row["folds"] or "[]"))
            except (TypeError, ValueError):
                pass
            C.metrics({"primary_mean": row["primary_mean"], "primary_std": row["primary_std"],
                       "trials": len(trials), "minutes": float(trials["seconds"].sum()) / 60})
            C.series("trial_" + label, list(trials.sort_values("trial")["primary_mean"]))
            C.tags({"run_log_id": row["run_id"]})
        if name == champ.get("name"):
            C._try("the champion model", lambda: _register_champion(C, spec, root))
        C.end()
    L.table(tr.get("leaderboard") or [], "leaderboard")
    L.table(tr.get("top_features") or [], "top_features")


def _log_evaluation(L, spec, root):
    ev = spec.get("evaluation") or {}
    label = ev.get("primary_label")
    L.params({"champion": ev.get("champion"), "primary_metric": label, "operating_point": ev.get("operating_point"),
              "bootstrap.n": (ev.get("bootstrap") or {}).get("n"), "bootstrap.cluster": ev.get("cluster"),
              **flat({k: v for k, v in (ev.get("slice_settings") or {}).items()}, "slices.")})
    L.tags({"test_opened": "true", "times_test_opened": len(ev.get("test_opened") or []),
            "v1_status": (ev.get("v1") or {}).get("status")})
    L.metrics({"v1_met": {"met": 1, "not_met": 0}.get((ev.get("v1") or {}).get("status"))})
    rows = {r["model"]: r for r in ev.get("metrics") or []}
    c = rows.get(ev.get("champion")) or {}
    L.metrics({"test_" + label: c.get("test"), "test_low": c.get("low"), "test_high": c.get("high"),
               "cv_" + label: c.get("cv"), "score_psi": ev.get("score_psi"), "target_psi": ev.get("target_psi"),
               **{f"calibration_{k}": v for k, v in (ev.get("calibration") or {}).items()},
               "n_slices_worse_than_reference": sum(1 for s in ev.get("slices") or [] if s.get("worse than reference"))})
    for comp in ev.get("comparisons") or []:
        other = str(comp.get("comparison", "")).split(" against ")[-1]
        L.metrics({f"vs_{other}_{k}": comp.get(k) for k in ("difference", "low", "high", "relative %")})
    for name, r in rows.items():
        C = L.child(name, {"stage": "evaluation", "kind": r.get("kind")})
        if C:
            C.metrics({k: v for k, v in r.items() if k not in ("model", "kind")})
            C.end()
    L.table(ev.get("metrics") or [], "test_metrics")
    L.table(ev.get("comparisons") or [], "comparisons")
    L.table(ev.get("slices") or [], "slices")
    L.table(ev.get("evidence") or [], "evidence")
    L.file((ev.get("files") or {}).get("model_card"), "docs")


def _log_review(L, spec, root):
    rv = spec.get("review") or {}
    L.params({"stages": rv.get("stages")})
    L.metrics({**{f"open_{k}": v for k, v in (rv.get("open") or {}).items()}, "accepted": rv.get("accepted"),
               "fixed_since_last": rv.get("fixed_since_last"),
               "reviewer_findings": (rv.get("reviewer") or {}).get("findings")})
    L.table(rv.get("open_blockers") or [], "open_blockers")
    L.parquet((rv.get("files") or {}).get("findings"), "findings")


# ---------------------------------------------------------------- what each A/B testing stage logs

def leaves(d, prefix=""):
    """The numbers in a nested dict as flat metrics: a pair of numbers (an interval) becomes _low and _high, and
    true or false becomes 1 or 0. Text is left out."""
    out = {}
    if isinstance(d, bool):
        return {prefix.rstrip("."): float(d)} if prefix else {}
    if isinstance(d, (int, float)):
        return {prefix.rstrip("."): d} if prefix else {}
    if isinstance(d, (list, tuple)):
        if len(d) == 2 and all(_num(x) is not None for x in d) and not any(isinstance(x, bool) for x in d):
            return {prefix + "low": d[0], prefix + "high": d[1]}
        return {}
    if isinstance(d, dict):
        for k, v in d.items():
            out.update(leaves(v, f"{prefix}{k}."))
    return {k.rstrip("."): v for k, v in out.items()}


def _texts(d, keys):
    return {k: d.get(k) for k in keys if isinstance(d.get(k), (str, int, float)) and d.get(k) is not None}


def _metric_brief(m):
    return f"{m.get('name')} ({m.get('type')}, {m.get('direction')})" + (f", max harm {m['max_harm']}" if m.get("max_harm") else "")


def _log_ab_framing(L, spec, root):
    fr = spec.get("framing") or {}
    L.params({**_texts(fr, ("status", "test_name", "test_state", "unit", "owner", "mde")),
              "primary_metric": _metric_brief(fr.get("primary_metric") or {}),
              "secondary_metrics": [_metric_brief(m) for m in fr.get("secondary_metrics") or []],
              "guardrail_metrics": [_metric_brief(m) for m in fr.get("guardrail_metrics") or []],
              "variants": [f"{v.get('name')} ({v.get('role')}, {v.get('planned_share')})" for v in fr.get("variants") or []]})
    L.metrics({"n_variants": len(fr.get("variants") or []), "n_secondary": len(fr.get("secondary_metrics") or []),
               "n_guardrails": len(fr.get("guardrail_metrics") or []),
               **{f"planned_share.{v.get('name')}": v.get("planned_share") for v in fr.get("variants") or []}})
    # Logged before any result is seen: the run time is the record that the plan came first
    L.tags({**{k: fr.get(k) for k in ("hypothesis", "change", "decision", "ship_rule", "population")},
            "mlflow.note.content": fr.get("hypothesis"), "preregistered": "true" if fr.get("status") == "confirmed" else "false"})
    L.table([{"role": role, **m} for role, ms in (("primary", [fr.get("primary_metric") or {}]),
                                                   ("secondary", fr.get("secondary_metrics") or []),
                                                   ("guardrail", fr.get("guardrail_metrics") or [])) for m in ms], "metrics")
    L.table(fr.get("risks") or [], "risks")
    L.file(os.path.join(ml_common.project_paths(spec, root)["docs"], "test_plan.md"), "docs")


def _log_ab_design(L, spec, root):
    de = spec.get("design") or {}
    L.params(_texts(de, ("status", "alpha", "power", "sides", "looks", "variance_reduction")))
    L.metrics(leaves({k: v for k, v in de.items() if k not in ("alpha", "power")}))


def _log_ab_validation(L, spec, root):
    va, da = spec.get("validation") or {}, spec.get("data") or {}
    L.params({**_texts(da, ("source", "unit_col", "group_col", "date_col")),
              **flat(da.get("variant_values") or {}, "variant_values."), **flat(da.get("metric_cols") or {}, "metric_cols."),
              "pre_treatment_cols": da.get("pre_treatment_cols"), "post_treatment_cols": da.get("post_treatment_cols"),
              "result": va.get("result")})
    L.metrics({"n_failures": len(va.get("failures") or []), "n_warnings": len(va.get("warnings") or []),
               **leaves(va.get("srm") or {}, "srm."), **leaves(va.get("units") or {}, "units.")})
    L.tags({"result": va.get("result"), "override": (va.get("override") or {}).get("reason")})
    L.table([{"kind": "failure", "text": t} for t in va.get("failures") or []]
            + [{"kind": "warning", "text": t} for t in va.get("warnings") or []], "problems")
    bal = va.get("balance")
    L.table(bal if isinstance(bal, list) else [{"column": k, **(v if isinstance(v, dict) else {"value": v})}
                                               for k, v in (bal or {}).items()], "balance")


def _comparisons(an):
    p = an.get("primary") or {}
    prim = p.get("all_treatments") or ([p] if p else [])
    return ([("primary", r) for r in prim] + [("secondary", r) for r in an.get("secondary") or []]
            + [("guardrail", r) for r in an.get("guardrails") or []])


def _log_ab_analysis(L, spec, root):
    an = spec.get("analysis") or {}
    p = an.get("primary") or {}
    L.params({**_texts(an, ("alpha", "sides", "mde_abs", "cuped_col")), "primary_metric": p.get("metric"),
              "method": p.get("method")})
    L.metrics({**_texts(an, ("units_analysed", "units_dropped_crossed")),
               **{f"primary.{k}": v for k, v in leaves({k: v for k, v in p.items() if k != "all_treatments"}).items()},
               "n_guardrails_breached": sum(1 for g in an.get("guardrails") or [] if g.get("breached"))})
    L.tags({"result_label": p.get("result_label"), "significant": p.get("significant")})
    for role, r in _comparisons(an):
        C = L.child(f"{role}: {r.get('metric')} ({r.get('treatment')})",
                    {"stage": "analysis", "role": role, "metric": r.get("metric"), "treatment": r.get("treatment"),
                     "result_label": r.get("result_label"), "breached": r.get("breached")})
        if C:
            C.params({"method": r.get("method")})
            C.metrics(leaves(r))
            C.end()
    L.parquet((an.get("files") or {}).get("results"), "results")
    L.table(an.get("checks") or [], "checks")


def _log_ab_segments(L, spec, root):
    se = spec.get("segments") or {}
    res = se.get("results") or []
    L.params({**_texts(se, ("status", "metric")), "post_treatment_ok": se.get("post_treatment_ok"),
              "columns": [r.get("column") for r in res if isinstance(r, dict)]})
    for r in res if isinstance(res, list) else []:
        col = r.get("column")
        L.metrics({f"{col}.{k}": v for k, v in leaves({k: v for k, v in r.items() if k not in ("segments", "effects")}).items()})
        rows = r.get("segments") or r.get("effects")
        if isinstance(rows, list):
            L.table(rows, f"segments_{col}")


def _log_ab_readout(L, spec, root):
    ro = spec.get("readout") or {}
    L.params(_texts(ro, ("status", "recommendation", "result_label")))
    L.tags({"recommendation": ro.get("recommendation"), "recommendation_reason": ro.get("recommendation_reason"),
            "result_label": ro.get("result_label"), "mlflow.note.content": ro.get("recommendation_reason")})
    bi = ro.get("business_impact")
    L.tags({"business_impact": bi if isinstance(bi, str) else None, "ship_rule_met": ro.get("ship_rule_met")})
    prim = {k: v for k, v in ((spec.get("analysis") or {}).get("primary") or {}).items() if k != "all_treatments"}
    L.metrics({**(leaves(bi, "business_impact.") if isinstance(bi, dict) else {}),
               "ship_rule_met": ro.get("ship_rule_met"),
               **{f"primary.{k}": v for k, v in leaves(prim).items()}})   # the result the readout rests on
    L.table([{"kind": "risk", "text": t} for t in ro.get("risks") or []]
            + [{"kind": "next step", "text": t} for t in ro.get("next_steps") or []], "risks_and_next_steps")
    L.file(os.path.join(ml_common.project_paths(spec, root)["docs"], "ab_readout.md"), "docs")


AB_LOGGERS = {"framing": _log_ab_framing, "design": _log_ab_design, "validation": _log_ab_validation,
              "analysis": _log_ab_analysis, "segments": _log_ab_segments, "readout": _log_ab_readout}

LOGGERS = {"split": _log_split, "features": _log_features, "baseline": _log_baseline, "training": _log_training,
           "evaluation": _log_evaluation, "review": _log_review}
if KIND == "ab":
    LOGGERS = AB_LOGGERS


# ---------------------------------------------------------------- one run per stage notebook

def candidate(spec=None, root=None):
    """The version experiment this run belongs to, or None.

    Two sources. A workspace run (an agent working on a copy of the project) sets ML_ITERATE_EXPERIMENT and the
    parent run in the environment, or ml-autopilot's workspace.py writes them to .ml_workspace.json at the
    workspace root (each shell command of an agent starts a fresh shell, so the file is the reliable source): its
    runs never become current. An iteration opened by ml-iterate, in the project itself or in its own worktree
    (--worktree), is read from the spec (the open iteration's version and parent run), because each Claude Code
    command starts a fresh shell: its runs become current, so MLflow keeps matching the spec, but stay candidates
    until the iteration closes, and the alias 'champion' only moves when the version is promoted."""
    exp = os.environ.get("ML_ITERATE_EXPERIMENT")
    if exp:
        return {"experiment": exp, "iteration": os.environ.get("ML_ITERATE_ITERATION"),
                "parent": os.environ.get("ML_ITERATE_PARENT_RUN"),
                "agent_session": os.environ.get("ML_AGENT_SESSION"), "in_place": False}
    ws = ml_common.experiment_workspace(root) if root else None
    if ws and ws.get("experiment"):
        return {"experiment": ws["experiment"], "iteration": ws.get("iteration"), "parent": ws.get("parent_run"),
                "agent_session": os.environ.get("ML_AGENT_SESSION") or ws.get("agent_session"), "in_place": False}
    it = (spec or {}).get("iterations") or {}
    e = next((x for x in it.get("entries") or [] if x.get("id") == it.get("current") and x.get("status") == "open"),
             None)
    if not e:
        return None
    return {"experiment": e.get("version") or f"V{int(e['id']) + 1}", "iteration": e.get("id"),
            "parent": e.get("mlflow_parent_run"), "agent_session": os.environ.get("ML_AGENT_SESSION"),
            "in_place": True}


def version_alias(version):
    """The alias each version's champion keeps: champion-v1, champion-v2, ... MLflow reserves v1, v2 for itself."""
    return f"champion-{str(version).lower()}"


def candidate_alias(experiment):
    """The alias of the champion an experiment in a workspace registered: candidate-v2-a."""
    return "candidate-" + re.sub(r"[^a-z0-9-]+", "-", str(experiment).lower()).strip("-")


def current_version(spec):
    """The version runs are logged for: V1 until ml-iterate opens V2, and so on."""
    return str(((spec or {}).get("version") or {}).get("current") or "V1")


def _git_branch(root):
    try:
        import git_ops
        return git_ops.branch(root) if git_ops.is_repo(root) else None
    except ImportError:
        return None


def _demote(client, exp_id, stage, keep_run_id):
    """Clear the current tag on the other stage runs of a stage. Child runs (models, comparisons) are left alone:
    stage runs carry run_type=stage, and stage runs logged before that tag existed have no parent run."""
    for old in client.search_runs([exp_id], f"tags.stage = '{stage}' and tags.current = 'true'", max_results=500):
        t = old.data.tags
        is_stage_run = t.get("run_type") == "stage" or "mlflow.parentRunId" not in t
        if old.info.run_id != keep_run_id and is_stage_run:
            client.set_tag(old.info.run_id, "current", "false")


def log_stage(spec_path, stage, notebook, skill):
    """Log one run of a stage notebook, from what the notebook just wrote to the spec, plus its charts. Marks it
    as the current run of the stage and records its id in spec[stage]['mlflow']. Returns that record, or None."""
    spec_path = os.path.abspath(spec_path)
    root = os.path.dirname(spec_path)
    spec = ml_common.migrate(ml_common.load(spec_path))
    if stage not in LOGGERS:
        return None
    mlflow, uri = ready(spec)
    if mlflow is None:
        _warn(uri)
        return None
    try:
        name, exp_id = ensure_experiment(spec, root, mlflow, uri)
        if (spec.get("tracking") or {}).get("experiment_id") != str(exp_id):
            write_tracking_section(spec_path, uri, name, exp_id, skill)
            spec = ml_common.migrate(ml_common.load(spec_path))
        from mlflow.tracking import MlflowClient
        client = MlflowClient()
        section = spec.get(stage) or {}
        tags = {"project": (spec.get("project") or {}).get("name"), "project_root": root, "stage": stage,
                "skill": skill, "notebook": ml_common.relpath(notebook, root), "run_type": "stage",
                "current": "false" if (candidate(spec, root) and not candidate(spec, root)["in_place"]) else "true",
                "version": current_version(spec), "git_branch": _git_branch(root),
                "agent_session": os.environ.get("ML_AGENT_SESSION"),
                "data_version": (spec.get("training") or {}).get("data_version") if KIND == "ml" else None,
                "workflow": KIND,
                "spec_hash": hashlib.sha256(json.dumps(section, sort_keys=True, default=str).encode()).hexdigest()[:12],
                "mlflow.source.name": ml_common.relpath(notebook, root), "mlflow.source.type": "NOTEBOOK"}
        import logging
        import warnings
        warnings.filterwarnings("ignore", module=r"mlflow\..*")
        logging.getLogger("mlflow").setLevel(logging.ERROR)
        cand = candidate(spec, root)
        if cand:
            tags.update({"candidate": "true", "experiment": cand["experiment"], "iteration": cand["iteration"],
                         "agent_session": cand["agent_session"], "mlflow.parentRunId": cand["parent"]})
        run = client.create_run(exp_id, run_name=stage, tags={k: str(v) for k, v in tags.items() if v is not None})
        L = Logger(mlflow, run.info.run_id, root)
        status = "FINISHED"
        try:
            LOGGERS[stage](L, spec, root)
            for p in list(_FIGURES):
                L.file(p, "charts")
            _FIGURES.clear()
            with tempfile.TemporaryDirectory() as d:
                p = os.path.join(d, f"spec_{stage}.yaml")
                with open(p, "w") as f:
                    f.write(ml_common.dump_text({stage: section}))
                L.file(p, "spec")
        except Exception as e:  # noqa: BLE001
            status = "FAILED"
            _warn(f"logging the {stage} run stopped: {type(e).__name__}: {str(e)[:200]}")
        L.end(status)
        if not cand or cand["in_place"]:   # a workspace run never displaces the project's current run
            _demote(client, exp_id, stage, run.info.run_id)
        ref = {"run_id": run.info.run_id, "experiment_id": str(exp_id), "url": ui_link(uri, exp_id, run.info.run_id),
               "logged_at": _dt.datetime.now().replace(microsecond=0).isoformat()}
        s = ml_common.load(spec_path)
        if isinstance(s.get(stage), dict):
            s[stage]["mlflow"] = ref
            ml_common.save(spec_path, s)
        print(f"MLflow: logged the {stage} run to {name}" + (f": {ref['url']}" if ref["url"] else "."))
        return ref
    except Exception as e:  # noqa: BLE001
        _warn(f"could not log the {stage} run: {type(e).__name__}: {str(e)[:300]}")
        return None


def attach_build(spec_path, stage, answers_path=None, git_info=None):
    """After a stage build: put the kept answers file and the git commit on the stage's run, so the run can be
    rebuilt from MLflow alone and traced to exact code."""
    try:
        spec = ml_common.load(spec_path)
        ref = (spec.get(stage) or {}).get("mlflow") or {}
        if not ref.get("run_id"):
            return
        mlflow, uri = ready(spec)
        if mlflow is None:
            return
        from mlflow.tracking import MlflowClient
        client = MlflowClient()
        if answers_path and os.path.exists(answers_path):
            client.log_artifact(ref["run_id"], answers_path, "answers")
        g = git_info or {}
        for k, v in {"mlflow.source.git.commit": g.get("commit"), "git_commit": g.get("commit"),
                     "git_branch": g.get("branch")}.items():
            if v:
                client.set_tag(ref["run_id"], k, str(v))
    except Exception as e:  # noqa: BLE001  the build succeeded; this only adds links to it
        _warn(f"could not add the answers or the commit to the {stage} run: {type(e).__name__}: {str(e)[:200]}")


def restore_current(spec_path):
    """Make the runs the spec names (spec[stage]['mlflow']) the current run of each stage again, for example after
    ml-iterate rolled a version back to its snapshot. The alias 'champion' is not touched."""
    client, exp_id, spec = _client_for(spec_path)
    if client is None:
        return None
    done = []
    for stage in LOGGED_STAGES:
        rid = ((spec.get(stage) or {}).get("mlflow") or {}).get("run_id")
        if not rid:
            continue
        try:
            _demote(client, exp_id, stage, rid)
            client.set_tag(rid, "current", "true")
            done.append(stage)
        except Exception as e:  # noqa: BLE001
            _warn(f"could not make the {stage} run current again: {e}")
    return done


# ---------------------------------------------------------------- Claude Code tracing

def setup_claude_tracing(root, uri, experiment_id, hook_command=None):
    """Trace Claude Code sessions in this project into the project's experiment, with MLflow's Claude Code plugin
    (`mlflow autolog claude`, which writes .claude/settings.json and .claude/settings.local.json). With
    hook_command, also add a SessionStart hook that puts the session id in ML_AGENT_SESSION for every command, so
    the runs a session logs carry its id. Returns (ok, message)."""
    import shutil
    import subprocess
    cmd = None
    for cand in (os.path.expanduser("~/.local/bin/mlflow"), shutil.which("mlflow")):
        # Skip an mlflow inside a virtual environment (a project's .venv): that is the client, not the tool.
        in_venv = cand and os.path.exists(os.path.join(os.path.dirname(os.path.dirname(cand)), "pyvenv.cfg"))
        if cand and os.path.exists(cand) and (cand == os.path.expanduser("~/.local/bin/mlflow") or not in_venv):
            cmd = [cand]
            break
    if cmd is None and shutil.which("uv"):
        cmd = [shutil.which("uv"), "tool", "run", "mlflow"]
    if cmd is None:
        return False, "the mlflow command is not installed (uv tool install mlflow)"
    env = {**os.environ, "MLFLOW_DISABLE_AGENT_HINT": "1", "MLFLOW_DISABLE_TELEMETRY": "true"}
    r = subprocess.run(cmd + ["autolog", "claude", "-d", root, "-u", uri, "-e", str(experiment_id)],
                       capture_output=True, text=True, env=env, timeout=300)
    if r.returncode != 0:
        return False, "mlflow autolog claude failed: " + (r.stderr or r.stdout).strip()[-300:]
    if hook_command:
        path = os.path.join(root, ".claude", "settings.json")
        try:
            with open(path, encoding="utf-8") as f:
                settings = json.load(f)
        except (OSError, ValueError):
            settings = {}
        starts = settings.setdefault("hooks", {}).setdefault("SessionStart", [])
        present = any(h.get("command") == hook_command for m in starts for h in m.get("hooks") or [])
        if not present:
            starts.append({"hooks": [{"type": "command", "command": hook_command}]})
        with open(path, "w", encoding="utf-8") as f:
            json.dump(settings, f, indent=2)
            f.write("\n")
    return True, f"Claude Code sessions in this project are traced to experiment {experiment_id} on {uri}"


def attach_notebook(notebook_path, stage, spec_path, since):
    """Attach the executed notebook to the run its own last cell logged (called after the notebook ran)."""
    try:
        spec = ml_common.load(spec_path)
        ref = (spec.get(stage) or {}).get("mlflow") or {}
        if not ref.get("run_id") or (ref.get("logged_at") or "") < since:
            return
        mlflow, uri = ready(spec)
        if mlflow is None:
            return
        from mlflow.tracking import MlflowClient
        MlflowClient().log_artifact(ref["run_id"], notebook_path, "notebook")
        print(f"MLflow run: {ref.get('url') or ref['run_id']}")
    except Exception:  # noqa: BLE001  attaching the notebook is a convenience
        pass


# ---------------------------------------------------------------- for the status skills

def status_notes(spec, setup_hint):
    """(lines for Needs attention, a one line link to the experiment or None), for ml-status and ab-status."""
    tr = spec.get("tracking") or {}
    done = [s for s in LOGGED_STAGES if isinstance(spec.get(s), dict) and spec.get(s)
            and (spec.get(s) or {}).get("status") not in ("skipped", "not_started")]
    if not tr.get("experiment_id"):
        return ([f"MLflow tracking is not set up for this project, so its runs are not logged. {setup_hint}"]
                if done else []), None
    notes = []
    uri = tracking_uri(spec)
    if not reachable(uri):
        notes.append(f"The MLflow server at {uri} is not reachable, so new runs are not logged. Start it with "
                     "`python3 ~/Projects/claude-config/skills/tools/mlflow_server.py start`.")
    missing = [s for s in done if not (spec.get(s) or {}).get("mlflow")]
    if missing:
        notes.append("No MLflow run is logged for " + ", ".join(missing)
                     + ". A run is logged the next time each of those notebooks runs with the server up.")
    return notes, f"**MLflow:** experiment `{tr.get('experiment')}`" + (f", {tr['ui']}" if tr.get("ui") else "") + "."


# ---------------------------------------------------------------- command line

def setup(spec_path):
    spec = ml_common.migrate(ml_common.load(spec_path))
    mlflow, uri = ready(spec)
    if mlflow is None:
        print("MLflow: " + uri)
        return 1
    name, exp_id = ensure_experiment(spec, os.path.dirname(spec_path), mlflow, uri)
    write_tracking_section(spec_path, uri, name, exp_id)
    print(f"Tracking: experiment {name} (id {exp_id}) on {uri}")
    link = ui_link(uri, exp_id)
    if link:
        print(f"Open: {link}")
    return 0


def status(spec_path):
    spec = ml_common.migrate(ml_common.load(spec_path))
    uri = tracking_uri(spec)
    tr = spec.get("tracking") or {}
    print(f"Address: {uri} ({'reachable' if reachable(uri) else 'not reachable'})")
    print(f"Experiment: {tr.get('experiment') or 'not set up'}" + (f" (id {tr['experiment_id']})" if tr.get("experiment_id") else ""))
    for stage in LOGGED_STAGES:
        ref = (spec.get(stage) or {}).get("mlflow")
        if spec.get(stage):
            print(f"  {stage}: " + (f"run {ref['run_id']} at {ref.get('logged_at')}" if ref else "no run logged"))
    return 0


def export(spec_path, folder=None):
    spec = ml_common.migrate(ml_common.load(spec_path))
    mlflow, uri = ready(spec)
    if mlflow is None:
        print("MLflow: " + uri)
        return 1
    from mlflow.tracking import MlflowClient
    import pandas as pd
    client = MlflowClient()
    exp_id = (spec.get("tracking") or {}).get("experiment_id")
    if not exp_id:
        print("This project has no experiment yet. Run `tracking.py setup` first.")
        return 1
    root = os.path.dirname(os.path.abspath(spec_path))
    out = os.path.abspath(folder or os.path.join(root, "mlflow_export"))
    os.makedirs(out, exist_ok=True)
    runs = mlflow.search_runs([exp_id], max_results=50000)
    runs.to_parquet(os.path.join(out, "runs.parquet"), index=False)
    for rid in runs["run_id"]:
        d = os.path.join(out, "artifacts", rid)
        os.makedirs(d, exist_ok=True)
        try:
            mlflow.artifacts.download_artifacts(run_id=rid, dst_path=d)
        except Exception as e:  # noqa: BLE001
            print(f"Could not download the artifacts of run {rid}: {e}")
        history = []
        for k in client.get_run(rid).data.metrics:
            history += [{"run_id": rid, "key": k, "value": m.value, "step": m.step, "timestamp": m.timestamp}
                        for m in client.get_metric_history(rid, k)]
        if history:
            pd.DataFrame(history).to_parquet(os.path.join(d, "metric_history.parquet"), index=False)
    print(f"Exported {len(runs)} runs of experiment {exp_id} to {ml_common.relpath(out, root)}")
    return 0


# ---------------------------------------------------------------- ml-iterate experiments

def _client_for(spec_path):
    spec = ml_common.migrate(ml_common.load(spec_path))
    mlflow, uri = ready(spec)
    if mlflow is None:
        _warn(uri)
        return None, None, spec
    root = os.path.dirname(os.path.abspath(spec_path))
    _, exp_id = ensure_experiment(spec, root, mlflow, uri)
    from mlflow.tracking import MlflowClient
    return MlflowClient(), exp_id, spec


def _parent_run(client, exp_id, experiment):
    runs = client.search_runs([exp_id], f"tags.run_type = 'experiment' and tags.experiment = '{experiment}'",
                              max_results=1)
    return runs[0] if runs else None


def start_experiment(spec_path, experiment, hypothesis=None, tags=None):
    """Create the parent run of an ml-iterate experiment and return its id (or None when MLflow is down).
    ml-iterate passes it to the stage notebooks in ML_ITERATE_PARENT_RUN."""
    client, exp_id, spec = _client_for(spec_path)
    if client is None:
        return None
    old = _parent_run(client, exp_id, experiment)
    if old:
        return old.info.run_id
    t = {"run_type": "experiment", "experiment": experiment, "candidate": "true", "version": experiment,
         "project": (spec.get("project") or {}).get("name"), "workflow": KIND,
         "iteration": os.environ.get("ML_ITERATE_ITERATION"), "agent_session": os.environ.get("ML_AGENT_SESSION"),
         "mlflow.note.content": hypothesis, **(tags or {})}
    run = client.create_run(exp_id, run_name=f"experiment {experiment}",
                            tags={k: str(v) for k, v in t.items() if v is not None})
    return run.info.run_id


def promote(spec_path, experiment):
    """Make an experiment's stage runs the current run of each stage, and move the alias 'champion' to the
    champion it registered. Records only: the spec, written when the experiment was promoted, decides."""
    client, exp_id, spec = _client_for(spec_path)
    if client is None:
        return None
    runs = client.search_runs([exp_id], f"tags.experiment = '{experiment}' and tags.run_type = 'stage'",
                              order_by=["attributes.start_time DESC"], max_results=500)
    newest = {}
    for r in runs:
        newest.setdefault(r.data.tags.get("stage"), r)
    for stage, r in newest.items():
        _demote(client, exp_id, stage, r.info.run_id)
        client.set_tag(r.info.run_id, "current", "true")
        client.set_tag(r.info.run_id, "candidate", "promoted")
    moved = None
    name = (spec.get("tracking") or {}).get("registered_model") or registered_model_name(spec)
    version = None
    train_run = newest.get("training")
    if train_run:   # the champion child run of the experiment's training run records the version it registered
        kids = client.search_runs([exp_id], f"tags.mlflow.parentRunId = '{train_run.info.run_id}' and tags.kind = 'champion'",
                                  max_results=5)
        version = next((k.data.tags.get("registered_version") for k in kids if k.data.tags.get("registered_version")), None)
    if version is None:
        try:
            tagged = [v for v in client.search_model_versions(f"name='{name}'") if v.tags.get("experiment") == experiment]
        except Exception:  # noqa: BLE001  no registered model yet
            tagged = []
        version = max(tagged, key=lambda v: int(v.version)).version if tagged else None
    if version is not None:
        client.set_registered_model_alias(name, "champion", version)
        ver_tag = client.get_model_version(name, version).tags.get("version")
        if ver_tag:   # the adopted experiment's champion becomes its version's champion (champion-v2)
            client.set_registered_model_alias(name, version_alias(ver_tag), version)
        mv = client.get_model_version(name, version)
        if mv.tags.get("candidate") == "true":
            client.set_model_version_tag(name, version, "candidate", "promoted")
        moved = version
    parent = _parent_run(client, exp_id, experiment)
    if parent:
        client.set_tag(parent.info.run_id, "verdict", "promoted")
        client.set_tag(parent.info.run_id, "candidate", "promoted")
    return {"stages": sorted(s for s in newest if s), "champion_version": moved}


def reject(spec_path, experiment, reason=None):
    """Mark an experiment as rejected. Its runs stay for the record; nothing current changes."""
    client, exp_id, _ = _client_for(spec_path)
    if client is None:
        return None
    parent = _parent_run(client, exp_id, experiment)
    if not parent:
        return False
    client.set_tag(parent.info.run_id, "verdict", "rejected")
    if reason:
        client.set_tag(parent.info.run_id, "verdict_reason", str(reason)[:5000])
    return True


def main():
    args = sys.argv[1:]
    cmd = args[0] if args else "status"
    spec_path = ml_common.spec_path()
    if not os.path.exists(spec_path):
        sys.exit(f"No {ml_common.SPEC_NAME} found. Run this from the project root.")
    if cmd == "setup":
        sys.exit(setup(spec_path))
    if cmd == "status":
        sys.exit(status(spec_path))
    if cmd == "export":
        sys.exit(export(spec_path, args[1] if len(args) > 1 else None))
    if cmd == "tracing":
        spec = ml_common.migrate(ml_common.load(spec_path))
        tr = spec.get("tracking") or {}
        if not tr.get("experiment_id"):
            sys.exit("Set up tracking first: tracking.py setup")
        ok, msg = setup_claude_tracing(os.path.dirname(spec_path), tracking_uri(spec), tr["experiment_id"],
                                       args[1] if len(args) > 1 else None)
        print(msg)
        sys.exit(0 if ok else 1)
    if cmd == "restore-current":
        print(json.dumps(restore_current(spec_path)))
        sys.exit(0)
    if cmd in ("start-experiment", "promote", "reject"):
        if len(args) < 2:
            sys.exit(f"tracking.py {cmd} needs the experiment name.")
        if cmd == "start-experiment":
            rid = start_experiment(spec_path, args[1], args[2] if len(args) > 2 else None)
            print(rid or "")
            sys.exit(0 if rid else 1)
        if cmd == "promote":
            res = promote(spec_path, args[1])
            print(json.dumps(res))
            sys.exit(0 if res is not None else 1)
        res = reject(spec_path, args[1], args[2] if len(args) > 2 else None)
        print(json.dumps(res))
        sys.exit(0 if res else 1)
    sys.exit(__doc__)


if __name__ == "__main__":
    main()
