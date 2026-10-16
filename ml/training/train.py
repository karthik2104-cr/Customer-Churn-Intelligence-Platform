"""Compare classifiers and save one deployed pipeline.

Split policy:
- 20% stratified holdout is touched only after the model, calibration choice, and threshold are frozen.
- The remaining rows are split into 75% train and 25% internal validation.
- Preprocessing is inside each pipeline and fit only on the rows that pipeline sees.
- Model choice uses cross-validated average precision on the training split.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, precision_recall_curve, roc_curve
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold, cross_val_score, train_test_split
from sklearn.base import clone
from sklearn.tree import DecisionTreeClassifier

from ml.data.eda import build_eda, render_eda_markdown, write_charts
from ml.data.loader import clean_customers, dataset_label_for_path, filter_trainable, read_raw_csv
from ml.data.quality import build_quality_report
from ml.data.schema import (
    CATEGORICAL_FEATURES,
    DATASET_MIRROR_NOTE,
    DATASET_SOURCE_URL,
    EXCLUSION_REASON,
    ID_COLUMN,
    MODEL_FEATURES,
    NUMERIC_FEATURES,
    PROTECTED_COLUMNS,
    TARGET_COLUMN,
)
from ml.evaluation.associations import training_associations
from ml.evaluation.importance import grouped_permutation_importance
from ml.evaluation.metrics import binary_metrics, choose_medium_threshold, tune_cost_threshold
from ml.evaluation.serialize import to_jsonable
from ml.features.preprocessing import make_model_pipeline
from ml.training.estimators import BalancedHistGradientBoosting, RandomOversampleClassifier

ROOT = Path(__file__).resolve().parents[2]

SCORING = {
    "average_precision": "average_precision",
    "roc_auc": "roc_auc",
    "recall": "recall",
    "precision": "precision",
    "f1": "f1",
    "accuracy": "accuracy",
}


@dataclass
class Spec:
    name: str
    estimator: object
    scale: bool
    params: dict
    n_iter: int | None


def _versions() -> dict:
    import matplotlib
    import pandas as pd_module
    import sklearn

    versions = {
        "numpy": np.__version__,
        "pandas": pd_module.__version__,
        "scikit-learn": sklearn.__version__,
        "matplotlib": matplotlib.__version__,
    }
    for module_name in ("xgboost", "shap", "plotly"):
        try:
            module = __import__(module_name)
            versions[module_name] = getattr(module, "__version__", "installed")
        except Exception:
            versions[module_name] = None
    return versions


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 16), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _downsample(values_x, values_y, limit: int = 40) -> dict:
    x_values = np.asarray(values_x, dtype=float)
    y_values = np.asarray(values_y, dtype=float)
    if len(x_values) > limit:
        index = np.unique(np.linspace(0, len(x_values) - 1, limit).astype(int))
    else:
        index = np.arange(len(x_values))
    return {
        "x": [float(value) for value in x_values[index]],
        "y": [float(value) for value in y_values[index]],
    }


def _curves(y_true, probabilities) -> dict:
    fpr, tpr, _ = roc_curve(y_true, probabilities)
    precision, recall, _ = precision_recall_curve(y_true, probabilities)
    try:
        fraction, mean_predicted = calibration_curve(y_true, probabilities, n_bins=8, strategy="quantile")
    except ValueError:
        fraction, mean_predicted = calibration_curve(y_true, probabilities, n_bins=5, strategy="uniform")
    roc = _downsample(fpr, tpr)
    pr = _downsample(recall, precision)
    return {
        "roc_curve": {"fpr": roc["x"], "tpr": roc["y"]},
        "pr_curve": {"recall": pr["x"], "precision": pr["y"]},
        "calibration_curve": {
            "mean_predicted": [float(value) for value in mean_predicted],
            "fraction_positive": [float(value) for value in fraction],
        },
    }


def _latency(pipeline, frame: pd.DataFrame) -> dict:
    sample = frame.head(min(400, len(frame)))
    pipeline.predict_proba(sample.head(min(5, len(sample))))
    repeats = 5
    started = time.perf_counter()
    for _ in range(repeats):
        pipeline.predict_proba(sample)
    elapsed = time.perf_counter() - started
    return {
        "rows": int(len(sample)),
        "repeats": repeats,
        "milliseconds_per_row": float(elapsed / (repeats * max(len(sample), 1)) * 1000),
    }


def _baselines_and_spec(frame: pd.DataFrame) -> tuple[dict, dict]:
    baselines: dict = {}
    numeric: dict = {}
    categorical: dict = {}
    for column in NUMERIC_FEATURES:
        values = pd.to_numeric(frame[column], errors="coerce")
        median = float(values.median())
        baselines[column] = median
        numeric[column] = {"min": float(values.min()), "max": float(values.max()), "median": median}
    for column in CATEGORICAL_FEATURES:
        mode = frame[column].mode(dropna=True)
        baselines[column] = None if mode.empty else str(mode.iloc[0])
        categorical[column] = sorted(str(value) for value in frame[column].dropna().unique().tolist())
    return baselines, {"numeric": numeric, "categorical": categorical}


def _combo_count(params: dict) -> int:
    total = 1
    for values in params.values():
        total *= max(len(list(values)), 1)
    return total


def _best_cv(search) -> dict:
    index = int(search.best_index_)
    results = search.cv_results_

    def grab(metric: str) -> tuple[float, float]:
        return (
            float(results[f"mean_test_{metric}"][index]),
            float(results[f"std_test_{metric}"][index]),
        )

    ap_mean, ap_std = grab("average_precision")
    roc_mean, roc_std = grab("roc_auc")
    recall_mean, recall_std = grab("recall")
    precision_mean, precision_std = grab("precision")
    f1_mean, f1_std = grab("f1")
    accuracy_mean, accuracy_std = grab("accuracy")
    return {
        "cv_average_precision_mean": ap_mean,
        "cv_average_precision_std": ap_std,
        "cv_roc_auc_mean": roc_mean,
        "cv_roc_auc_std": roc_std,
        "cv_recall_at_0_5_mean": recall_mean,
        "cv_recall_at_0_5_std": recall_std,
        "cv_precision_at_0_5_mean": precision_mean,
        "cv_precision_at_0_5_std": precision_std,
        "cv_f1_at_0_5_mean": f1_mean,
        "cv_f1_at_0_5_std": f1_std,
        "cv_accuracy_at_0_5_mean": accuracy_mean,
        "cv_accuracy_at_0_5_std": accuracy_std,
    }


def _grouped_param(cv_results: dict, key: str) -> dict | None:
    if key not in cv_results:
        return None
    grouped: dict[str, list[float]] = {}
    for value, score in zip(cv_results[key], cv_results["mean_test_average_precision"]):
        grouped.setdefault(str(value), []).append(float(score))
    return {
        name: {"mean_average_precision": float(np.mean(scores)), "configurations": len(scores)}
        for name, scores in grouped.items()
    }


def build_specs(y_train: pd.Series, seed: int, fast: bool) -> tuple[list[Spec], str | None]:
    positives = max(int((np.asarray(y_train) == 1).sum()), 1)
    negatives = int((np.asarray(y_train) == 0).sum())
    scale_pos_weight = float(negatives / positives)
    jobs = 1 if fast else -1
    specs = [
        Spec(
            name="logistic_regression",
            estimator=LogisticRegression(max_iter=1000, solver="liblinear", random_state=seed),
            scale=True,
            params={
                "model__C": [1.0] if fast else [0.2, 1.0, 3.0],
                "model__class_weight": [None, "balanced"],
            },
            n_iter=None,
        ),
        Spec(
            name="decision_tree",
            estimator=DecisionTreeClassifier(random_state=seed),
            scale=False,
            params={
                "model__max_depth": [3, 6] if fast else [4, 8, 12],
                "model__min_samples_leaf": [10] if fast else [10, 40],
                "model__class_weight": [None, "balanced"],
            },
            n_iter=None,
        ),
        Spec(
            name="random_forest",
            estimator=RandomForestClassifier(
                n_estimators=40 if fast else 180,
                random_state=seed,
                n_jobs=jobs,
            ),
            scale=False,
            params={
                "model__max_depth": [6] if fast else [8, 16, None],
                "model__min_samples_leaf": [5] if fast else [2, 8],
                "model__max_features": ["sqrt"],
                "model__class_weight": [None, "balanced_subsample"],
            },
            n_iter=2 if fast else 6,
        ),
        Spec(
            name="hist_gradient_boosting",
            estimator=BalancedHistGradientBoosting(
                max_iter=30 if fast else 180,
                random_state=seed,
            ),
            scale=False,
            params={
                "model__learning_rate": [0.1] if fast else [0.05, 0.1],
                "model__max_depth": [3] if fast else [3, 6],
                "model__min_samples_leaf": [10] if fast else [20],
                "model__class_weight": [None, "balanced"],
            },
            n_iter=2 if fast else 6,
        ),
    ]
    xgboost_note = None
    if fast:
        return specs, "XGBoost is skipped in fast mode."
    try:
        from xgboost import XGBClassifier
    except Exception as exc:
        return specs, f"XGBoost is unavailable ({type(exc).__name__}: {exc})."
    specs.append(
        Spec(
            name="xgboost",
            estimator=XGBClassifier(
                objective="binary:logistic",
                eval_metric="logloss",
                tree_method="hist",
                n_estimators=180,
                subsample=0.9,
                colsample_bytree=0.9,
                random_state=seed,
                n_jobs=jobs,
                verbosity=0,
            ),
            scale=False,
            params={
                "model__max_depth": [3, 5],
                "model__learning_rate": [0.05, 0.1],
                "model__min_child_weight": [1, 5],
                "model__scale_pos_weight": [1.0, scale_pos_weight],
            },
            n_iter=6,
        )
    )
    return specs, xgboost_note


def _fit_spec(spec: Spec, x_train: pd.DataFrame, y_train: pd.Series, seed: int, folds: int) -> tuple[object, dict]:
    pipeline = make_model_pipeline(spec.estimator, spec.scale)
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    started = time.perf_counter()
    if spec.n_iter is None or _combo_count(spec.params) <= spec.n_iter:
        search = GridSearchCV(
            pipeline,
            spec.params,
            scoring=SCORING,
            refit="average_precision",
            cv=splitter,
            n_jobs=1,
            error_score="raise",
        )
    else:
        search = RandomizedSearchCV(
            pipeline,
            spec.params,
            n_iter=spec.n_iter,
            scoring=SCORING,
            refit="average_precision",
            cv=splitter,
            random_state=seed,
            n_jobs=1,
            error_score="raise",
        )
    search.fit(x_train, y_train)
    elapsed = time.perf_counter() - started
    summary = _best_cv(search)
    summary.update(
        {
            "name": spec.name,
            "status": "ok",
            "best_params": to_jsonable(search.best_params_),
            "fit_seconds": float(elapsed),
            "class_weight_experiment": _grouped_param(search.cv_results_, "param_model__class_weight"),
            "scale_pos_weight_experiment": _grouped_param(search.cv_results_, "param_model__scale_pos_weight"),
            "cv_metric_note": (
                "average precision and ROC-AUC do not use a cutoff. "
                "Accuracy, precision, recall, and F1 in the comparison table use the cross-validation default cutoff of 0.5. "
                "The business cutoff is chosen later on the internal validation split."
            ),
        }
    )
    return search.best_estimator_, summary


def _resampling_experiment(x_train, y_train, seed: int, folds: int) -> dict:
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    balanced = make_model_pipeline(
        LogisticRegression(max_iter=500, solver="liblinear", class_weight="balanced", random_state=seed),
        scale_numeric=True,
    )
    oversampled = RandomOversampleClassifier(
        estimator=make_model_pipeline(
            LogisticRegression(max_iter=500, solver="liblinear", class_weight=None, random_state=seed),
            scale_numeric=True,
        ),
        random_state=seed,
    )
    balanced_scores = cross_val_score(balanced, x_train, y_train, cv=splitter, scoring="average_precision", n_jobs=1)
    over_scores = cross_val_score(oversampled, x_train, y_train, cv=splitter, scoring="average_precision", n_jobs=1)
    balanced_mean = float(balanced_scores.mean())
    over_mean = float(over_scores.mean())
    return {
        "model": "logistic_regression",
        "class_weight_balanced_average_precision": balanced_mean,
        "random_oversample_average_precision": over_mean,
        "adopted": bool(over_mean >= balanced_mean + 0.01),
        "adoption_margin": 0.01,
        "applied_to_deployed_model": False,
        "note": (
            "Oversampling is fit only on training rows inside each cross-validation fold. "
            "It is deployed only when it beats class_weight='balanced' by at least 0.01 average precision "
            "and logistic regression is the selected model."
        ),
    }


def _select(results: list[dict]) -> dict:
    eligible = [row for row in results if row.get("status") == "ok"]
    if not eligible:
        raise RuntimeError("No model finished training.")
    eligible.sort(
        key=lambda row: (
            row["cv_average_precision_mean"],
            row["cv_recall_at_0_5_mean"],
            row["cv_roc_auc_mean"],
        ),
        reverse=True,
    )
    return eligible[0]


def _xy(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    features = frame.loc[:, list(MODEL_FEATURES)].copy()
    target = (frame[TARGET_COLUMN] == "Yes").astype(int)
    return features, target


def train(
    csv_path: str | Path,
    output_dir: str | Path,
    cleaned_path: str | Path | None = None,
    seed: int = 42,
    fast: bool = False,
    fn_cost: float = 5.0,
    fp_cost: float = 1.0,
) -> dict:
    csv_path = Path(csv_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw = read_raw_csv(csv_path)
    cleaned, notes = clean_customers(raw, require_target=True, require_identifier=True)
    quality = build_quality_report(raw, cleaned, notes)
    trainable, exclusion = filter_trainable(cleaned)
    before = len(trainable)
    trainable = trainable.drop_duplicates(ID_COLUMN, keep="last").reset_index(drop=True)
    exclusion["duplicate_ids_removed_before_split"] = int(before - len(trainable))

    label = dataset_label_for_path(csv_path)
    eda = build_eda(trainable)
    eda["dataset_label"] = label
    (output_dir / "eda_report.json").write_text(json.dumps(to_jsonable(eda), indent=2), encoding="utf-8")
    (output_dir / "eda_report.md").write_text(render_eda_markdown(eda), encoding="utf-8")
    chart_notes = write_charts(trainable, output_dir / "figures")
    (output_dir / "data_quality.json").write_text(json.dumps(to_jsonable(quality), indent=2), encoding="utf-8")
    if cleaned_path is not None:
        Path(cleaned_path).parent.mkdir(parents=True, exist_ok=True)
        trainable.to_csv(cleaned_path, index=False)

    train_val, test_df = train_test_split(
        trainable, test_size=0.2, stratify=trainable[TARGET_COLUMN], random_state=seed
    )
    train_df, val_df = train_test_split(
        train_val, test_size=0.25, stratify=train_val[TARGET_COLUMN], random_state=seed
    )
    if not set(train_df.index).isdisjoint(val_df.index) or not set(train_df.index).isdisjoint(test_df.index):
        raise RuntimeError("Train, validation, and test indexes overlap.")
    if not set(val_df.index).isdisjoint(test_df.index):
        raise RuntimeError("Validation and test indexes overlap.")

    x_train, y_train = _xy(train_df)
    x_val, y_val = _xy(val_df)
    x_test, y_test = _xy(test_df)
    folds = 2 if fast else 3
    specs, model_note = build_specs(y_train, seed, fast)
    fitted: dict[str, object] = {}
    results: list[dict] = []
    for spec in specs:
        print(f"Training {spec.name}...", flush=True)
        estimator, summary = _fit_spec(spec, x_train, y_train, seed, folds)
        fitted[spec.name] = estimator
        results.append(summary)
        print(
            f"  CV average precision {summary['cv_average_precision_mean']:.4f} "
            f"in {summary['fit_seconds']:.1f}s",
            flush=True,
        )

    resampling = _resampling_experiment(x_train, y_train, seed, folds)
    selected = _select(results)
    selected_name = selected["name"]
    template = clone(fitted[selected_name])
    if resampling["adopted"] and selected_name == "logistic_regression":
        template = RandomOversampleClassifier(estimator=clone(fitted[selected_name]), random_state=seed)
        resampling["applied_to_deployed_model"] = True

    raw_model = clone(template).fit(x_train, y_train)
    raw_val = raw_model.predict_proba(x_val)[:, 1]
    raw_brier = float(brier_score_loss(y_val, raw_val))
    calibration = {
        "applied": False,
        "method": None,
        "validation_brier_uncalibrated": raw_brier,
        "validation_brier_calibrated": None,
        "reason": "Calibration was not applied.",
    }
    val_proba = raw_val
    try:
        calibrated = CalibratedClassifierCV(clone(template), method="sigmoid", cv=folds)
        calibrated.fit(x_train, y_train)
        cal_val = calibrated.predict_proba(x_val)[:, 1]
        cal_brier = float(brier_score_loss(y_val, cal_val))
        calibration["validation_brier_calibrated"] = cal_brier
        if cal_brier < raw_brier - 0.001:
            calibration["applied"] = True
            calibration["method"] = "sigmoid"
            calibration["reason"] = (
                "Sigmoid calibration lowered internal-validation Brier score by more than 0.001, so it was kept."
            )
            val_proba = cal_val
            comparison_model = calibrated
        else:
            calibration["reason"] = (
                "Sigmoid calibration did not improve internal-validation Brier score by more than 0.001, "
                "so the uncalibrated model was kept."
            )
            comparison_model = raw_model
    except Exception as exc:
        calibration["reason"] = f"Calibration was skipped ({type(exc).__name__}: {exc})."
        comparison_model = raw_model

    best_threshold, threshold_grid = tune_cost_threshold(y_val, val_proba, fn_cost=fn_cost, fp_cost=fp_cost)
    high_threshold = float(best_threshold["threshold"])
    base_rate = float(y_train.mean())
    medium_threshold = choose_medium_threshold(y_val, val_proba, high_threshold, base_rate)
    threshold_rationale = (
        f"High-risk threshold {high_threshold:.2f} minimizes {fn_cost:.0f} x false negatives + "
        f"{fp_cost:.0f} x false positives on the internal validation split "
        f"(validation cost {best_threshold['cost']:.0f}, precision {best_threshold['precision']:.3f}, "
        f"recall {best_threshold['recall']:.3f}). "
        f"Medium-risk threshold {medium_threshold:.2f} is the lowest cutoff below that decision point whose "
        f"validation precision is still at least the training churn rate of {base_rate:.3f}. "
        "Customers at or above the high threshold are classified as likely to churn. "
        "Medium is an elevated-risk band, not a second positive class. "
        "These cutoffs come from the validation cost search and can be changed in Settings. "
        "Validation metrics at this cutoff are optimistic because the cutoff was chosen on that same split. "
        "Use the held-out test metrics as the unbiased estimate."
    )
    validation_metrics = binary_metrics(y_val, val_proba, high_threshold)
    validation_metrics["label"] = "internal_validation_threshold_was_selected_on_this_split"

    if calibration["applied"]:
        deployed = CalibratedClassifierCV(clone(template), method="sigmoid", cv=folds)
    else:
        deployed = clone(template)
    x_fit = pd.concat([x_train, x_val], axis=0)
    y_fit = pd.concat([y_train, y_val], axis=0)
    deployed.fit(x_fit, y_fit)

    test_proba = deployed.predict_proba(x_test)[:, 1]
    test_metrics = binary_metrics(y_test, test_proba, high_threshold)
    test_metrics["curves"] = _curves(y_test, test_proba)
    test_metrics["latency"] = _latency(deployed, x_test)
    test_metrics["label"] = "held_out_test_after_model_selection"
    test_metrics["average_precision_ranking"] = float(average_precision_score(y_test, test_proba))

    importance = grouped_permutation_importance(
        comparison_model,
        x_val,
        y_val,
        n_repeats=1 if fast else 3,
        seed=seed,
    )
    associations = training_associations(train_df)
    baselines, feature_spec = _baselines_and_spec(x_fit)
    from ml.business.segments import fit_segments

    segment_artifact = fit_segments(train_df, seed=seed, fast=fast)
    segment_summary = {
        "k": segment_artifact["k"],
        "silhouette_by_k": segment_artifact["silhouette_by_k"],
        "selected_silhouette": segment_artifact["selected_silhouette"],
        "profiles": segment_artifact["profiles"],
        "feature_names": segment_artifact["feature_names"],
        "excluded": segment_artifact["excluded"],
        "fit_rows": segment_artifact["fit_rows"],
        "note": segment_artifact["note"],
    }

    selection_reason = (
        f"{selected_name} had the highest training-split cross-validated average precision "
        f"({selected['cv_average_precision_mean']:.4f}). "
        "Average precision is the selection metric because the classes are imbalanced and accuracy alone can hide missed churners. "
        "Ties break on recall at the 0.5 cutoff, then ROC-AUC. The held-out test set was not used to choose the model."
    )
    background = x_fit.sample(n=min(120, len(x_fit)), random_state=seed)
    trained_at = datetime.now(timezone.utc).isoformat()
    metadata = {
        "trained_at": trained_at,
        "seed": seed,
        "fast_mode": fast,
        "dataset": {
            "label": label,
            "source_url": DATASET_SOURCE_URL,
            "mirror_note": DATASET_MIRROR_NOTE,
            "path": str(csv_path),
            "sha256": _sha256(csv_path),
            "rows_raw": quality["rows_raw"],
            "rows_trainable": int(len(trainable)),
        },
        "excluded_features": list(PROTECTED_COLUMNS),
        "exclusion_reason": EXCLUSION_REASON,
        "model_features": list(MODEL_FEATURES),
        "class_mapping": {"retained": 0, "churn": 1, "positive_label": "Yes", "negative_label": "No"},
        "split": {
            "test_fraction": 0.2,
            "validation_fraction_of_remaining": 0.25,
            "n_train": int(len(train_df)),
            "n_validation": int(len(val_df)),
            "n_test": int(len(test_df)),
            "splits_are_disjoint": True,
            "train_churn_rate": float(y_train.mean()),
            "validation_churn_rate": float(y_val.mean()),
            "test_churn_rate": float(y_test.mean()),
        },
        "row_exclusions": exclusion,
        "library_versions": _versions(),
        "model_availability_note": model_note,
        "comparison": results,
        "comparison_scope": "training_split_cross_validation_only",
        "resampling_experiment": resampling,
        "selected_model": selected_name,
        "selection_metric": "average_precision",
        "selection_reason": selection_reason,
        "calibration": calibration,
        "cost_assumption": {
            "false_negative_cost": fn_cost,
            "false_positive_cost": fp_cost,
            "note": (
                "Missing a customer who churns is weighted five times a contact with a customer who would have stayed. "
                "This is an explicit assumption, not a measured business cost."
            ),
        },
        "thresholds": {
            "decision": high_threshold,
            "medium": medium_threshold,
            "high": high_threshold,
            "rationale": threshold_rationale,
            "validation_search": threshold_grid,
        },
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "feature_importance": importance,
        "associations": associations,
        "feature_spec": feature_spec,
        "baselines": baselines,
        "segments": segment_summary,
        "chart_notes": chart_notes,
        "positive_class_index": 1,
    }
    artifact = {
        "pipeline": deployed,
        "feature_names": list(MODEL_FEATURES),
        "baselines": baselines,
        "feature_spec": feature_spec,
        "positive_label": "Yes",
        "decision_threshold": high_threshold,
        "medium_threshold": medium_threshold,
        "shap_background": background,
        "excluded_features": list(PROTECTED_COLUMNS),
        "selected_model": selected_name,
        "trained_at": trained_at,
    }
    joblib.dump(artifact, output_dir / "model.joblib")
    joblib.dump(
        {
            "scaler": segment_artifact["scaler"],
            "model": segment_artifact["model"],
            "total_median": segment_artifact["total_median"],
            "feature_names": segment_artifact["feature_names"],
            "names": segment_artifact["names"],
            "k": segment_artifact["k"],
        },
        output_dir / "segments.joblib",
    )
    (output_dir / "metadata.json").write_text(json.dumps(to_jsonable(metadata), indent=2), encoding="utf-8")
    print(
        f"Selected {selected_name}. Held-out average precision "
        f"{test_metrics['average_precision']:.4f}, recall {test_metrics['recall']:.4f}, "
        f"F1 {test_metrics['f1']:.4f} at threshold {high_threshold:.2f}."
    )
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the churn model and write evaluation artifacts.")
    parser.add_argument("--csv", type=Path, default=ROOT / "data" / "raw" / "Telco-Customer-Churn.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "ml" / "artifacts")
    parser.add_argument("--cleaned", type=Path, default=ROOT / "data" / "processed" / "telco_cleaned.csv")
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    train(
        csv_path=args.csv,
        output_dir=args.output,
        cleaned_path=args.cleaned,
        seed=args.seed,
        fast=args.fast,
    )


if __name__ == "__main__":
    main()
