import io
from typing import Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_fscore_support,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.multiclass import type_of_target


def make_ohe() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def infer_task_type(y: pd.Series) -> str:
    non_null_target = y.dropna()
    if non_null_target.empty:
        return "classification"

    target_kind = type_of_target(non_null_target)
    if target_kind in {"binary", "multiclass"}:
        return "classification"
    return "regression"


def prepare_target(y: pd.Series, task_type: str) -> pd.Series:
    if task_type == "classification":
        return y.astype(str).fillna("MISSING_TARGET")

    y_numeric = pd.to_numeric(y, errors="coerce")
    if y_numeric.isna().any():
        bad_samples = y[y_numeric.isna()].head(5).tolist()
        raise ValueError(
            f"Regression target has non-numeric values. Examples: {bad_samples}"
        )
    return y_numeric


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_features = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_features = [col for col in X.columns if col not in numeric_features]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", make_ohe()),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
    )


def get_models(task_type: str) -> Dict[str, object]:
    if task_type == "classification":
        return {
            "Logistic Regression": LogisticRegression(max_iter=3000),
            "Random Forest": RandomForestClassifier(
                n_estimators=300, random_state=42, n_jobs=-1
            ),
            "Gradient Boosting": GradientBoostingClassifier(random_state=42),
        }

    return {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=300, random_state=42, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingRegressor(random_state=42),
    }


def pick_best_model_with_cv(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    preprocessor: ColumnTransformer,
    models: Dict[str, object],
    task_type: str,
    cv_folds: int,
) -> Tuple[str, pd.DataFrame]:
    scoring = "accuracy" if task_type == "classification" else "r2"
    rows = []

    for model_name, model in models.items():
        pipeline = Pipeline(
            steps=[("preprocessor", preprocessor), ("model", model)]
        )
        try:
            cv_scores = cross_val_score(
                pipeline,
                X_train,
                y_train,
                cv=cv_folds,
                scoring=scoring,
                error_score="raise",
            )
            rows.append(
                {
                    "model": model_name,
                    "cv_mean": float(np.mean(cv_scores)),
                    "cv_std": float(np.std(cv_scores)),
                    "error": "",
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "model": model_name,
                    "cv_mean": np.nan,
                    "cv_std": np.nan,
                    "error": str(exc),
                }
            )

    cv_df = pd.DataFrame(rows)
    valid_rows = cv_df[cv_df["cv_mean"].notna()]
    if valid_rows.empty:
        raise RuntimeError("Cross-validation failed for all candidate models.")

    best_row = valid_rows.sort_values("cv_mean", ascending=False).iloc[0]
    return str(best_row["model"]), cv_df


def evaluate_classification(
    y_test: pd.Series,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
) -> Tuple[Dict[str, float], pd.DataFrame]:
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="weighted", zero_division=0
    )
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision_weighted": float(precision),
        "recall_weighted": float(recall),
        "f1_weighted": float(f1),
    }

    unique_labels = np.unique(y_test)
    if y_proba is not None and len(unique_labels) == 2 and y_proba.shape[1] == 2:
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_test, y_proba[:, 1]))
        except Exception:
            pass

    cm = confusion_matrix(y_test, y_pred, labels=unique_labels)
    cm_df = pd.DataFrame(
        cm,
        index=[f"Actual: {label}" for label in unique_labels],
        columns=[f"Pred: {label}" for label in unique_labels],
    )
    return metrics, cm_df


def evaluate_regression(
    y_test: pd.Series,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    return {
        "mae": float(mean_absolute_error(y_test, y_pred)),
        "rmse": float(rmse),
        "r2": float(r2_score(y_test, y_pred)),
    }


def extract_feature_importance(trained_pipeline: Pipeline) -> Optional[pd.DataFrame]:
    model = trained_pipeline.named_steps["model"]
    preprocessor = trained_pipeline.named_steps["preprocessor"]

    try:
        feature_names = preprocessor.get_feature_names_out()
    except Exception:
        return None

    if hasattr(model, "feature_importances_"):
        importance_values = np.asarray(model.feature_importances_)
    elif hasattr(model, "coef_"):
        coef_values = np.asarray(model.coef_)
        if coef_values.ndim == 1:
            importance_values = np.abs(coef_values)
        else:
            importance_values = np.mean(np.abs(coef_values), axis=0)
    else:
        return None

    if len(importance_values) != len(feature_names):
        return None

    importance_df = pd.DataFrame(
        {"feature": feature_names, "importance": importance_values}
    )
    return importance_df.sort_values("importance", ascending=False)


def safe_stop() -> None:
    """
    Stop Streamlit execution.
    In bare Python mode (`python app.py`), `st.stop()` does not halt the script,
    so we force a clean exit to avoid follow-up NameError/flow issues.
    """
    st.stop()
    raise SystemExit(0)


st.set_page_config(page_title="ML Automation App", layout="wide")
st.title("Machine Learning Automation (Streamlit)")
st.write(
    "Upload a CSV, choose your target, and automatically train/evaluate a model "
    "with built-in preprocessing."
)

uploaded_file = st.file_uploader("Upload training CSV", type=["csv"])
if uploaded_file is None:
    st.info("Upload a CSV file to start training.")
    safe_stop()

raw_df: Optional[pd.DataFrame] = None
try:
    raw_df = pd.read_csv(uploaded_file)
except Exception as exc:
    st.error(f"Could not read CSV: {exc}")
    safe_stop()

if raw_df is None or raw_df.empty:
    st.error("Uploaded dataset is empty.")
    safe_stop()

st.subheader("Dataset Preview")
st.dataframe(raw_df.head(20), use_container_width=True)

with st.expander("Dataset Overview"):
    col1, col2, col3 = st.columns(3)
    col1.metric("Rows", f"{raw_df.shape[0]:,}")
    col2.metric("Columns", f"{raw_df.shape[1]:,}")
    col3.metric("Missing Cells", f"{int(raw_df.isna().sum().sum()):,}")
    st.write("Column Types")
    st.dataframe(raw_df.dtypes.astype(str).rename("dtype"), use_container_width=True)

target_col = st.selectbox("Target column", raw_df.columns.tolist())
df = raw_df.dropna(subset=[target_col]).copy()
if len(df) < len(raw_df):
    dropped = len(raw_df) - len(df)
    st.warning(f"Dropped {dropped} rows with missing target values.")

feature_cols = [c for c in df.columns if c != target_col]
if not feature_cols:
    st.error("Need at least one feature column besides the target.")
    safe_stop()

X = df[feature_cols]
y_raw = df[target_col]
auto_task = infer_task_type(y_raw)

task_mode = st.radio(
    "Task type",
    ["Auto detect", "Classification", "Regression"],
    horizontal=True,
)
task_type = auto_task if task_mode == "Auto detect" else task_mode.lower()
st.caption(f"Using task type: **{task_type}** (auto-detected: {auto_task})")

try:
    y = prepare_target(y_raw, task_type)
except ValueError as exc:
    st.error(str(exc))
    safe_stop()

models = get_models(task_type)

st.sidebar.header("Training Settings")
test_size = st.sidebar.slider("Test size", min_value=0.1, max_value=0.4, value=0.2)
cv_folds = st.sidebar.slider("CV folds (for Auto model)", 3, 10, 5)
random_state = st.sidebar.number_input(
    "Random state", min_value=0, max_value=99999, value=42, step=1
)
model_choice = st.sidebar.selectbox(
    "Model",
    ["Auto (best by CV)"] + list(models.keys()),
)

train_button = st.button("Train Model", type="primary")
if train_button:
    try:
        with st.spinner("Training model..."):
            stratify_target = y if task_type == "classification" else None
            try:
                X_train, X_test, y_train, y_test = train_test_split(
                    X,
                    y,
                    test_size=test_size,
                    random_state=int(random_state),
                    stratify=stratify_target,
                )
            except ValueError:
                # Fallback for low-sample / highly imbalanced classes.
                X_train, X_test, y_train, y_test = train_test_split(
                    X,
                    y,
                    test_size=test_size,
                    random_state=int(random_state),
                    stratify=None,
                )

            preprocessor = build_preprocessor(X)
            selected_model_name = model_choice
            cv_results = None

            if model_choice == "Auto (best by CV)":
                selected_model_name, cv_results = pick_best_model_with_cv(
                    X_train=X_train,
                    y_train=y_train,
                    preprocessor=preprocessor,
                    models=models,
                    task_type=task_type,
                    cv_folds=cv_folds,
                )

            model = models[selected_model_name]
            trained_pipeline = Pipeline(
                steps=[("preprocessor", preprocessor), ("model", model)]
            )
            trained_pipeline.fit(X_train, y_train)

            y_pred = trained_pipeline.predict(X_test)
            if task_type == "classification":
                y_proba = (
                    trained_pipeline.predict_proba(X_test)
                    if hasattr(trained_pipeline, "predict_proba")
                    else None
                )
                metrics, confusion_df = evaluate_classification(y_test, y_pred, y_proba)
                regression_plot_df = None
            else:
                metrics = evaluate_regression(y_test, y_pred)
                confusion_df = None
                regression_plot_df = pd.DataFrame(
                    {"actual": y_test.to_numpy(), "predicted": y_pred}
                )

            feature_importance_df = extract_feature_importance(trained_pipeline)

            st.session_state["training_result"] = {
                "task_type": task_type,
                "selected_model_name": selected_model_name,
                "cv_results": cv_results,
                "metrics": metrics,
                "confusion_df": confusion_df,
                "regression_plot_df": regression_plot_df,
                "feature_importance_df": feature_importance_df,
                "pipeline": trained_pipeline,
                "feature_columns": feature_cols,
            }
    except Exception as exc:
        st.error(f"Training failed: {exc}")

if "training_result" in st.session_state:
    result = st.session_state["training_result"]
    st.success(f"Training complete. Best model: {result['selected_model_name']}")

    if result["cv_results"] is not None:
        st.subheader("Cross-Validation Results")
        st.dataframe(
            result["cv_results"].sort_values("cv_mean", ascending=False),
            use_container_width=True,
        )

    st.subheader("Evaluation Metrics")
    metric_df = pd.DataFrame(
        [{"metric": k, "value": v} for k, v in result["metrics"].items()]
    )
    st.dataframe(metric_df, use_container_width=True, hide_index=True)

    if result["task_type"] == "classification" and result["confusion_df"] is not None:
        st.subheader("Confusion Matrix")
        st.dataframe(result["confusion_df"], use_container_width=True)

    if result["task_type"] == "regression" and result["regression_plot_df"] is not None:
        st.subheader("Actual vs Predicted")
        st.scatter_chart(
            result["regression_plot_df"],
            x="actual",
            y="predicted",
            use_container_width=True,
        )

    if result["feature_importance_df"] is not None:
        st.subheader("Top Feature Importance")
        top_features = result["feature_importance_df"].head(20).set_index("feature")
        st.bar_chart(top_features, use_container_width=True)

    model_buffer = io.BytesIO()
    joblib.dump(result["pipeline"], model_buffer)
    model_bytes = model_buffer.getvalue()
    model_filename = result["selected_model_name"].lower().replace(" ", "_")
    st.download_button(
        label="Download Trained Pipeline (.joblib)",
        data=model_bytes,
        file_name=f"{model_filename}_pipeline.joblib",
        mime="application/octet-stream",
    )

    st.subheader("Batch Prediction")
    prediction_file = st.file_uploader(
        "Upload CSV for predictions",
        type=["csv"],
        key="prediction_file",
    )
    if prediction_file is not None:
        try:
            prediction_df = pd.read_csv(prediction_file)
            missing_cols = [
                col
                for col in result["feature_columns"]
                if col not in prediction_df.columns
            ]
            if missing_cols:
                st.error(f"Missing required feature columns: {missing_cols}")
            else:
                scored_df = prediction_df.copy()
                input_df = prediction_df[result["feature_columns"]]
                scored_df["prediction"] = result["pipeline"].predict(input_df)

                if (
                    result["task_type"] == "classification"
                    and hasattr(result["pipeline"], "predict_proba")
                ):
                    proba = result["pipeline"].predict_proba(input_df)
                    scored_df["confidence"] = np.max(proba, axis=1)

                st.dataframe(scored_df.head(50), use_container_width=True)
                st.download_button(
                    label="Download Predictions CSV",
                    data=scored_df.to_csv(index=False).encode("utf-8"),
                    file_name="predictions.csv",
                    mime="text/csv",
                )
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")
