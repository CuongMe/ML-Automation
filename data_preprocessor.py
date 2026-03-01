"""
data_preprocessor.py

Turns a raw DataFrame + column selection into train/test splits with a
fitted scikit-learn preprocessing pipeline attached, ready for model training.

Responsibilities
----------------
- Detect whether the task is classification or regression.
- Identify numeric vs categorical feature columns.
- Build a ColumnTransformer that handles imputation, scaling, and encoding.
- Split data into stratified train / test sets.
- Return a single PreprocessResult with everything the training step needs.

Nothing here imports Streamlit — all logic is pure Python / pandas / sklearn.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    KBinsDiscretizer,
    MinMaxScaler,
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
)

# A target column with fewer unique values than this threshold (as a fraction
# of total rows) is treated as classification; otherwise regression.
_CLASSIFICATION_UNIQUE_RATIO = 0.05
# Never treat a column as classification if it has more unique values than this,
# even if the ratio is small (e.g. an integer ID column in a tiny dataset).
_CLASSIFICATION_MAX_UNIQUE = 20


# ---------------------------------------------------------------------------
# Public data structures
# ---------------------------------------------------------------------------


@dataclass
class PreprocessOptions:
    """Optional preprocessing steps to apply to the feature set.

    All flags default to ``False`` / sensible defaults so the basic pipeline
    works without any extra configuration. Enable individual steps as needed.

    Data Cleaning
    -------------
    remove_sparse:
        Drop feature columns where the fraction of missing values exceeds
        ``sparse_threshold`` before the train/test split.
    sparse_threshold:
        Missing-value fraction above which a column is considered sparse
        (default 0.5 = 50 %).
    impute_numeric_strategy:
        How to fill missing values in numeric columns.
        ``"median"`` (default), ``"mean"``, or ``"constant"`` (fills with 0).
    impute_categorical_strategy:
        How to fill missing values in categorical columns.
        ``"most_frequent"`` (default) or ``"constant"`` (fills with ``"missing"``).

    Transformations
    ---------------
    discretize:
        Bin numeric columns into ``n_bins`` equal-width intervals using
        ``KBinsDiscretizer``.  Applied inside the numeric branch.
    n_bins:
        Number of bins when ``discretize`` is True (default 5).
    continuize:
        Encode categorical columns as ordinal integers (``OrdinalEncoder``)
        instead of one-hot vectors. Useful for tree-based models.
    normalize:
        Scale numeric features to the [0, 1] range using ``MinMaxScaler``
        after standard scaling.
    randomize:
        Shuffle all rows before the train/test split.

    Feature Selection
    -----------------
    select_relevant:
        Keep only the ``n_relevant`` most statistically significant features
        using ``SelectKBest`` (score function chosen by task type).
    n_relevant:
        Number of features to keep when ``select_relevant`` is True (default 10).
    select_random:
        Keep a reproducible random subset of ``n_random`` features.
    n_random:
        Number of features to keep when ``select_random`` is True (default 10).

    Dimensionality Reduction
    ------------------------
    apply_pca:
        Reduce dimensions to ``n_pca_components`` principal components.
    n_pca_components:
        Number of PCA components to retain (default 10).
    apply_cur:
        Select ``n_cur_components`` original feature dimensions using column
        leverage scores (CUR decomposition). More interpretable than PCA
        because the selected axes correspond to real input features.
    n_cur_components:
        Number of columns to retain when ``apply_cur`` is True (default 10).
    """

    # Data cleaning
    remove_sparse: bool = False
    sparse_threshold: float = 0.5
    impute_numeric_strategy: str = "median"           # "median" | "mean" | "constant"
    impute_categorical_strategy: str = "most_frequent"  # "most_frequent" | "constant"

    # Transformations
    discretize: bool = False
    n_bins: int = 5
    continuize: bool = False
    normalize: bool = False
    randomize: bool = False

    # Feature selection
    select_relevant: bool = False
    n_relevant: int = 10
    select_random: bool = False
    n_random: int = 10

    # Dimensionality reduction
    apply_pca: bool = False
    n_pca_components: int = 10
    apply_cur: bool = False
    n_cur_components: int = 10


@dataclass
class PreprocessConfig:
    """Inputs for a preprocessing run.

    Attributes
    ----------
    target_col:
        Name of the column the model should predict.
    feature_cols:
        Names of the columns the model receives as inputs.
    task_type:
        ``"auto"`` lets the preprocessor decide; pass ``"classification"`` or
        ``"regression"`` to override.
    test_size:
        Fraction of rows held out for evaluation (default 20 %).
    random_state:
        Seed for reproducible train/test splits.
    """

    target_col: str
    feature_cols: List[str]
    task_type: str = "auto"          # "auto" | "classification" | "regression"
    test_size: float = 0.2
    random_state: int = 42
    options: PreprocessOptions = field(default_factory=PreprocessOptions)


@dataclass
class PreprocessResult:
    """Outputs of a successful preprocessing run.

    Attributes
    ----------
    X_train, X_test:
        Numpy arrays of transformed features for train and test sets.
    y_train, y_test:
        Target arrays for train and test sets.
    pipeline:
        Fitted :class:`sklearn.pipeline.Pipeline` that can transform new data
        (or be combined with a model into a full pipeline for export).
    task_type:
        Resolved task type — always ``"classification"`` or ``"regression"``.
    numeric_features:
        Feature column names routed through numeric preprocessing.
    categorical_features:
        Feature column names routed through categorical preprocessing.
    feature_names_out:
        Ordered list of output feature names after transformation
        (useful for feature-importance analysis).
    n_train:
        Number of rows in the training split.
    n_test:
        Number of rows in the test split.
    """

    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    pipeline: Pipeline
    task_type: str                           # "classification" | "regression"
    numeric_features: List[str] = field(default_factory=list)
    categorical_features: List[str] = field(default_factory=list)
    feature_names_out: List[str] = field(default_factory=list)
    n_train: int = 0
    n_test: int = 0


@dataclass
class PreprocessError:
    """Returned instead of PreprocessResult when preprocessing cannot proceed."""

    message: str


# ---------------------------------------------------------------------------
# Custom sklearn-compatible transformers
# ---------------------------------------------------------------------------


class CURDecomposition(BaseEstimator, TransformerMixin):
    """CUR matrix decomposition: selects original columns by leverage score.

    Unlike PCA, CUR retains actual feature dimensions rather than rotated
    axes, making the output easier to interpret. The top-``n_components``
    columns are chosen deterministically by their squared L2 column norms
    (a proxy for statistical importance / leverage).

    Parameters
    ----------
    n_components:
        Number of columns to select.
    """

    def __init__(self, n_components: int = 10) -> None:
        self.n_components = n_components
        self.selected_indices_: np.ndarray = np.array([], dtype=int)

    def fit(self, X: np.ndarray, y: None = None) -> "CURDecomposition":
        """Select columns with the highest squared L2 norm (leverage scores)."""
        k = min(self.n_components, X.shape[1])
        col_norms = np.sum(X ** 2, axis=0)
        # argsort returns ascending order; the last k are the highest-norm columns.
        self.selected_indices_ = np.argsort(col_norms)[-k:]
        return self

    def transform(self, X: np.ndarray, y: None = None) -> np.ndarray:
        """Return only the selected columns."""
        return X[:, self.selected_indices_]

    def get_feature_names_out(
        self, input_features: Optional[List[str]] = None
    ) -> np.ndarray:
        """Return feature names for the selected columns."""
        if input_features is not None:
            return np.array(input_features)[self.selected_indices_]
        return np.array([f"cur_{i}" for i in self.selected_indices_])


class _RandomFeatureSelector(BaseEstimator, TransformerMixin):
    """Randomly selects a fixed number of features.

    Uses a fixed random seed so the selection is reproducible across
    Streamlit re-runs.
    """

    def __init__(self, n_features: int = 10, random_state: int = 42) -> None:
        self.n_features = n_features
        self.random_state = random_state
        self.selected_indices_: np.ndarray = np.array([], dtype=int)

    def fit(self, X: np.ndarray, y: None = None) -> "_RandomFeatureSelector":
        """Randomly sample column indices without replacement."""
        rng = np.random.RandomState(self.random_state)
        k = min(self.n_features, X.shape[1])
        self.selected_indices_ = np.sort(rng.choice(X.shape[1], size=k, replace=False))
        return self

    def transform(self, X: np.ndarray, y: None = None) -> np.ndarray:
        """Return only the randomly selected columns."""
        return X[:, self.selected_indices_]

    def get_feature_names_out(
        self, input_features: Optional[List[str]] = None
    ) -> np.ndarray:
        """Return feature names for the selected columns."""
        if input_features is not None:
            return np.array(input_features)[self.selected_indices_]
        return np.array([f"random_{i}" for i in self.selected_indices_])


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _detect_task_type(target_series: pd.Series) -> str:
    """Infer whether a target column belongs to a classification or regression task.

    Rules (applied in order):
    1. Non-numeric dtype → classification (strings, booleans, categories).
    2. Unique value count ≤ ``_CLASSIFICATION_MAX_UNIQUE`` → classification.
    3. Unique value ratio ≤ ``_CLASSIFICATION_UNIQUE_RATIO`` → classification.
    4. Everything else → regression.
    """
    if not pd.api.types.is_numeric_dtype(target_series):
        return "classification"

    n_unique = target_series.nunique()
    n_total = len(target_series.dropna())

    if n_unique <= _CLASSIFICATION_MAX_UNIQUE:
        return "classification"

    if n_total > 0 and (n_unique / n_total) <= _CLASSIFICATION_UNIQUE_RATIO:
        return "classification"

    return "regression"


def _split_feature_types(
    X: pd.DataFrame,
) -> Tuple[List[str], List[str]]:
    """Partition feature columns into numeric and categorical lists.

    A column is treated as numeric when pandas considers its dtype numeric
    (int*, float*, bool).  All other dtypes (object, category, string …)
    are treated as categorical.
    """
    numeric = [
        col for col in X.columns
        if pd.api.types.is_numeric_dtype(X[col])
    ]
    categorical = [
        col for col in X.columns
        if not pd.api.types.is_numeric_dtype(X[col])
    ]
    return numeric, categorical


def _build_feature_pipeline(
    numeric_features: List[str],
    categorical_features: List[str],
    options: PreprocessOptions,
    task_type: str,
) -> Pipeline:
    """Construct a preprocessing Pipeline for mixed numeric/categorical data.

    Stage 1 — ColumnTransformer (fitted per column type):

    Numeric branch:
        1. ``SimpleImputer`` — fills missing values (strategy from *options*).
        2. ``StandardScaler`` — zero mean, unit variance.
        3. ``KBinsDiscretizer`` — (optional) bins continuous values into intervals.
        4. ``MinMaxScaler`` — (optional) rescales to [0, 1] after standard scaling.

    Categorical branch:
        1. ``SimpleImputer`` — fills missing values (strategy from *options*).
        2. ``OneHotEncoder`` or ``OrdinalEncoder`` — chosen by *options.continuize*.

    Stage 2 — Post-transformer steps (applied to the full transformed array):
        - ``SelectKBest`` — keep the N most statistically relevant features.
        - ``_RandomFeatureSelector`` — keep a random subset of N features.
        - ``PCA`` — reduce to N principal components.
        - ``CURDecomposition`` — select N original columns by leverage score.
    """
    transformers = []

    if numeric_features:
        numeric_steps: list = [
            ("imputer", SimpleImputer(strategy=options.impute_numeric_strategy)),
            ("scaler", StandardScaler()),
        ]
        if options.discretize:
            numeric_steps.append((
                "discretizer",
                KBinsDiscretizer(
                    n_bins=options.n_bins, encode="ordinal", strategy="uniform"
                ),
            ))
        if options.normalize:
            numeric_steps.append(("normalizer", MinMaxScaler()))
        transformers.append(("numeric", Pipeline(numeric_steps), numeric_features))

    if categorical_features:
        if options.continuize:
            encoder: OneHotEncoder | OrdinalEncoder = OrdinalEncoder(
                handle_unknown="use_encoded_value", unknown_value=-1
            )
        else:
            encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        categorical_steps: list = [
            ("imputer", SimpleImputer(strategy=options.impute_categorical_strategy)),
            ("encoder", encoder),
        ]
        transformers.append(("categorical", Pipeline(categorical_steps), categorical_features))

    column_transformer = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False,
    )

    # Assemble the full pipeline: column transformer first, then optional
    # post-transformer steps in the order they should be applied.
    pipeline_steps: list = [("preprocessor", column_transformer)]

    if options.select_relevant:
        score_fn = f_classif if task_type == "classification" else f_regression
        pipeline_steps.append((
            "select_relevant",
            SelectKBest(score_func=score_fn, k=options.n_relevant),
        ))

    if options.select_random:
        pipeline_steps.append((
            "select_random",
            _RandomFeatureSelector(n_features=options.n_random),
        ))

    if options.apply_pca:
        pipeline_steps.append(("pca", PCA(n_components=options.n_pca_components)))

    if options.apply_cur:
        pipeline_steps.append((
            "cur",
            CURDecomposition(n_components=options.n_cur_components),
        ))

    return Pipeline(pipeline_steps)


def _resolve_feature_names(pipeline: Pipeline, numeric: List[str], categorical: List[str]) -> List[str]:
    """Extract the ordered output feature names from the fitted pipeline.

    Tries the full pipeline first (works when all steps implement
    ``get_feature_names_out``, which is the case for every standard sklearn
    transformer in sklearn ≥ 1.4, PCA, SelectKBest, and our custom
    transformers). Falls back to the column transformer only, then to the
    raw input column names if everything else fails.
    """
    try:
        return list(pipeline.get_feature_names_out())
    except Exception:
        pass
    try:
        return list(pipeline.named_steps["preprocessor"].get_feature_names_out())
    except Exception:
        return numeric + categorical


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def preprocess_dataset(
    df: pd.DataFrame,
    config: PreprocessConfig,
) -> PreprocessResult | PreprocessError:
    """Run the full preprocessing pipeline on *df* using *config*.

    Steps
    -----
    1. Validate that the requested columns exist.
    2. Drop rows where the target is missing.
    3. Resolve task type (auto-detect or use the value from config).
    4. Split into train / test sets (stratified for classification).
    5. Fit the preprocessing pipeline on the training set only
       (prevents data leakage).
    6. Transform both splits.
    7. Return a :class:`PreprocessResult` with everything the training
       step will need.

    Returns
    -------
    PreprocessResult on success, PreprocessError on failure.
    """
    # --- 1. Column validation ---
    missing_cols = [
        c for c in [config.target_col] + config.feature_cols
        if c not in df.columns
    ]
    if missing_cols:
        return PreprocessError(
            f"Columns not found in dataset: {missing_cols}"
        )

    opts = config.options

    # --- 2. Drop rows with missing target ---
    working_df = df[config.feature_cols + [config.target_col]].copy()
    working_df = working_df.dropna(subset=[config.target_col])

    if working_df.empty:
        return PreprocessError("No rows remain after dropping missing target values.")

    feature_cols = list(config.feature_cols)

    # --- 2a. Remove sparse feature columns (before split to avoid leakage) ---
    if opts.remove_sparse:
        sparse_cols = [
            col for col in feature_cols
            if working_df[col].isna().mean() > opts.sparse_threshold
        ]
        if sparse_cols:
            working_df = working_df.drop(columns=sparse_cols)
            feature_cols = [c for c in feature_cols if c not in sparse_cols]
        if not feature_cols:
            return PreprocessError(
                f"All feature columns were removed as sparse "
                f"(threshold: {opts.sparse_threshold:.0%} missing)."
            )

    # --- 2b. Shuffle rows before the split ---
    if opts.randomize:
        working_df = working_df.sample(
            frac=1, random_state=config.random_state
        ).reset_index(drop=True)

    X = working_df[feature_cols]
    y = working_df[config.target_col]

    # --- 3. Task type ---
    if config.task_type == "auto":
        task_type = _detect_task_type(y)
    else:
        task_type = config.task_type

    # --- 4. Train / test split ---
    # Stratify on target for classification to preserve class balance.
    stratify = y if task_type == "classification" else None
    try:
        X_train_raw, X_test_raw, y_train, y_test = train_test_split(
            X, y,
            test_size=config.test_size,
            random_state=config.random_state,
            stratify=stratify,
        )
    except ValueError:
        # Stratification fails when a class has only one sample; fall back.
        X_train_raw, X_test_raw, y_train, y_test = train_test_split(
            X, y,
            test_size=config.test_size,
            random_state=config.random_state,
        )

    # --- 5. Build and fit the pipeline on training data only ---
    # Fitting only on X_train_raw prevents data leakage into the test set.
    numeric_features, categorical_features = _split_feature_types(X_train_raw)
    pipeline = _build_feature_pipeline(
        numeric_features, categorical_features, opts, task_type
    )
    pipeline.fit(X_train_raw, y_train)

    # --- 6. Transform both splits ---
    X_train = pipeline.transform(X_train_raw)
    X_test = pipeline.transform(X_test_raw)

    # --- 7. Collect output feature names ---
    feature_names_out = _resolve_feature_names(
        pipeline, numeric_features, categorical_features
    )

    return PreprocessResult(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train.to_numpy(),
        y_test=y_test.to_numpy(),
        pipeline=pipeline,
        task_type=task_type,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        feature_names_out=feature_names_out,
        n_train=len(X_train),
        n_test=len(X_test),
    )
