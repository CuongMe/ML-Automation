"""
ui.py

All Streamlit rendering functions. No business logic lives here — this module
only translates data and state into visual output.
"""

from dataclasses import dataclass
from typing import IO, List, Optional, Tuple, cast

import pandas as pd
import streamlit as st
from data_preprocessor import PreprocessOptions, PreprocessResult

# Session-state key and increment size for the incremental row preview.
_PREVIEW_ROWS_KEY = "preview_rows"
_PREVIEW_ROWS_STEP = 30


def init_page() -> None:
    """Configure the Streamlit page and render the app title.

    Must be called once at the start of every run, before any other
    Streamlit command, because ``set_page_config`` must be the first call.
    """
    st.set_page_config(page_title="ML Automation", layout="wide")
    st.title("Machine Learning Automation")
    st.write("Upload a CSV file to begin.")


def render_upload_widget() -> Optional[IO[bytes]]:
    """Render a CSV file-upload widget and return the uploaded file.

    Returns a Streamlit ``UploadedFile`` object (which implements ``IO[bytes]``)
    when the user has selected a file, or ``None`` while no file has been provided.
    """
    result = st.file_uploader("Upload CSV", type=["csv"])
    # UploadedFile satisfies IO[bytes]; cast makes the type checker happy
    # without importing Streamlit's internal UploadedFile class directly.
    return cast(Optional[IO[bytes]], result)


def render_waiting_upload() -> None:
    """Show a friendly placeholder when no file has been uploaded yet."""
    st.info("Waiting for CSV file.")


def render_error(message: str) -> None:
    """Display an error message using Streamlit's standard error style."""
    st.error(message)


def _friendly_dtype(dtype: str) -> str:
    """Convert a pandas dtype string to a plain-English label.

    Examples::

        int64            → Integer
        float32          → Decimal
        object           → Text
        bool             → Boolean
        datetime64[ns]   → DateTime
        category         → Category
        timedelta64[ns]  → Duration
    """
    lower = dtype.lower()
    if lower.startswith("int"):
        return "Integer"
    if lower.startswith("float"):
        return "Decimal"
    if lower == "object":
        return "Text"
    if lower == "bool":
        return "Boolean"
    if lower.startswith("datetime"):
        return "DateTime"
    if lower == "category":
        return "Category"
    if lower.startswith("timedelta"):
        return "Duration"
    return dtype  # fall back to the raw dtype string for any unrecognised type


def _clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of *df* with unnamed columns labelled ``(empty N)``.

    pandas auto-generates ``Unnamed: 0``, ``Unnamed: 1`` … for CSV columns
    that had no header. This replaces them with numbered placeholders such as
    ``(empty 1)``, ``(empty 2)`` so every column name stays unique and
    PyArrow can serialise the DataFrame without a duplicate-name error.
    """
    counter = 0
    rename_map: dict = {}
    for col in df.columns:
        if str(col).startswith("Unnamed:"):
            counter += 1
            rename_map[col] = f"(empty {counter})"
    return df.rename(columns=rename_map) if rename_map else df


def render_dataset(df: pd.DataFrame) -> None:
    """Render a data preview and summary statistics for the loaded DataFrame.

    The preview starts at 30 rows. Clicking '⬇ Show more' adds another
    30 rows each time until the full dataset is visible. An expandable
    overview section shows row/column counts, a per-column missing-value
    breakdown, and a human-readable column-type table.

    Columns with no header (pandas ``Unnamed: N`` placeholders) are shown in
    the preview with labelled placeholders but are excluded from all stats,
    type tables, and column counts because they carry no real data.
    """
    # Columns pandas auto-named because the CSV had no header for them.
    unnamed_mask = [str(c).startswith("Unnamed:") for c in df.columns]
    named_df = df.loc[:, [not u for u in unnamed_mask]]   # real columns only

    # --- Incremental row preview (shows all columns with cleaned names) ---
    if _PREVIEW_ROWS_KEY not in st.session_state:
        st.session_state[_PREVIEW_ROWS_KEY] = _PREVIEW_ROWS_STEP

    visible_rows: int = min(st.session_state[_PREVIEW_ROWS_KEY], len(df))
    display_df = _clean_column_names(df.head(visible_rows))

    st.subheader("Data Preview")
    st.dataframe(display_df, use_container_width=True)

    remaining = len(df) - visible_rows
    if remaining > 0:
        if st.button(f"⬇ Show more  ({remaining:,} rows remaining)"):
            st.session_state[_PREVIEW_ROWS_KEY] += _PREVIEW_ROWS_STEP
            st.rerun()
    else:
        st.caption(f"Showing all {len(df):,} rows.")

    # --- Dataset Overview (based on named columns only) ---
    with st.expander("Dataset Overview", expanded=True):
        total_missing = int(named_df.isna().sum().sum())
        total_cells = named_df.shape[0] * named_df.shape[1]
        missing_pct = (total_missing / total_cells * 100) if total_cells > 0 else 0.0

        col1, col2, col3 = st.columns(3)
        col1.metric("Rows", f"{named_df.shape[0]:,}")
        col2.metric("Columns", f"{named_df.shape[1]:,}")
        col3.metric(
            "Missing Cells",
            f"{total_missing:,}",
            delta=f"{missing_pct:.1f}% of all cells",
            delta_color="inverse",
        )

        # Per-column missing-value breakdown (named columns only).
        st.subheader("Missing Data")
        missing_counts = named_df.isna().sum()
        cols_with_missing = missing_counts[missing_counts > 0]

        if cols_with_missing.empty:
            st.success("No missing values found in any column.")
        else:
            missing_pct_series = (cols_with_missing / len(named_df) * 100).round(1)
            missing_df = pd.DataFrame({
                "Column": cols_with_missing.index.tolist(),
                "Missing Count": cols_with_missing.values,
                "Missing %": missing_pct_series.values,
            }).sort_values("Missing %", ascending=False).reset_index(drop=True)
            st.dataframe(missing_df, use_container_width=True, hide_index=True)

        # Column type table — unnamed/empty columns excluded entirely.
        st.subheader("Column Types")
        type_df = pd.DataFrame({
            "Column": named_df.columns.tolist(),
            "Type": [_friendly_dtype(str(d)) for d in named_df.dtypes],
            "Raw dtype": named_df.dtypes.astype(str).values,
        })
        st.dataframe(type_df, use_container_width=True, hide_index=True)


@dataclass
class ColumnSelection:
    """Holds the user's choice of target and feature columns."""

    target: str
    features: List[str]


def render_column_selection(named_columns: List[str]) -> Optional[ColumnSelection]:
    """Render target-column and feature-column selectors.

    Displays a divider and two widgets:
    - A selectbox to pick the single target (label) column.
    - A multiselect to pick the feature (input) columns, defaulting to
      every column except the chosen target.

    Returns a :class:`ColumnSelection` once both fields are valid,
    or ``None`` while the selection is incomplete so the caller can
    stop rendering early.
    """
    st.divider()
    st.subheader("Column Selection")

    # "" acts as a placeholder so the user must make an explicit choice.
    target_options = [""] + named_columns
    target = st.selectbox(
        "Target column  (what you want the model to predict)",
        options=target_options,
        index=0,
        help="Select the column your model should learn to predict.",
    )

    if not target:
        st.info("Select a target column to continue.")
        return None

    # Feature checkboxes — displayed in a bordered 3-column grid so the user
    # can tick/untick individual columns rather than hunt through a dropdown.
    st.markdown("**Feature columns** — tick the inputs the model will learn from")
    feature_cols_available = [c for c in named_columns if c != target]

    with st.container(border=True):
        grid = st.columns(3)
        selected_features: List[str] = []
        for i, feat in enumerate(feature_cols_available):
            # Each checkbox needs a unique key so Streamlit tracks it across reruns.
            if grid[i % 3].checkbox(feat, value=True, key=f"feat__{feat}"):
                selected_features.append(feat)

    if not selected_features:
        st.warning("Tick at least one feature column to continue.")
        return None

    return ColumnSelection(target=target, features=selected_features)


def render_preprocess_options(has_missing: bool) -> PreprocessOptions:
    """Render optional preprocessing step toggles and return the user's choices.

    Displays ten preprocessing operations grouped into three categories inside
    a collapsible expander. Enabling a step that has configurable parameters
    reveals those controls inline.

    Parameters
    ----------
    has_missing:
        Whether the dataset contains any missing values. When True, the
        "Remove Sparse Features" checkbox is ticked by default as a helpful
        nudge.
    """
    # Default values live here so variables are always defined even when
    # their parent checkbox is unchecked.
    remove_sparse = False
    sparse_threshold = 0.5
    impute_numeric: str = "median"
    impute_categorical: str = "most_frequent"
    discretize = False
    n_bins = 5
    continuize = False
    normalize = False
    randomize = False
    select_relevant = False
    n_relevant = 10
    select_random = False
    n_random = 10
    apply_pca = False
    n_pca = 10
    apply_cur = False
    n_cur = 10

    with st.expander("⚙️ Preprocessing Options", expanded=False):
        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown("**Data Cleaning**")

            remove_sparse = st.checkbox(
                "Remove Sparse Features",
                value=has_missing,
                help="Drop columns where more than the set % of values are missing.",
            )
            if remove_sparse:
                sparse_threshold = st.slider(
                    "Max allowed missing %", 0.1, 0.95, 0.5, 0.05,
                    key="sparse_thresh",
                )

            impute_numeric = st.selectbox(  # type: ignore[assignment]
                "Impute Numeric Strategy",
                options=["median", "mean", "constant"],
                index=0,
                help="How missing values in numeric columns are filled.",
            )
            impute_categorical = st.selectbox(  # type: ignore[assignment]
                "Impute Categorical Strategy",
                options=["most_frequent", "constant"],
                index=0,
                help="How missing values in categorical columns are filled.",
            )

            st.markdown("**Transformations**")

            discretize = st.checkbox(
                "Discretize Continuous Variables",
                help="Bin numeric columns into equal-width intervals (KBinsDiscretizer).",
            )
            if discretize:
                n_bins = int(st.number_input("Number of bins", 2, 50, 5, key="n_bins"))

            continuize = st.checkbox(
                "Continuize Discrete Variables",
                help="Encode categorical columns as ordinal integers instead of one-hot vectors.",
            )
            normalize = st.checkbox(
                "Normalize Features",
                help="Scale each numeric feature to the [0, 1] range (MinMaxScaler).",
            )
            randomize = st.checkbox(
                "Randomize Rows",
                help="Shuffle all rows before the train/test split.",
            )

        with col_b:
            st.markdown("**Feature Selection**")

            select_relevant = st.checkbox(
                "Select Relevant Features",
                help="Keep the K most statistically significant features (SelectKBest).",
            )
            if select_relevant:
                n_relevant = int(
                    st.number_input("Features to keep", 1, 500, 10, key="n_relevant")
                )

            select_random = st.checkbox(
                "Select Random Features",
                help="Keep a reproducible random subset of K features.",
            )
            if select_random:
                n_random = int(
                    st.number_input("Random features to keep", 1, 500, 10, key="n_random")
                )

            st.markdown("**Dimensionality Reduction**")

            apply_pca = st.checkbox(
                "Principal Component Analysis (PCA)",
                help="Project features onto N principal components.",
            )
            if apply_pca:
                n_pca = int(
                    st.number_input("PCA components", 1, 500, 10, key="n_pca")
                )

            apply_cur = st.checkbox(
                "CUR Matrix Decomposition",
                help=(
                    "Select the N most statistically significant original columns "
                    "using column leverage scores. More interpretable than PCA."
                ),
            )
            if apply_cur:
                n_cur = int(
                    st.number_input("CUR components", 1, 500, 10, key="n_cur")
                )

    return PreprocessOptions(
        remove_sparse=remove_sparse,
        sparse_threshold=sparse_threshold,
        impute_numeric_strategy=str(impute_numeric),
        impute_categorical_strategy=str(impute_categorical),
        discretize=discretize,
        n_bins=n_bins,
        continuize=continuize,
        normalize=normalize,
        randomize=randomize,
        select_relevant=select_relevant,
        n_relevant=n_relevant,
        select_random=select_random,
        n_random=n_random,
        apply_pca=apply_pca,
        n_pca_components=n_pca,
        apply_cur=apply_cur,
        n_cur_components=n_cur,
    )


def render_preprocessing_result(result: PreprocessResult) -> None:
    """Show a summary of what the preprocessor produced."""
    st.divider()
    st.subheader("Preprocessing Summary")

    tag = "Classification" if result.task_type == "classification" else "Regression"
    st.info(f"Detected task type: **{tag}**")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Training rows", f"{result.n_train:,}")
    col2.metric("Test rows", f"{result.n_test:,}")
    col3.metric("Numeric features", str(len(result.numeric_features)))
    col4.metric("Categorical features", str(len(result.categorical_features)))

    with st.expander("Feature breakdown", expanded=False):
        if result.numeric_features:
            st.markdown("**Numeric**")
            st.write(result.numeric_features)
        if result.categorical_features:
            st.markdown("**Categorical / Text**")
            st.write(result.categorical_features)
