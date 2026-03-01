"""
app.py

Application entry point. Orchestrates the top-level flow:
  1. Initialise the page.
  2. Collect a CSV file from the user.
  3. Load and validate the file via the data layer.
  4. Show a data preview and dataset overview.
  5. Let the user select target and feature columns.
  6. Preprocess the data and show a summary.

Business logic belongs in data_loader.py / data_preprocessor.py;
rendering belongs in ui.py.
"""

from data_loader import load_dataset
from data_preprocessor import PreprocessConfig, PreprocessError, preprocess_dataset
from ui import (
    ColumnSelection,
    init_page,
    render_column_selection,
    render_dataset,
    render_error,
    render_preprocessing_result,
    render_preprocess_options,
    render_upload_widget,
    render_waiting_upload,
)


def main() -> None:
    """Run the Streamlit app for one script execution cycle.

    Streamlit re-runs this function from top to bottom on every user
    interaction, so each step should be cheap and side-effect-free.
    """
    init_page()

    # --- Step 1: collect CSV file ---
    uploaded_file = render_upload_widget()
    if uploaded_file is None:
        render_waiting_upload()
        return

    # --- Step 2: parse and validate —- errors are returned, not raised ---
    load_result = load_dataset(uploaded_file)
    if load_result.error is not None:
        render_error(load_result.error)
        return

    if load_result.dataframe is None:
        render_error("CSV loading returned no data.")
        return

    df = load_result.dataframe

    # --- Step 3: preview + dataset overview ---
    render_dataset(df)

    # Named columns only (no Unnamed: placeholders) for selection widgets.
    named_cols = [c for c in df.columns if not str(c).startswith("Unnamed:")]

    # --- Step 4: target and feature column selection ---
    selection = render_column_selection(named_cols)
    if selection is None:
        return  # user hasn't finished selecting yet

    # --- Step 5: preprocessing options ---
    has_missing = bool(df[selection.features].isna().any().any())
    options = render_preprocess_options(has_missing=has_missing)

    # --- Step 6: preprocess ---
    config = PreprocessConfig(
        target_col=selection.target,
        feature_cols=selection.features,
        options=options,
    )
    preprocess_result = preprocess_dataset(df, config)

    if isinstance(preprocess_result, PreprocessError):
        render_error(preprocess_result.message)
        return

    render_preprocessing_result(preprocess_result)


if __name__ == "__main__":
    main()
