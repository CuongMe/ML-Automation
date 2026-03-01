"""
data_loader.py

Handles CSV loading with automatic encoding detection and result wrapping.
All public functions return a DatasetLoadResult instead of raising exceptions,
so callers never need a try/except block.
"""

from dataclasses import dataclass
from typing import IO, Iterable, Optional, Union

import pandas as pd


# Accepts a file path string or any file-like object (e.g. Streamlit UploadedFile).
FileLike = Union[str, IO[str], IO[bytes]]

# Tried in order when UTF-8 decoding fails. Covers Excel exports and legacy systems.
ENCODING_FALLBACKS = ("utf-8-sig", "cp1252", "latin1", "iso-8859-1", "cp874")


def _reset_stream(file_obj: FileLike) -> None:
    """Rewind a file-like object to the start so it can be read again.

    After a failed read attempt, file cursors are left at EOF.
    Plain path strings are skipped because they have no cursor to rewind.
    """
    if not isinstance(file_obj, str):
        try:
            file_obj.seek(0)
        except Exception:
            pass


def _read_with_encodings(file_obj: FileLike, encodings: Iterable[str]) -> pd.DataFrame:
    """Try reading the CSV with each encoding in sequence.

    Returns the DataFrame from the first encoding that succeeds.
    Re-raises immediately on non-encoding errors (e.g. malformed CSV structure).
    Raises the last UnicodeDecodeError if every encoding fails.
    """
    last_unicode_error: Optional[UnicodeDecodeError] = None
    for encoding in encodings:
        try:
            _reset_stream(file_obj)
            return pd.read_csv(file_obj, encoding=encoding)
        except UnicodeDecodeError as exc:
            last_unicode_error = exc  # try the next encoding
        except Exception:
            raise  # not an encoding issue — surface it straight away

    if last_unicode_error is not None:
        raise last_unicode_error
    raise ValueError("No encodings were provided to try.")


def load_csv_data(file_obj: FileLike) -> pd.DataFrame:
    """Read a CSV file into a DataFrame.

    Attempts UTF-8 first. If that raises a UnicodeDecodeError, retries
    with the common legacy encodings defined in ENCODING_FALLBACKS.
    """
    try:
        _reset_stream(file_obj)
        return pd.read_csv(file_obj)
    except UnicodeDecodeError:
        return _read_with_encodings(file_obj, ENCODING_FALLBACKS)


@dataclass
class DatasetLoadResult:
    """Outcome of a CSV load attempt.

    On success: ``dataframe`` holds the data and ``error`` is None.
    On failure: ``dataframe`` is None and ``error`` contains a human-readable message.
    """

    dataframe: Optional[pd.DataFrame]
    error: Optional[str] = None


def load_dataset(file_obj: Optional[FileLike]) -> DatasetLoadResult:
    """Load and validate a CSV file, returning a DatasetLoadResult.

    Wraps all error cases in the result object so the caller never needs
    a try/except block — just check ``result.error``.
    """
    if file_obj is None:
        return DatasetLoadResult(dataframe=None, error="No CSV file uploaded.")

    try:
        dataframe = load_csv_data(file_obj)
    except Exception as exc:
        return DatasetLoadResult(dataframe=None, error=f"Failed to load CSV: {exc}")

    if dataframe.empty:
        return DatasetLoadResult(dataframe=None, error="CSV is empty.")

    return DatasetLoadResult(dataframe=dataframe, error=None)
