import pandas as pd
from typing import List, Optional, Dict


def r_table(
    df: pd.DataFrame,
    columns: List[str],
    sortmode: str = "columns",
    na_indicator: Optional[List[str]] = None,
    treat_blank_as_na: bool = True,
    na_label_map: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """
    Create a R-style contingency table from a pandas DataFrame.

    This function emulates R's `table()` behavior while adding optional
    preprocessing steps commonly needed in data cleaning workflows, such as:
    - converting blank strings to missing values
    - creating explicit missing-value indicator columns
    - replacing missing values with custom labels before aggregation

    The output is a frequency table (counts of unique combinations of
    the selected columns), similar to:

        R: table(df$a, df$b)
        pandas equivalent: df.groupby(["a", "b"]).size()

    but with additional preprocessing controls.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing raw or preprocessed data.

    columns : list of str
        Column names to use as grouping keys for the frequency table.
        Each unique combination of these columns becomes one row in
        the output table.

        Must contain at least one valid column present in `df`.

    sortmode : {"columns", "freqdesc"}, default="columns"
        Determines how the output table is sorted.

        - "columns":
            Sort lexicographically by the grouping columns in the order
            they appear in `columns`. This mimics base R's default ordering.

        - "freqdesc":
            Sort rows by descending frequency count ("Freq"), useful for
            identifying the most common combinations.

    na_indicator : list of str, optional
        Columns to convert into explicit missing-value indicator variables.

        For each specified column:
        - values are replaced with a boolean series:
            True  -> original value was missing (NaN / NA)
            False -> original value was present

        Notes
        -----
        - This permanently changes the column's meaning within the function
          before grouping.
        - Useful for reproducing R's `is.na()` behavior inside summaries.
        - Should not include columns that are not present in `df`.

    treat_blank_as_na : bool, default=True
        If True, treats blank strings as missing values.

        Specifically:
        - "" (empty string)
        - "   " (whitespace-only strings)

        are converted to `pd.NA` before any grouping or transformation.

        This is important when data originates from CSV/Excel imports,
        where missing values may appear as empty strings instead of NA.

    na_label_map : dict of str -> str, optional
        Defines column-specific replacement labels for missing values.

        Example:
            {"gender": "Missing", "country": "Unknown"}

        Behavior:
        - Only applies to columns listed in the dictionary.
        - Missing values in those columns are replaced with the provided
          string label before grouping.
        - Forces affected columns to `object` dtype.

        Notes
        -----
        - This affects grouping behavior, since NA is no longer treated as
          a distinct missing value but as a categorical label.
        - Should be used carefully when consistency with statistical NA
          semantics is required.

    Returns
    -------
    pd.DataFrame
        A frequency table with the following structure:

        - One column per entry in `columns`
        - A final column named "Freq" representing counts of each
          unique combination
        - One row per unique combination of grouping variables

        The result is sorted according to `sortmode`.

    Raises
    ------
    ValueError
        If:
        - `columns` is empty
        - any column in `columns` is not found in `df`
        - `sortmode` is not one of {"columns", "freqdesc"}
        - a column in `na_indicator` is not found in `df`

    Notes
    -----
    Execution pipeline:

    1. Validate inputs
    2. Optionally convert blank strings to NA
    3. Optionally replace selected columns with NA indicators
    4. Optionally replace NA values with labels
    5. Group by selected columns and count occurrences
    6. Sort result according to `sortmode`

    Important behavior considerations:
    - The function operates on a copy of the DataFrame (`df.copy()`),
      so the original input is not modified.
    - Transformations applied in steps 2–4 affect grouping behavior.
    - Using `na_indicator` or `na_label_map` changes the semantic meaning
      of columns prior to aggregation.

    Examples
    --------
    Basic usage:

    >>> df = pd.DataFrame({
    ...     "a": ["x", "x", "y", "x"],
    ...     "b": [1, 1, 2, 2]
    ... })
    >>> r_table(df, ["a"])
       a  Freq
    0  x     3
    1  y     1

    Multiple grouping columns:

    >>> r_table(df, ["a", "b"])
       a  b  Freq
    0  x  1     2
    1  x  2     1
    2  y  2     1

    Frequency-sorted output:

    >>> r_table(df, ["a", "b"], sortmode="freqdesc")
       a  b  Freq
    0  x  1     2
    1  x  2     1
    2  y  2     1

    Treat blanks as missing:

    >>> df = pd.DataFrame({"a": ["x", "", "y", " "]})
    >>> r_table(df, ["a"], treat_blank_as_na=True)

    Create NA indicator column:

    >>> df = pd.DataFrame({"a": ["x", None, "y"]})
    >>> r_table(df, ["a"], na_indicator=["a"])

    Replace NA labels:

    >>> df = pd.DataFrame({"a": ["x", None, "y"]})
    >>> r_table(df, ["a"], na_label_map={"a": "Missing"})
    """

    if not columns:
        raise ValueError("`columns` must contain at least one column name.")

    missing_columns = [col for col in columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Columns not found in DataFrame: {missing_columns}")

    if sortmode not in {"columns", "freqdesc"}:
        raise ValueError("`sortmode` must be either 'columns' or 'freqdesc'.")

    df_work = df.copy()

    # --- Step 1: normalize blanks -> NA if requested ---
    if treat_blank_as_na:
        df_work = df_work.replace(r"^\s*$", pd.NA, regex=True)

    # --- Step 2: create NA indicator columns ---
    if na_indicator:
        for col in na_indicator:
            if col not in df_work.columns:
                raise ValueError(f"Column '{col}' not found for na_indicator")

            # replace the column with boolean NA indicator
            df_work[col] = df_work[col].isna()

    # --- Step 3: optional NA labeling ---
    if na_label_map:
        for col, label in na_label_map.items():
            if col in df_work.columns:
                df_work[col] = df_work[col].astype("object").where(
                    ~df_work[col].isna(), label
                )

    # --- Step 4: group and count ---
    result_df = (
        df_work.groupby(columns, dropna=False)
        .size()
        .reset_index(name="Freq")
    )

    # --- Step 5: sorting ---
    if sortmode == "columns":
        result_df = result_df.sort_values(
            by=columns,
            ascending=True,
            kind="stable"
        )
    elif sortmode == "freqdesc":
        result_df = result_df.sort_values(
            by="Freq",
            ascending=False,
            kind="stable"
        )

    return result_df.reset_index(drop=True)