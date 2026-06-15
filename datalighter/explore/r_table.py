import pandas as pd
from typing import List


def r_table(
    df: pd.DataFrame,
    columns: List[str],
    sortmode: str = "columns"
) -> pd.DataFrame:
    """
    Mimic the behavior of R's `table()` function using a pandas DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.

    columns : list[str]
        List of one or more column names to tabulate.

    sortmode : str, default="columns"
        Sorting strategy for the output DataFrame.

        Allowed values:
        - "columns":
            Sort by the provided columns in the same order they appear
            in `columns`.

        - "freqdesc":
            Sort by descending frequency count.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing:
        - the grouping columns
        - a column named "Freq" with occurrence counts

    Raises
    ------
    ValueError
        If:
        - `columns` is empty
        - a requested column does not exist
        - `sortmode` is invalid

    Examples
    --------
    >>> df = pd.DataFrame({
    ...     "a": ["x", "x", "y", "x"],
    ...     "b": [1, 1, 2, 2]
    ... })

    >>> r_table(df, ["a"])
       a  Freq
    0  x     3
    1  y     1

    >>> r_table(df, ["a", "b"])
       a  b  Freq
    0  x  1     2
    1  x  2     1
    2  y  2     1

    >>> r_table(df, ["a", "b"], sortmode="freqdesc")
       a  b  Freq
    0  x  1     2
    1  x  2     1
    2  y  2     1
    """

    if not columns:
        raise ValueError("`columns` must contain at least one column name.")

    missing_columns = [col for col in columns if col not in df.columns]
    if missing_columns:
        raise ValueError(
            f"Columns not found in DataFrame: {missing_columns}"
        )

    if sortmode not in {"columns", "freqdesc"}:
        raise ValueError(
            "`sortmode` must be either 'columns' or 'freqdesc'."
        )

    result_df = (
        df.groupby(columns, dropna=False)
        .size()
        .reset_index(name="Freq")
    )

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

    result_df = result_df.reset_index(drop=True)

    return result_df
