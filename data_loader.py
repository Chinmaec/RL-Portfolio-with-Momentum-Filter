import pandas as pd

def load_data_returns(
    csv_path,
    parse_date=True,
    na_method="drop",   # "drop" or "fill"
    fill_value=0.0
):
    df = pd.read_csv(csv_path)

    # Find date column (DATE / Date / date / anything containing "date")
    date_col = next((c for c in df.columns if str(c).strip().lower() == "date"), None)
    if date_col is None:
        date_col = next((c for c in df.columns if "date" in str(c).strip().lower()), None)
    if date_col is None:
        raise ValueError("No date column found.")

    # Standardize date column name
    if date_col != "date":
        df = df.rename(columns={date_col: "date"})

    # Parse and set datetime index
    if parse_date:
        df["date"] = pd.to_datetime(df["date"])
        df = df.dropna(subset=["date"])

    df = df.set_index("date").sort_index()

    # Returns
    returns = df.pct_change()

    # Handle NaNs
    if na_method == "drop":
        returns = returns.dropna()
    elif na_method == "fill":
        returns = returns.fillna(fill_value)
    else:
        raise ValueError("na_method must be 'drop' or 'fill'.")

    return returns
