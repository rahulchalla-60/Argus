import pandas as pd

def load_dataset(path: str, nrows: int | None = None) -> pd.DataFrame:
    # ponytail: full load is ~1.5GB RAM; nrows lets us dev fast with small slices
    return pd.read_csv(path, nrows=nrows)

def get_summary(df):
    return {
        "rows": len(df),
        "unique_senders": df["nameOrig"].nunique(),
        "unique_receivers": df["nameDest"].nunique(),
        "min_step": df["step"].min(),
        "max_step": df["step"].max(),
        "min_amount": df["amount"].min(),
        "max_amount": df["amount"].max(),
        "median_amount": df["amount"].median()
    }