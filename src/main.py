from data_loader import (
    load_dataset,
    get_summary
)

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = BASE_DIR / "data" / "paysim.csv"

def main():
    df = load_dataset(DATASET_PATH)


    summary = get_summary(df)

    print(summary)

if __name__ == "__main__":
    main()