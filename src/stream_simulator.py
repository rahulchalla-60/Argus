import pandas as pd
from pathlib import Path

class StreamSimulator:
    def __init__(self, csv_path: str | Path, nrows: int | None = None):
        # PaySim is pre-sorted by step; nrows allows fast dev slicing
        self.df = pd.read_csv(csv_path, nrows=nrows)

    def stream(self):
        for row in self.df.itertuples(index=False):
            yield row._asdict()

if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent.parent
    data_path = base_dir / "data" / "paysim.csv"
    sim = StreamSimulator(data_path, nrows=5)
    for txn in sim.stream():
        print(f"[{txn['step']}] {txn['type']} | {txn['nameOrig']} -> {txn['nameDest']} | ${txn['amount']:,.2f} | Fraud: {txn['isFraud']}")
