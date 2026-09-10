from pathlib import Path
from typing import Generator
import pandas as pd


class StreamSimulator:
    def __init__(self, csv_path: str | Path, nrows: int | None = None):
        # PaySim is pre-sorted by step; nrows allows fast dev slicing
        self.df = pd.read_csv(csv_path, nrows=nrows)

    def stream(self):
        for row in self.df.itertuples(index=False):
            yield row._asdict()


def stream_paysim_transactions(
    csv_path: str | Path,
    chunk_size: int = 100,
    max_chunks: int | None = None
) -> Generator[list[dict], None, None]:
    """
    Streams transactions from paysim.csv in list-of-dict chunks for batch processing.
    """
    chunks_read = 0
    for chunk in pd.read_csv(csv_path, chunksize=chunk_size):
        yield chunk.to_dict(orient="records")
        chunks_read += 1
        if max_chunks is not None and chunks_read >= max_chunks:
            break


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent.parent
    data_path = base_dir / "data" / "paysim.csv"
    sim = StreamSimulator(data_path, nrows=5)
    for txn in sim.stream():
        print(f"[{txn['step']}] {txn['type']} | {txn['nameOrig']} -> {txn['nameDest']} | ${txn['amount']:,.2f} | Fraud: {txn['isFraud']}")
