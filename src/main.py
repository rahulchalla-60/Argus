from pathlib import Path
from stream_simulator import StreamSimulator

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = BASE_DIR / "data" / "paysim.csv"

def main():
    sim = StreamSimulator(DATASET_PATH, nrows=10)
    for txn in sim.stream():
        print(f"[{txn['step']}] {txn['type']} | {txn['nameOrig']} -> {txn['nameDest']} | ${txn['amount']:,.2f} | Fraud: {txn['isFraud']}")

if __name__ == "__main__":
    main()