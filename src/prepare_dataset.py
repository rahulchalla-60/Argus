import random
import time
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from feature_engine import FeatureEngine

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = BASE_DIR / "data" / "paysim.csv"
TRAIN_CSV = BASE_DIR / "data" / "train_data.csv"
TEST_CSV = BASE_DIR / "data" / "test_data.csv"
FULL_CSV = BASE_DIR / "data" / "training_data.csv"


def prepare_datasets_fast(
    csv_path: Path = DATASET_PATH,
    normal_sample_target: int = 27000,
    chunksize: int = 250000,
    seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    random.seed(seed)
    t0 = time.perf_counter()
    print("=== Phase 3.2: Fast Single-Pass Dataset Preparation ===")

    fe = FeatureEngine(time_window=24)
    fraud_accounts: set[str] = set()
    normal_accounts: set[str] = set()

    # Pass 1 (Single Pass): Stream chunks, update active accounts & sample
    print("Streaming transactions (Single Pass)...")
    for chunk in pd.read_csv(
        csv_path,
        chunksize=chunksize,
        usecols=["step", "type", "amount", "nameOrig", "nameDest", "isFraud"],
        dtype={"step": "int32", "type": "category", "amount": "float32", "isFraud": "int8"}
    ):
        # 1. Identify all fraud rows in chunk
        fraud_mask = chunk["isFraud"] == 1
        fraud_rows = chunk[fraud_mask]
        for r in fraud_rows.itertuples(index=False):
            fraud_accounts.add(r.nameOrig)
            fraud_accounts.add(r.nameDest)
            fe.update_transaction(r.nameOrig, r.nameDest, float(r.amount), int(r.step))

        # 2. Sample normal rows up to target
        if len(normal_accounts) < normal_sample_target:
            normal_rows = chunk[~fraud_mask]
            sampled = normal_rows.sample(n=min(len(normal_rows), 2000), random_state=seed)
            for r in sampled.itertuples(index=False):
                if r.nameOrig not in fraud_accounts and len(normal_accounts) < normal_sample_target:
                    normal_accounts.add(r.nameOrig)
                    fe.update_transaction(r.nameOrig, r.nameDest, float(r.amount), int(r.step))

    # Clean normal accounts pool of any accidental fraud overlap
    normal_accounts = normal_accounts - fraud_accounts
    all_target_accounts = fraud_accounts | normal_accounts

    print(f"Extraction complete in {time.perf_counter() - t0:.2f}s.")
    print(f"Accounts extracted: {len(fraud_accounts)} Fraud, {len(normal_accounts)} Normal.")

    records = []
    for acc_id in all_target_accounts:
        acc = fe.get_features(acc_id)
        if acc is None:
            continue
        is_fraud = 1 if acc_id in fraud_accounts else 0
        records.append({
            "account_id": acc_id,
            "forwarding_delay": -1 if acc.forwarding_delay is None else acc.forwarding_delay,
            "velocity": acc.velocity,
            "fan_in": acc.fan_in,
            "fan_out": acc.fan_out,
            "pass_through_ratio": acc.pass_through_ratio,
            "counterparties": len(acc.unique_senders) + len(acc.unique_receivers),
            "total_received": acc.total_received,
            "total_sent": acc.total_sent,
            "label": is_fraud
        })

    df_full = pd.DataFrame(records).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    df_full.to_csv(FULL_CSV, index=False)

    train_df, test_df = train_test_split(
        df_full,
        test_size=0.20,
        random_state=seed,
        stratify=df_full["label"]
    )

    train_df.to_csv(TRAIN_CSV, index=False)
    test_df.to_csv(TEST_CSV, index=False)

    print(f"\n[DONE] Saved Datasets in {time.perf_counter() - t0:.2f}s total:")
    print(f"  • Full Dataset  : {FULL_CSV} ({len(df_full)} rows)")
    print(f"  • Train (80%)   : {TRAIN_CSV} ({len(train_df)} rows: {train_df['label'].value_counts().to_dict()})")
    print(f"  • Test (20%)    : {TEST_CSV} ({len(test_df)} rows: {test_df['label'].value_counts().to_dict()})")

    return train_df, test_df


if __name__ == "__main__":
    prepare_datasets_fast()
