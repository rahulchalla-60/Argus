import random
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from feature_engine import FeatureEngine

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = BASE_DIR / "data" / "paysim.csv"
TRAIN_CSV = BASE_DIR / "data" / "train_data.csv"
TEST_CSV = BASE_DIR / "data" / "test_data.csv"
FULL_CSV = BASE_DIR / "data" / "training_data.csv"


def prepare_datasets(
    csv_path: Path = DATASET_PATH,
    normal_sample_target: int = 27000,
    chunksize: int = 100000,
    seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    random.seed(seed)
    print("=== Phase 3.2: Preparing Train / Test Datasets ===")
    
    fe = FeatureEngine(time_window=24)
    fraud_accounts: set[str] = set()
    normal_accounts_pool: set[str] = set()

    print("Pass 1: Identifying all fraud accounts and sampling normal accounts...")
    for chunk in pd.read_csv(csv_path, chunksize=chunksize, usecols=["step", "type", "amount", "nameOrig", "nameDest", "isFraud"]):
        for r in chunk.itertuples(index=False):
            if r.isFraud == 1:
                fraud_accounts.add(r.nameOrig)
                fraud_accounts.add(r.nameDest)
            else:
                if len(normal_accounts_pool) < normal_sample_target * 2:
                    normal_accounts_pool.add(r.nameOrig)

    normal_candidates = list(normal_accounts_pool - fraud_accounts)
    sampled_normal_accounts = set(random.sample(normal_candidates, min(normal_sample_target, len(normal_candidates))))
    target_accounts = fraud_accounts | sampled_normal_accounts
    print(f"Target accounts: {len(fraud_accounts)} Fraud accounts, {len(sampled_normal_accounts)} Normal accounts.")

    print("Pass 2: Streaming full dataset through FeatureEngine...")
    for chunk in pd.read_csv(csv_path, chunksize=chunksize, usecols=["step", "type", "amount", "nameOrig", "nameDest"]):
        for r in chunk.itertuples(index=False):
            fe.update_transaction(
                sender=r.nameOrig,
                receiver=r.nameDest,
                amount=float(r.amount),
                timestamp=int(r.step)
            )

    print("Extracting feature vectors...")
    records = []
    for acc_id in target_accounts:
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

    df_full = pd.DataFrame(records)
    df_full.to_csv(FULL_CSV, index=False)

    # Stratified 80/20 Split
    train_df, test_df = train_test_split(
        df_full,
        test_size=0.20,
        random_state=seed,
        stratify=df_full["label"]
    )

    train_df.to_csv(TRAIN_CSV, index=False)
    test_df.to_csv(TEST_CSV, index=False)

    print(f"\n[DONE] Saved Datasets:")
    print(f"  • Full Dataset  : {FULL_CSV} ({len(df_full)} rows)")
    print(f"  • Train (80%)   : {TRAIN_CSV} ({len(train_df)} rows: {train_df['label'].value_counts().to_dict()})")
    print(f"  • Test (20%)    : {TEST_CSV} ({len(test_df)} rows: {test_df['label'].value_counts().to_dict()})")

    return train_df, test_df


if __name__ == "__main__":
    prepare_datasets()
