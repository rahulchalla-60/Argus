import time
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_predict
import xgboost as xgb

from risk_engine import RiskEngine
from risk_propagation import compute_neighbor_risk
from graph_engine import GraphEngine

BASE_DIR = Path(__file__).resolve().parent.parent
TRAIN_CSV = BASE_DIR / "data" / "train_data.csv"
TEST_CSV = BASE_DIR / "data" / "test_data.csv"
DATASET_PATH = BASE_DIR / "data" / "paysim.csv"
MODELS_DIR = BASE_DIR / "models"
TRAIN_AUG_CSV = BASE_DIR / "data" / "train_graph_augmented.csv"
TEST_AUG_CSV = BASE_DIR / "data" / "test_graph_augmented.csv"

BASELINE_FEATURES = [
    "forwarding_delay", "velocity", "fan_in", "fan_out",
    "pass_through_ratio", "counterparties", "total_received", "total_sent"
]

GRAPH_FEATURES = BASELINE_FEATURES + ["neighbor_risk"]


def build_graph_and_augment_oof() -> tuple[pd.DataFrame, pd.DataFrame]:
    print("Step 1: Building Graph Engine from PaySim transaction stream...")
    ge = GraphEngine(window_size=200000)

    # Ingest transactions through GraphEngine so full sliding-window and node metadata are updated
    for chunk in pd.read_csv(DATASET_PATH, chunksize=50000, nrows=200000, usecols=["step", "type", "amount", "nameOrig", "nameDest", "isFraud"]):
        for r in chunk.itertuples(index=False):
            ge.add_transaction(
                sender=r.nameOrig,
                receiver=r.nameDest,
                amount=float(r.amount),
                timestamp=int(r.step),
                type=r.type,
                isFraud=r.isFraud
            )

    train_df = pd.read_csv(TRAIN_CSV)
    test_df = pd.read_csv(TEST_CSV)

    X_train = train_df[BASELINE_FEATURES]
    y_train = train_df["label"]
    X_test = test_df[BASELINE_FEATURES]

    # Step 2: Generate Out-of-Fold (OOF) base probabilities on Train Set (Zero target leakage)
    print("Step 2: Generating Out-of-Fold (OOF) base probabilities via 5-Fold Stratified CV...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    pos_count = int(y_train.sum())
    neg_count = len(y_train) - pos_count
    scale_pos_weight = (neg_count / pos_count) if pos_count > 0 else 1.0

    base_model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        n_jobs=-1,
        verbosity=0,
        random_state=42,
        eval_metric="logloss"
    )

    oof_probs = cross_val_predict(
        base_model,
        X_train,
        y_train,
        cv=cv,
        method="predict_proba",
        n_jobs=-1
    )[:, 1]

    # Build account-level risk map for train
    train_account_scores = (
        train_df.assign(base_prob=oof_probs)
        .groupby("account_id")["base_prob"]
        .max()
        .to_dict()
    )

    # Fit base model on full train to predict on test
    base_model.fit(X_train, y_train)
    test_probs = base_model.predict_proba(X_test)[:, 1]
    test_account_scores = (
        test_df.assign(base_prob=test_probs)
        .groupby("account_id")["base_prob"]
        .max()
        .to_dict()
    )

    # Combined map for graph querying
    full_risk_map = {**train_account_scores, **test_account_scores}

    # Step 3: Compute neighbor_risk for train and test sets
    print("Step 3: Calculating neighbor_risk topological feature from graph...")
    train_df["neighbor_risk"] = [
        compute_neighbor_risk(ge.graph, acc, full_risk_map) for acc in train_df["account_id"]
    ]
    test_df["neighbor_risk"] = [
        compute_neighbor_risk(ge.graph, acc, full_risk_map) for acc in test_df["account_id"]
    ]

    train_df.to_csv(TRAIN_AUG_CSV, index=False)
    test_df.to_csv(TEST_AUG_CSV, index=False)

    print(f"Augmented datasets saved to:\n  • {TRAIN_AUG_CSV}\n  • {TEST_AUG_CSV}")
    return train_df, test_df


def benchmark_with_and_without_graph():
    print("=== Phase 4.2: Model Retraining & Leak-Free Graph Benchmark ===")
    
    build_graph_and_augment_oof()

    # 1. Baseline Model (8 features - No Neighbor Risk)
    print("\n--- Model A: Baseline XGBoost (8 Features) ---")
    base_engine = RiskEngine(features=BASELINE_FEATURES)
    base_engine.train(TRAIN_CSV, model_type="xgboost")
    base_eval = base_engine.evaluate(TEST_CSV)

    # 2. Graph-Enhanced Model (9 features - With Neighbor Risk)
    print("\n--- Model B: Graph-Enhanced XGBoost (9 Features: + neighbor_risk) ---")
    graph_engine_model = RiskEngine(features=GRAPH_FEATURES)
    graph_engine_model.train(TRAIN_AUG_CSV, model_type="xgboost")
    graph_eval = graph_engine_model.evaluate(TEST_AUG_CSV)

    # Save Graph-Enhanced model
    graph_engine_model.save(MODELS_DIR / "risk_model_graph_enhanced.pkl")
    graph_engine_model.save(MODELS_DIR / "risk_model.pkl")

    # 3. Print Side-by-Side Comparison Table
    print("\n" + "=" * 70)
    print(f"{'Metric':<22} | {'Baseline (8 Feat)':<20} | {'Graph-Enhanced (9 Feat)':<22}")
    print("-" * 70)
    print(f"{'Accuracy':<22} | {base_eval['accuracy'] * 100:.2f}%{'':<13} | {graph_eval['accuracy'] * 100:.2f}%{'':<15}")
    print(f"{'Precision':<22} | {base_eval['precision'] * 100:.2f}%{'':<13} | {graph_eval['precision'] * 100:.2f}%{'':<15}")
    print(f"{'Recall':<22} | {base_eval['recall'] * 100:.2f}%{'':<13} | {graph_eval['recall'] * 100:.2f}%{'':<15}")
    print(f"{'F1-Score':<22} | {base_eval['f1'] * 100:.2f}%{'':<13} | {graph_eval['f1'] * 100:.2f}%{'':<15}")
    auc_base = f"{base_eval['roc_auc']:.4f}" if base_eval['roc_auc'] is not None else "N/A"
    auc_graph = f"{graph_eval['roc_auc']:.4f}" if graph_eval['roc_auc'] is not None else "N/A"
    print(f"{'ROC-AUC':<22} | {auc_base:<20} | {auc_graph:<22}")
    print("=" * 70)

    print("\nConfusion Matrix (Baseline):      ", base_eval["confusion_matrix"])
    print("Confusion Matrix (Graph-Enhanced):", graph_eval["confusion_matrix"])

    print("\nFeature Importances (Graph-Enhanced Model):")
    for feat, imp in sorted(graph_engine_model.get_feature_importances().items(), key=lambda x: x[1], reverse=True):
        print(f"  {feat:22s}: {imp:.4f}")


if __name__ == "__main__":
    benchmark_with_and_without_graph()
