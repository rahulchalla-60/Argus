import time
from pathlib import Path
from risk_engine import RiskEngine

BASE_DIR = Path(__file__).resolve().parent.parent
TRAIN_CSV = BASE_DIR / "data" / "train_data.csv"
TEST_CSV = BASE_DIR / "data" / "test_data.csv"
MODELS_DIR = BASE_DIR / "models"


def train_and_compare_models():
    print("=== Phase 3.3: ML Model Training & Comparative Benchmark ===")
    
    # 1. Train XGBoost
    print("\n--- Training Model 1: XGBoost ---")
    t0 = time.perf_counter()
    xgb_engine = RiskEngine()
    xgb_meta = xgb_engine.train(TRAIN_CSV, model_type="xgboost")
    xgb_train_time = time.perf_counter() - t0
    xgb_eval = xgb_engine.evaluate(TEST_CSV)
    
    xgb_path = MODELS_DIR / "risk_model_xgboost.pkl"
    default_path = MODELS_DIR / "risk_model.pkl"
    xgb_engine.save(xgb_path)
    xgb_engine.save(default_path)
    print(f"XGBoost trained in {xgb_train_time:.2f}s and saved to {xgb_path}")

    # 2. Train Random Forest
    print("\n--- Training Model 2: Random Forest ---")
    t0 = time.perf_counter()
    rf_engine = RiskEngine()
    rf_meta = rf_engine.train(TRAIN_CSV, model_type="random_forest")
    rf_train_time = time.perf_counter() - t0
    rf_eval = rf_engine.evaluate(TEST_CSV)

    rf_path = MODELS_DIR / "risk_model_random_forest.pkl"
    rf_engine.save(rf_path)
    print(f"Random Forest trained in {rf_train_time:.2f}s and saved to {rf_path}")

    # 3. Print Comparison Table
    print("\n" + "=" * 65)
    print(f"{'Metric':<22} | {'XGBoost':<18} | {'Random Forest':<18}")
    print("-" * 65)
    print(f"{'Training Time':<22} | {xgb_train_time:.3f}s{'':<12} | {rf_train_time:.3f}s{'':<12}")
    print(f"{'Accuracy':<22} | {xgb_eval['accuracy'] * 100:.2f}%{'':<11} | {rf_eval['accuracy'] * 100:.2f}%{'':<11}")
    print(f"{'Precision':<22} | {xgb_eval['precision'] * 100:.2f}%{'':<11} | {rf_eval['precision'] * 100:.2f}%{'':<11}")
    print(f"{'Recall':<22} | {xgb_eval['recall'] * 100:.2f}%{'':<11} | {rf_eval['recall'] * 100:.2f}%{'':<11}")
    print(f"{'F1-Score':<22} | {xgb_eval['f1'] * 100:.2f}%{'':<11} | {rf_eval['f1'] * 100:.2f}%{'':<11}")
    auc_xgb_str = f"{xgb_eval['roc_auc']:.4f}" if xgb_eval['roc_auc'] is not None else "N/A"
    auc_rf_str = f"{rf_eval['roc_auc']:.4f}" if rf_eval['roc_auc'] is not None else "N/A"
    print(f"{'ROC-AUC':<22} | {auc_xgb_str:<18} | {auc_rf_str:<18}")
    print("=" * 65)

    print("\nConfusion Matrix (XGBoost):", xgb_eval["confusion_matrix"])
    print("Confusion Matrix (Random Forest):", rf_eval["confusion_matrix"])

    print("\nFeature Importances (XGBoost):")
    for feat, imp in sorted(xgb_meta["feature_importances"].items(), key=lambda x: x[1], reverse=True):
        print(f"  {feat:22s}: {imp:.4f}")

    print("\nFeature Importances (Random Forest):")
    for feat, imp in sorted(rf_meta["feature_importances"].items(), key=lambda x: x[1], reverse=True):
        print(f"  {feat:22s}: {imp:.4f}")


if __name__ == "__main__":
    train_and_compare_models()
