import sys
from pathlib import Path

# Add src to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

from risk_engine import RiskEngine

BASE_DIR = Path(__file__).resolve().parent.parent
TRAIN_CSV = BASE_DIR / "data" / "train_data.csv"
TEST_CSV = BASE_DIR / "data" / "test_data.csv"
MODELS_DIR = BASE_DIR / "models"


def test_risk_engine_uninitialized():
    re = RiskEngine()
    
    raised_predict = False
    try:
        re.predict_risk({"velocity": 1})
    except RuntimeError:
        raised_predict = True
    assert raised_predict, "Expected RuntimeError on uninitialized predict_risk"

    raised_eval = False
    try:
        re.evaluate(TEST_CSV)
    except RuntimeError:
        raised_eval = True
    assert raised_eval, "Expected RuntimeError on uninitialized evaluate"


def test_risk_engine_invalid_model_type():
    re = RiskEngine()
    raised = False
    try:
        re.train(TRAIN_CSV, model_type="invalid_model")
    except ValueError:
        raised = True
    assert raised, "Expected ValueError on unknown model_type"


def test_risk_engine_training_and_inference():
    re = RiskEngine()
    meta = re.train(TRAIN_CSV, model_type="xgboost")
    assert meta["model_type"] == "xgboost"
    assert meta["samples"] > 0
    assert len(meta["feature_importances"]) == 8

    # Evaluate on test set
    eval_res = re.evaluate(TEST_CSV)
    assert eval_res["accuracy"] > 0.80
    assert eval_res["roc_auc"] > 0.85
    assert "tn" in eval_res["confusion_matrix"]
    assert "tp" in eval_res["confusion_matrix"]

    # Predict single risk
    mule_feat = {
        "forwarding_delay": 8,
        "velocity": 5,
        "fan_in": 1,
        "fan_out": 1,
        "pass_through_ratio": 0.98,
        "counterparties": 2,
        "total_received": 1000000.0,
        "total_sent": 980000.0
    }
    score = re.predict_risk(mule_feat)
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0

    # Test Save & Load
    save_path = MODELS_DIR / "temp_test_model.pkl"
    re.save(save_path)
    assert save_path.exists()

    re_loaded = RiskEngine(save_path)
    score_loaded = re_loaded.predict_risk(mule_feat)
    assert abs(score - score_loaded) < 1e-5

    # Cleanup temp model
    if save_path.exists():
        save_path.unlink()

    print("[PASS] RiskEngine training, evaluation, saving, and inference.")


if __name__ == "__main__":
    test_risk_engine_uninitialized()
    test_risk_engine_invalid_model_type()
    test_risk_engine_training_and_inference()
    print("\nAll RiskEngine unit tests passed successfully.")
