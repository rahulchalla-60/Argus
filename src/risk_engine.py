import joblib
from pathlib import Path
from typing import Any
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)
import xgboost as xgb

DEFAULT_FEATURES = [
    "forwarding_delay", "velocity", "fan_in", "fan_out",
    "pass_through_ratio", "counterparties", "total_received", "total_sent"
]


class RiskEngine:
    def __init__(self, model_path: str | Path | None = None, features: list[str] | None = None):
        self.model = None
        self.features = list(features) if features else list(DEFAULT_FEATURES)
        self.model_type = None

        if model_path and Path(model_path).exists():
            self.load(model_path)

    def _check_model(self):
        if self.model is None:
            raise RuntimeError("Model is not trained or loaded. Call train() or load().")

    def _validate_data(self, df: pd.DataFrame, require_label: bool = True):
        missing = set(self.features) - set(df.columns)
        if missing:
            raise ValueError(f"Input data is missing required feature columns: {missing}")
        if require_label and "label" not in df.columns:
            raise ValueError("Input data is missing required target column: 'label'")

    def train(self, train_csv: str | Path, model_type: str = "xgboost") -> dict[str, Any]:
        df = pd.read_csv(train_csv)
        self._validate_data(df, require_label=True)

        X = df[self.features]
        y = df["label"]
        pos_count = int(y.sum())
        neg_count = len(y) - pos_count
        self.model_type = model_type

        if model_type == "xgboost":
            scale_pos_weight = (neg_count / pos_count) if pos_count > 0 else 1.0
            self.model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                scale_pos_weight=scale_pos_weight,
                n_jobs=-1,
                verbosity=0,
                random_state=42,
                eval_metric="logloss"
            )
        elif model_type == "random_forest":
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                class_weight="balanced",
                n_jobs=-1,
                random_state=42
            )
        else:
            raise ValueError(f"Unknown model_type: '{model_type}'. Expected 'xgboost' or 'random_forest'.")

        self.model.fit(X, y)
        importances = self.get_feature_importances()

        return {
            "model_type": model_type,
            "samples": len(df),
            "pos_samples": pos_count,
            "neg_samples": neg_count,
            "feature_importances": importances,
        }

    def evaluate(self, test_csv: str | Path) -> dict[str, Any]:
        self._check_model()
        df = pd.read_csv(test_csv)
        self._validate_data(df, require_label=True)

        X_test = df[self.features]
        y_test = df["label"]

        probs = self.model.predict_proba(X_test)[:, 1]
        preds = (probs >= 0.5).astype(int)

        cm = confusion_matrix(y_test, preds, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

        try:
            auc = float(roc_auc_score(y_test, probs))
        except ValueError:
            auc = None

        return {
            "accuracy": float(accuracy_score(y_test, preds)),
            "precision": float(precision_score(y_test, preds, zero_division=0)),
            "recall": float(recall_score(y_test, preds, zero_division=0)),
            "f1": float(f1_score(y_test, preds, zero_division=0)),
            "roc_auc": auc,
            "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        }

    def predict_risk(self, feature_dict: dict[str, Any]) -> float:
        self._check_model()
        clean_row = {}
        for f in self.features:
            val = feature_dict.get(f)
            if val is None:
                clean_row[f] = -1.0 if f == "forwarding_delay" else 0.0
            else:
                clean_row[f] = float(val)
        row = pd.DataFrame([clean_row])
        return float(self.model.predict_proba(row)[0][1])

    def score_account(
        self,
        account_id: str,
        feature_engine: Any,
        db: Any = None,
        step: int = 0
    ) -> float:
        self._check_model()
        feat_dict = feature_engine.get_features_dict(account_id)
        if not feat_dict:
            return 0.0

        score = self.predict_risk(feat_dict)

        if db:
            db.save_risk_score(
                account_id=account_id,
                risk_score=score,
                step=step,
                model_type=self.model_type or "xgboost"
            )

        return score

    def get_feature_importances(self) -> dict[str, float]:
        self._check_model()
        if not hasattr(self.model, "feature_importances_"):
            return {}
        return {f: float(imp) for f, imp in zip(self.features, self.model.feature_importances_)}

    def save(self, output_path: str | Path):
        self._check_model()
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        bundle = {
            "model": self.model,
            "features": self.features,
            "model_type": self.model_type,
        }
        joblib.dump(bundle, str(out))

    def load(self, model_path: str | Path):
        bundle = joblib.load(str(model_path))
        if isinstance(bundle, dict) and "model" in bundle:
            self.model = bundle["model"]
            self.features = bundle.get("features", DEFAULT_FEATURES)
            self.model_type = bundle.get("model_type")
        else:
            self.model = bundle
            self.features = DEFAULT_FEATURES
            self.model_type = "unknown"
