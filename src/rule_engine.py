from typing import Any
from feature_engine import AccountFeatures


class RuleEngine:
    def __init__(
        self,
        ratio_threshold: float = 0.95,
        delay_threshold: int = 30,
        fan_in_threshold: int = 20,
        velocity_threshold: int = 5,
    ):
        self.ratio_threshold = ratio_threshold
        self.delay_threshold = delay_threshold
        self.fan_in_threshold = fan_in_threshold
        self.velocity_threshold = velocity_threshold

    def evaluate(self, features: dict[str, Any] | AccountFeatures) -> dict[str, Any]:
        data = features.to_dict() if isinstance(features, AccountFeatures) else features

        reasons = []
        is_direct_high = False
        medium_triggers = 0

        # Rule 1: High Pass-Through Ratio (Direct HIGH)
        ratio = data.get("pass_through_ratio")
        if ratio is not None and ratio > self.ratio_threshold:
            is_direct_high = True
            reasons.append(f"pass_through_ratio ({ratio:.2f}) > {self.ratio_threshold}")

        # Rule 2: Rapid Forwarding Delay (MEDIUM)
        delay = data.get("forwarding_delay")
        if delay is not None and delay < self.delay_threshold:
            medium_triggers += 1
            reasons.append(f"forwarding_delay ({delay}s) < {self.delay_threshold}s")

        # Rule 3: High Fan-In (MEDIUM)
        fan_in = data.get("fan_in")
        if fan_in is not None and fan_in > self.fan_in_threshold:
            medium_triggers += 1
            reasons.append(f"fan_in ({fan_in}) > {self.fan_in_threshold}")

        # Rule 4: High Velocity (MEDIUM)
        velocity = data.get("velocity")
        if velocity is not None and velocity > self.velocity_threshold:
            medium_triggers += 1
            reasons.append(f"velocity ({velocity}) > {self.velocity_threshold}")

        # Numeric score computation (0 - 100)
        score = 0
        if is_direct_high:
            score += 70
        score += medium_triggers * 30
        score = min(100, score)

        # Level classification
        if is_direct_high or medium_triggers >= 2:
            risk_level = "HIGH"
        elif medium_triggers == 1:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"

        return {
            "risk_level": risk_level,
            "risk_score": score,
            "reasons": reasons,
            "triggered_count": (1 if is_direct_high else 0) + medium_triggers,
        }
