from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler


@dataclass
class JointMetrics:
    mae_h1: float
    mae_a1: float
    mae_h2: float
    mae_a2: float


class JointCornersModel:
    """Joint multimarket model for segment corners [h1H, a1H, h2H, a2H]."""

    def __init__(self, model_dir: str = "models"):
        self.model_dir = Path(model_dir)
        self.model_path = self.model_dir / "joint_corners_model.joblib"
        self.scaler_path = self.model_dir / "joint_corners_scaler.joblib"
        self.calibration_path = self.model_dir / "joint_corners_calibration.joblib"

        self.model: MultiOutputRegressor | None = None
        self.scaler: StandardScaler | None = None
        self.calibration: Dict[str, float] = {"h1": 1.0, "a1": 1.0, "h2": 1.0, "a2": 1.0}

    @staticmethod
    def build_targets(df: pd.DataFrame, idx: pd.Index) -> pd.DataFrame:
        required = ["corners_home_ft", "corners_away_ft", "corners_home_ht", "corners_away_ht"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns for joint targets: {missing}")

        base = df.loc[idx, required].fillna(0).copy()
        y = pd.DataFrame(index=base.index)
        y["h1"] = base["corners_home_ht"].clip(lower=0)
        y["a1"] = base["corners_away_ht"].clip(lower=0)
        y["h2"] = (base["corners_home_ft"] - base["corners_home_ht"]).clip(lower=0)
        y["a2"] = (base["corners_away_ft"] - base["corners_away_ht"]).clip(lower=0)
        return y

    def train(self, X: pd.DataFrame, y: pd.DataFrame, timestamps: pd.Series | None = None) -> JointMetrics:
        valid_idx = X.dropna().index.intersection(y.dropna().index)
        Xv = X.loc[valid_idx]
        yv = y.loc[valid_idx]

        if timestamps is not None:
            ts = timestamps.loc[valid_idx]
            order = np.argsort(ts.values)
            Xv = Xv.iloc[order]
            yv = yv.iloc[order]

        X_train_raw, X_test_raw, y_train, y_test = train_test_split(
            Xv, yv, test_size=0.2, shuffle=False
        )

        self.scaler = StandardScaler()
        X_train = self.scaler.fit_transform(X_train_raw)
        X_test = self.scaler.transform(X_test_raw)

        base = HistGradientBoostingRegressor(max_depth=6, learning_rate=0.05, max_iter=300, random_state=42)
        self.model = MultiOutputRegressor(base)
        self.model.fit(X_train, y_train)

        y_pred = np.clip(self.model.predict(X_test), 0, None)

        # family calibration factors (simple scale correction by mean ratio)
        eps = 1e-6
        means_true = np.maximum(y_test.mean(axis=0).values, eps)
        means_pred = np.maximum(y_pred.mean(axis=0), eps)
        factors = means_true / means_pred
        self.calibration = {
            "h1": float(factors[0]),
            "a1": float(factors[1]),
            "h2": float(factors[2]),
            "a2": float(factors[3]),
        }

        y_cal = y_pred.copy()
        y_cal[:, 0] *= self.calibration["h1"]
        y_cal[:, 1] *= self.calibration["a1"]
        y_cal[:, 2] *= self.calibration["h2"]
        y_cal[:, 3] *= self.calibration["a2"]

        return JointMetrics(
            mae_h1=float(mean_absolute_error(y_test.iloc[:, 0], y_cal[:, 0])),
            mae_a1=float(mean_absolute_error(y_test.iloc[:, 1], y_cal[:, 1])),
            mae_h2=float(mean_absolute_error(y_test.iloc[:, 2], y_cal[:, 2])),
            mae_a2=float(mean_absolute_error(y_test.iloc[:, 3], y_cal[:, 3])),
        )

    def predict_segments(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.model is None or self.scaler is None:
            self.load()
        if self.model is None or self.scaler is None:
            raise RuntimeError("JointCornersModel is not trained")

        Xs = self.scaler.transform(X)
        pred = np.clip(self.model.predict(Xs), 0, None)
        pred[:, 0] *= self.calibration.get("h1", 1.0)
        pred[:, 1] *= self.calibration.get("a1", 1.0)
        pred[:, 2] *= self.calibration.get("h2", 1.0)
        pred[:, 3] *= self.calibration.get("a2", 1.0)

        return pd.DataFrame(pred, columns=["h1", "a1", "h2", "a2"], index=X.index)

    def predict_totals(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        seg = self.predict_segments(X)
        home_total = (seg["h1"] + seg["h2"]).values
        away_total = (seg["a1"] + seg["a2"]).values
        match_total = home_total + away_total
        return home_total, away_total, match_total

    def save(self) -> None:
        if self.model is None or self.scaler is None:
            raise RuntimeError("Cannot save before training")
        self.model_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, self.model_path)
        joblib.dump(self.scaler, self.scaler_path)
        joblib.dump(self.calibration, self.calibration_path)

    def load(self) -> bool:
        if not self.model_path.exists() or not self.scaler_path.exists():
            return False
        self.model = joblib.load(self.model_path)
        self.scaler = joblib.load(self.scaler_path)
        if self.calibration_path.exists():
            self.calibration = joblib.load(self.calibration_path)
        return True
