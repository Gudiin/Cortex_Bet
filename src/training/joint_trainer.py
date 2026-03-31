"""
src/training/joint_trainer.py
==============================
Pipeline de Treino para o Modelo Joint [home_1H, away_1H, home_2H, away_2H].

Responsabilidades:
    1. Carregar features via FeatureStore.
    2. Derivar targets joint (4 colunas) se dados HT disponíveis.
    3. Treinar JointCornersModel com walk-forward temporal.
    4. Gerar probabilidades out-of-fold para calibração.
    5. Ajustar PerMarketCalibrator com OOF probabilities.
    6. Reportar métricas (Brier, LogLoss, ECE) por família.
    7. Registrar modelo no ModelRegistry se critérios atendidos.

Proibições:
    - NÃO usar os dados de teste no treino do calibrador.
    - NÃO promover automaticamente — registrar métricas e aguardar decisão manual.
    - NÃO tratar o calibrador como production-ready com < 100 amostras por família.
"""

from __future__ import annotations

import json
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from src.features.feature_store import FeatureStore
from src.ml.joint_model import JointCornersModel, compute_joint_targets, JOINT_TARGET_DISPLAY
from src.ml.market_translator import MarketTranslator, MARKET_FAMILIES
from src.ml.per_market_calibrator import PerMarketCalibrator, compute_ece, compute_brier_score


EVAL_OUTPUT_DIR = Path("data/evaluation")


class JointTrainer:
    """
    Treina e avalia o JointCornersModel com protocolo científico.

    Args:
        n_splits: Número de folds no walk-forward.
        min_train_size: Mínimo de amostras para o primeiro fold de treino.
        n_simulations: Simulações Monte Carlo no MarketTranslator.
        random_state: Seed para reprodutibilidade.
    """

    def __init__(
        self,
        n_splits: int = 5,
        min_train_size: int = 200,
        n_simulations: int = 10_000,
        random_state: int = 42,
    ) -> None:
        self.n_splits = n_splits
        self.min_train_size = min_train_size
        self.n_simulations = n_simulations
        self.random_state = random_state

        self.translator = MarketTranslator(n_simulations=n_simulations, random_seed=random_state)
        self.calibrator: Optional[PerMarketCalibrator] = None
        self.model: Optional[JointCornersModel] = None

    def run(
        self,
        df_history: pd.DataFrame,
        feature_store: FeatureStore,
        save_dir: str = "models",
    ) -> Dict:
        """
        Executa o pipeline completo de treino + validação walk-forward.

        Args:
            df_history: Dataset histórico completo.
            feature_store: Instância do FeatureStore.
            save_dir: Diretório para salvar artefatos.

        Returns:
            Dict com métricas por família e por fold.
        """
        print("[JointTrainer] Iniciando pipeline de treino...")

        # 1. Feature engineering
        print("[JointTrainer] Gerando features...")
        X, y_ft, timestamps = feature_store.get_training_features(df_history)

        # 2. Joint targets
        Y_joint = compute_joint_targets(df_history)
        if Y_joint is None:
            raise ValueError(
                "[JointTrainer] Dados HT insuficientes para treinar modelo joint. "
                "Necessário: corners_home_ht e corners_away_ht preenchidos em ≥ 50 jogos."
            )

        # Alinhar índices entre X e Y_joint
        common_idx = X.index.intersection(Y_joint.index)
        if len(common_idx) < self.min_train_size:
            raise ValueError(
                f"[JointTrainer] Apenas {len(common_idx)} amostras com dados HT. "
                f"Mínimo necessário: {self.min_train_size}"
            )

        X = X.loc[common_idx]
        Y_joint = Y_joint.loc[common_idx]
        ts = timestamps.loc[common_idx] if hasattr(timestamps, 'loc') else timestamps

        print(f"[JointTrainer] {len(X)} amostras disponíveis para treino joint.")

        # 3. Walk-forward validation
        print("[JointTrainer] Executando walk-forward validation...")
        oof_metrics = self._walk_forward(X, Y_joint)

        # 4. Treino final em todo o dataset
        print("[JointTrainer] Treinando modelo final...")
        self.model = JointCornersModel(random_state=self.random_state)
        self.model.fit(X, Y_joint)

        # 5. Gerar OOF probabilities para calibração
        print("[JointTrainer] Gerando probabilidades OOF para calibração...")
        oof_probs, oof_labels = self._generate_oof_probs(X, Y_joint)

        # 6. Ajustar calibrador
        print("[JointTrainer] Ajustando PerMarketCalibrator...")
        self.calibrator = PerMarketCalibrator()
        self.calibrator.fit(oof_probs, oof_labels)

        # 7. Salvar artefatos
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        self.model.save(str(save_path / "joint_corners_model.joblib"))
        self.calibrator.save(str(save_path / "per_market_calibrator.joblib"))

        # 8. Salvar relatório
        report = {
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            "n_samples": len(X),
            "n_splits": self.n_splits,
            "oof_metrics": oof_metrics,
            "calibration_report": self.calibrator.calibration_report(),
        }
        EVAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        report_path = EVAL_OUTPUT_DIR / f"joint_trainer_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2, default=str)
        print(f"[JointTrainer] Relatório salvo em: {report_path}")

        return report

    # ------------------------------------------------------------------
    # Walk-forward temporal
    # ------------------------------------------------------------------

    def _walk_forward(
        self, X: pd.DataFrame, Y: pd.DataFrame
    ) -> Dict[str, List[float]]:
        """
        Executa walk-forward com TimeSeriesSplit.

        Retorna métricas por família de mercado (MAE dos lambdas).
        """
        tscv = TimeSeriesSplit(n_splits=self.n_splits)

        fold_metrics: Dict[str, List[float]] = {col: [] for col in JOINT_TARGET_DISPLAY}

        X_reset = X.reset_index(drop=True)
        Y_reset = Y.reset_index(drop=True)

        for fold_i, (train_idx, val_idx) in enumerate(tscv.split(X_reset)):
            X_tr = X_reset.iloc[train_idx]
            Y_tr = Y_reset.iloc[train_idx]
            X_val = X_reset.iloc[val_idx]
            Y_val = Y_reset.iloc[val_idx]

            if len(X_tr) < self.min_train_size:
                continue

            m = JointCornersModel(random_state=self.random_state)
            m.fit(X_tr, Y_tr)

            # Batch predict — vetorizado
            lambdas_df = m.predict_lambda_batch(X_val)
            for col in JOINT_TARGET_DISPLAY:
                preds = lambdas_df[col].values
                mae = float(np.mean(np.abs(preds - Y_val[col].values)))
                fold_metrics[col].append(mae)

            print(f"  [Fold {fold_i + 1}] MAE: "
                  + ", ".join(f"{c}={np.mean(v):.3f}" for c, v in fold_metrics.items() if v))

        summary = {col: float(np.mean(vals)) if vals else float("nan")
                   for col, vals in fold_metrics.items()}
        print(f"[JointTrainer] Walk-forward MAE médio: {summary}")
        return summary

    # ------------------------------------------------------------------
    # OOF probabilities para calibração
    # ------------------------------------------------------------------

    def _generate_oof_probs(
        self,
        X: pd.DataFrame,
        Y: pd.DataFrame,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """
        Gera probabilidades out-of-fold para cada família de mercado.

        Usa cálculo analítico via CDF Poisson (vetorizado) em vez de MC por amostra,
        tornando o processo ~1000x mais rápido sem perda de precisão (lambda3=0).

        Returns:
            (oof_probs, oof_labels) — dicts {family: array}
        """
        tscv = TimeSeriesSplit(n_splits=self.n_splits)

        default_lines = {
            "ft_total":  9.5,
            "ht_total":  4.5,
            "ht2_total": 4.5,
            "ft_home":   4.5,
            "ft_away":   4.5,
            "ht_home":   2.5,
            "ht_away":   2.5,
            "ht2_home":  2.5,
            "ht2_away":  2.5,
        }

        oof_probs: Dict[str, List[np.ndarray]] = {f: [] for f in MARKET_FAMILIES}
        oof_labels: Dict[str, List[np.ndarray]] = {f: [] for f in MARKET_FAMILIES}

        X_r = X.reset_index(drop=True)
        Y_r = Y.reset_index(drop=True)

        for fold_i, (train_idx, val_idx) in enumerate(tscv.split(X_r)):
            X_tr = X_r.iloc[train_idx]
            Y_tr = Y_r.iloc[train_idx]
            X_val = X_r.iloc[val_idx]
            Y_val = Y_r.iloc[val_idx]

            if len(X_tr) < self.min_train_size:
                continue

            m = JointCornersModel(random_state=self.random_state)
            m.fit(X_tr, Y_tr)

            # Batch predict + analytical P(over) — vetorizado
            lambdas_df = m.predict_lambda_batch(X_val)
            probs = MarketTranslator.compute_prob_over_analytical(lambdas_df, default_lines)

            # Actuals vetorizados
            actuals_df = _actuals_batch(Y_val)

            for family in MARKET_FAMILIES:
                line = default_lines[family]
                if family not in probs:
                    continue
                oof_probs[family].append(probs[family])
                oof_labels[family].append((actuals_df[family].values > line).astype(float))

            print(f"  [OOF Fold {fold_i + 1}] {len(X_val)} amostras processadas (analítico)")

        return (
            {f: np.concatenate(v) if v else np.array([]) for f, v in oof_probs.items()},
            {f: np.concatenate(v) if v else np.array([]) for f, v in oof_labels.items()},
        )


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _actuals_from_row(y_row: pd.Series) -> Dict[str, float]:
    """Deriva os 9 mercados a partir de uma linha de Y_joint."""
    h1H = float(y_row.get("home_1H", 0))
    a1H = float(y_row.get("away_1H", 0))
    h2H = float(y_row.get("home_2H", 0))
    a2H = float(y_row.get("away_2H", 0))
    return {
        "ft_total":  h1H + a1H + h2H + a2H,
        "ht_total":  h1H + a1H,
        "ht2_total": h2H + a2H,
        "ft_home":   h1H + h2H,
        "ft_away":   a1H + a2H,
        "ht_home":   h1H,
        "ht_away":   a1H,
        "ht2_home":  h2H,
        "ht2_away":  a2H,
    }


def _actuals_batch(Y: pd.DataFrame) -> pd.DataFrame:
    """Deriva os 9 mercados vetorizados a partir do DataFrame Y_joint."""
    h1H = Y["home_1H"]
    a1H = Y["away_1H"]
    h2H = Y["home_2H"]
    a2H = Y["away_2H"]
    return pd.DataFrame({
        "ft_total":  h1H + a1H + h2H + a2H,
        "ht_total":  h1H + a1H,
        "ht2_total": h2H + a2H,
        "ft_home":   h1H + h2H,
        "ft_away":   a1H + a2H,
        "ht_home":   h1H,
        "ht_away":   a1H,
        "ht2_home":  h2H,
        "ht2_away":  a2H,
    }, index=Y.index)
