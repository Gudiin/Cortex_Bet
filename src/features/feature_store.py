from typing import Tuple
import pandas as pd
import warnings

# Core feature pipeline — shared with training and inference
from src.ml.features_v2 import create_advanced_features


class FeatureStore:
    """
    Central repository for Feature Engineering logic.
    Single Source of Truth for both Training and Inference.

    Architecture:
    - Training: ``get_training_features()`` — batch mode, full dataset.
    - Inference: ``build_match_features()`` (static) + ``get_inference_features()`` — single match.

    Design Pattern: Facade + Adapter
    """

    def __init__(self, db_manager):
        self.db = db_manager

    # ------------------------------------------------------------------
    # TRAINING (batch mode)
    # ------------------------------------------------------------------
    def get_training_features(
        self, df_history: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
        """
        Generates features for Model Training (Batch Mode).

        Args:
            df_history: Full historical dataset.

        Returns:
            X (Features), y (Target), timestamps (Validation Split)
        """
        X, y, timestamps, _ = create_advanced_features(df_history)
        return X, y, timestamps

    # ------------------------------------------------------------------
    # INFERENCE (single match) — SINGLE SOURCE OF TRUTH
    # ------------------------------------------------------------------
    @staticmethod
    def build_match_features(
        home_id: int,
        away_id: int,
        df_history: pd.DataFrame,
        window_long: int = 5,
    ) -> pd.DataFrame:
        """
        **Canonical** logic for inference feature generation.

        This is the ONLY place in the codebase that constructs the
        dummy-row pipeline for predicting a future match.  All other
        modules (``neural_engine``, legacy ``prepare_features_for_prediction``)
        MUST delegate here instead of duplicating the logic.

        Args:
            home_id:     ID of the home team.
            away_id:     ID of the away team.
            df_history:  Full historical DataFrame (already loaded from DB).
            window_long: Rolling window size for home/away specific averages.

        Returns:
            pd.DataFrame: Single-row DataFrame with model features *and*
                          display metrics merged in, ready for
                          ``model.predict()``.

        Raises:
            ValueError: If historical data is insufficient for a reliable
                        prediction (< 5 games per team).
        """
        # 1. Filter relevant history (both teams)
        relevant_games = df_history.loc[
            (df_history["home_team_id"] == home_id)
            | (df_history["away_team_id"] == home_id)
            | (df_history["home_team_id"] == away_id)
            | (df_history["away_team_id"] == away_id)
        ].copy()

        # 2. Winsorization — prevents 0-1 corner outliers from skewing EMAs
        if not relevant_games.empty:
            relevant_games["corners_home_ft"] = relevant_games[
                "corners_home_ft"
            ].clip(lower=3.0)
            relevant_games["corners_away_ft"] = relevant_games[
                "corners_away_ft"
            ].clip(lower=3.0)
            last_tournament_id = relevant_games["tournament_id"].iloc[-1]
        else:
            last_tournament_id = "Unknown"

        # 3. Dummy row — represents the future match to predict.
        # shift(1) inside create_advanced_features means the dummy's zeros
        # do NOT contaminate the computed rolling averages.
        import time

        future_timestamp = int(time.time()) + 86400  # +1 day (safe buffer)

        dummy_row = pd.DataFrame(
            [
                {
                    "match_id": 999999999,  # sentinel ID
                    "start_timestamp": future_timestamp,
                    "home_team_id": home_id,
                    "away_team_id": away_id,
                    "corners_home_ft": 0,
                    "corners_away_ft": 0,
                    "shots_ot_home_ft": 0,
                    "shots_ot_away_ft": 0,
                    "home_score": 0,
                    "away_score": 0,
                    "corners_home_ht": 0,
                    "corners_away_ht": 0,
                    "dangerous_attacks_home": 0,
                    "dangerous_attacks_away": 0,
                    "expected_goals_home": 0.0,
                    "expected_goals_away": 0.0,
                    "possession_home": 0,
                    "possession_away": 0,
                    "tournament_id": last_tournament_id,
                    "tournament_name": "Prediction",
                }
            ]
        )

        # 4. Run the full feature pipeline
        df_combined = pd.concat([relevant_games, dummy_row], ignore_index=True)
        X, _, _, df_display = create_advanced_features(
            df_combined, window_long=window_long
        )

        # 5. Extract last row (the dummy = the match to predict)
        features_single = X.iloc[[-1]].copy()
        display_single = df_display.iloc[[-1]].copy()

        # Attach display metrics for downstream UI consumption
        for col in display_single.columns:
            features_single[col] = display_single[col].values[0]

        # 6. Validate data sufficiency
        FeatureStore._validate_history_sufficiency(
            features_single, len(relevant_games), home_id, away_id, relevant_games
        )

        return features_single

    def get_inference_features(
        self, match_id: int, home_id: int, away_id: int
    ) -> pd.DataFrame:
        """
        Entry point for callers that own a ``db_manager`` but not a
        pre-loaded ``df_history``.

        Fetches history from DB then delegates to ``build_match_features()``.

        Returns:
            pd.DataFrame: Single-row feature DataFrame ready for
                          ``model.predict()``.
        """
        df = self.db.get_historical_data()
        return FeatureStore.build_match_features(home_id, away_id, df)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _validate_history_sufficiency(
        features: pd.DataFrame,
        n_games: int,
        home_id: int,
        away_id: int,
        history_df: pd.DataFrame,
    ):
        """
        Raises ``ValueError`` if not enough historical data exists for a
        scientifically valid prediction.
        """
        # Check if history is genuinely insufficient.
        # Previously we checked if home_avg_corners == 0, but with new EMA features
        # it computes a value even with just 1 or 2 games.
        # We must explicitly count games per team to enforce the requirement.
        h_count = len(
            history_df[
                (history_df["home_team_id"] == home_id)
                | (history_df["away_team_id"] == home_id)
            ]
        )
        a_count = len(
            history_df[
                (history_df["home_team_id"] == away_id)
                | (history_df["away_team_id"] == away_id)
            ]
        )

        if h_count < 5 or a_count < 5:
            h_count = len(
                history_df[
                    (history_df["home_team_id"] == home_id)
                    | (history_df["away_team_id"] == home_id)
                ]
            )
            a_count = len(
                history_df[
                    (history_df["home_team_id"] == away_id)
                    | (history_df["away_team_id"] == away_id)
                ]
            )

            msg = "Histórico insuficiente."
            if h_count < 5:
                msg += f" Mandante: {h_count} jogos."
            if a_count < 5:
                msg += f" Visitante: {a_count} jogos."

            raise ValueError(f"{msg} (Mínimo 5 jogos recentes).")
