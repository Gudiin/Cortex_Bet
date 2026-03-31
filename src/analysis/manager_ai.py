"""
ManagerAI - Orquestrador Central de Previsões.

Regra de Negócio:
    Unifica JointCornersModel (champion), NeuralChallenger (shadow)
    e StatisticalAnalyzer em um pipeline de previsão único.

    O JointCornersModel é o modelo de produção — 4 lambdas (h1H, a1H, h2H, a2H)
    com coerência HT/FT garantida por construção.

CHAMPION_ONLY_MODE (refactor/multimercado-cientifico):
    Quando CHAMPION_ONLY_MODE=True (default), o output final usa EXCLUSIVAMENTE
    o champion. O challenger continua sendo executado e logado em shadow mode,
    mas NÃO contribui para o output de produção.

    Variável de ambiente: CHAMPION_ONLY_MODE=0 desativa (modo legado).
    Ver: docs/audit/04_plano_migracao.md — Fase 1.
"""

from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np
import os
import joblib
from pathlib import Path

# ---------------------------------------------------------------------------
# Feature flag — Fase 1 do plano de migração multimercado
# CHAMPION_ONLY_MODE=True => output = champion_conf (sem blend com challenger)
# CHAMPION_ONLY_MODE=False => comportamento legado (blend 0.5/0.5)
# ---------------------------------------------------------------------------
_env_flag = os.environ.get("CHAMPION_ONLY_MODE", "1").strip().lower()
CHAMPION_ONLY_MODE: bool = _env_flag not in ("0", "false", "no")

# Domain Models (Single Source of Truth)
from src.domain.models import PredictionResult

# Components
from src.features.feature_store import FeatureStore
from src.models.model_v2 import ProfessionalPredictor
from src.models.neural_engine import NeuralChallenger
from src.models.model_registry import ModelRegistry
from src.analysis.statistical import StatisticalAnalyzer
from src.ml.calibration import MultiThresholdCalibrator
from src.ml.joint_model import JointCornersModel

class ManagerAI:
    """
    Central Orchestrator (The 'Manager').
    Unifies ProfessionalPredictor (Ensemble), NeuralChallenger (MLP), and StatisticalAnalyzer.
    """
    
    def __init__(self, db_manager):
        self.db_manager = db_manager
        
        # 1. Initialize Components
        self.feature_store = FeatureStore(db_manager)
        self.ensemble = ProfessionalPredictor()
        self.neural = NeuralChallenger()
        self.statistical = StatisticalAnalyzer()
        
        # 2. Load Models (Joint first — it's the champion)
        self.joint_model = self._load_joint_model()
        self._load_ensemble()
        # Neural loads itself in __init__
        
        # 3. Load Calibrator
        self.calibrator = self._load_calibrator()

        # 4. Resolve runtime roles from registry (with safe fallback)
        self.registry = self._load_registry()
        self.runtime_champion_id = "joint_corners_v1"
        self.runtime_challenger_id = "neural_challenger_v1"
        self._resolve_runtime_model_roles()
        
    def _load_ensemble(self):
        import warnings as _w
        try:
            with _w.catch_warnings():
                _w.filterwarnings("ignore", message=".*serialized model.*", category=UserWarning)
                _w.filterwarnings("ignore", message=".*older version.*", category=UserWarning)
                self.ensemble.load_model()
            print("ManagerAI: Ensemble Loaded.")
        except Exception as e:
            print(f"ManagerAI: Ensemble load failed: {e}")

    def _load_joint_model(self):
        """Load JointCornersModel (4-lambda champion) from artifact."""
        import warnings as _w
        joint_path = Path('models/joint_corners_model.joblib')
        if not joint_path.exists():
            print("ManagerAI: JointCornersModel not found. Falling back to legacy models.")
            return None
        try:
            with _w.catch_warnings():
                _w.filterwarnings("ignore", message=".*serialized model.*", category=UserWarning)
                _w.filterwarnings("ignore", message=".*older version.*", category=UserWarning)
                model = JointCornersModel.load(str(joint_path))
            print(f"ManagerAI: JointCornersModel loaded ({len(model.feature_names_)} features).")
            return model
        except Exception as e:
            print(f"ManagerAI: JointCornersModel load failed: {e}")
            return None

    def _load_calibrator(self):
        try:
            calibrator_path = Path('data/calibrator_temperature.pkl')
            if calibrator_path.exists():
                return joblib.load(calibrator_path)
            else:
                return None
        except Exception as e:
            print(f"ManagerAI: Calibrator load warning: {e}")
            return None

    def _load_registry(self):
        """Load model registry when available; fallback to legacy fixed roles on failure."""
        try:
            return ModelRegistry()
        except Exception as e:
            print(f"ManagerAI: Registry load warning: {e}")
            return None

    def _resolve_runtime_model_roles(self):
        """
        Resolve active champion/challenger model IDs from registry.

        Fallback defaults preserve legacy behavior:
            champion=ensemble_v1, challenger=neural_challenger_v1
        """
        if self.registry is None:
            print("ManagerAI: Registry unavailable. Using legacy runtime roles.")
            return

        try:
            roles = self.registry.get_runtime_roles()
            champion_id = roles["champion_id"]
            challenger_id = roles["challenger_id"]

            self.runtime_champion_id = champion_id
            self.runtime_challenger_id = challenger_id
            print(
                "ManagerAI: Runtime roles loaded from registry "
                f"(champion={self.runtime_champion_id}, challenger={self.runtime_challenger_id})."
            )
        except Exception as e:
            print(f"ManagerAI: Registry role resolution warning: {e}")

    def _model_total_from_id(self, model_id: str, features_vector: pd.DataFrame) -> float:
        """Return total corner expectation for a model ID using configurable runtime adapters."""
        adapter = None
        if self.registry is not None:
            try:
                adapter = self.registry.get_runtime_adapter(model_id)
            except Exception:
                adapter = None

        # Backward-compatible fallback when registry adapter is unavailable.
        if adapter is None:
            if "joint" in model_id.lower():
                adapter = "joint"
            elif "neural" in model_id.lower():
                adapter = "neural"
            else:
                adapter = "ensemble"

        if adapter == "joint" and self.joint_model is not None:
            lambdas = self.joint_model.predict_lambda(features_vector)
            return float(lambdas["ft_total"])

        if adapter == "neural":
            l_home, l_away = self.neural.predict_lambda(features_vector)
            return float(l_home + l_away)

        # Default adapter is ensemble.
        return float(self.ensemble.predict(features_vector)[0])

    def predict_match(self, match_id: int, match_metadata: Optional[Dict[str, Any]] = None) -> PredictionResult:
        """
        Executes full prediction pipeline for a single match.
        Args:
            match_id: ID of the match.
            match_metadata: Optional dictionary with match details.
        """
        # 1. Get Match Metadata
        # We need basic info to call feature store
        
        # For robustness, we'll fetch full historical df once.
        df_history = self.db_manager.get_historical_data() 
        
        if match_metadata:
            # Use injected metadata (Fast Path for Scanner/Future Games)
            home_id = match_metadata['home_id']
            away_id = match_metadata['away_id']
            start_ts = match_metadata['timestamp']
            tourn_id = match_metadata['tournament_id']
            home_name = match_metadata['home_name']
            away_name = match_metadata['away_name']
        else:
            # Legacy/Fallback lookup in history (Slow Path)
            row = df_history[df_history['match_id'] == match_id]
            if row.empty:
                raise ValueError(f"Match {match_id} not found in history and no metadata provided.")
            
            match_data_row = row.iloc[0]
            home_id = match_data_row['home_team_id']
            away_id = match_data_row['away_team_id']
            start_ts = match_data_row['start_timestamp']
            tourn_id = match_data_row.get('tournament_id', 0)
            home_name = str(home_id) # Placeholder
            away_name = str(away_id)
        
        # 2. Feature Generation (Cached)
        # 2. Feature Generation (Cached)
        features_vector = self.feature_store.get_inference_features(
            match_id, home_id, away_id
        )
        
        # 3. Raw per-model inference (kept for compatibility fields in PredictionResult)
        ensemble_raw = self._model_total_from_id("ensemble_v1", features_vector)
        neural_total = self._model_total_from_id("neural_challenger_v1", features_vector)

        # 4. Registry-driven runtime roles (effective champion/challenger in inference path)
        champion_raw = self._model_total_from_id(self.runtime_champion_id, features_vector)
        challenger_raw = self._model_total_from_id(self.runtime_challenger_id, features_vector)

        # Business Logic: Determine Best Line (Dynamic Value Analysis)
        line_val, pick, is_over, prob_score = self._find_best_line(
            projected_mu=champion_raw,
            neural_mu=challenger_raw
        )
        
        # 5. Calibration
        ensemble_conf = self._get_confidence(ensemble_raw, line_val, is_over)
        champion_conf = self._get_confidence(champion_raw, line_val, is_over)
        
        # Neural Confidence (Probabilistic)
        from scipy.stats import poisson
        if is_over:
            neural_conf = 1 - poisson.cdf(int(line_val), neural_total)
            challenger_conf = 1 - poisson.cdf(int(line_val), challenger_raw)
        else:
            neural_conf = poisson.cdf(int(line_val), neural_total)
            challenger_conf = poisson.cdf(int(line_val), challenger_raw)
            
        # 6. Consensus & Final Decision
        # -----------------------------------------------------------------------
        # CHAMPION_ONLY_MODE (Fase 1 — refactor/multimercado-cientifico):
        #   True  => output = champion_conf (sem blend — 1 champion oficial)
        #   False => comportamento legado (blend champion 0.5 + prob_score 0.5)
        # O challenger continua calculado para shadow logging independente do modo.
        # -----------------------------------------------------------------------
        if CHAMPION_ONLY_MODE:
            final_conf = min(0.95, champion_conf)
            agreement = 1.0  # não altera output; mantido para log
        else:
            # Legado: blend com fator de concordância
            agreement = 1.0
            if (is_over and challenger_raw > line_val) or \
               (not is_over and challenger_raw < line_val):
                agreement = 1.1
            final_conf = (champion_conf * 0.5 + prob_score * 0.5) * agreement
            final_conf = min(0.95, final_conf)
        
        # P2-A FIX: Cálculo de Odd Justa (Fair Odds)
        # O usuário prefere prever o valor real da odd (Odd Justa) em vez de fixar 
        # um EV irreal baseado em 1.95 estático.
        fair_odds = round((1 / final_conf), 2) if final_conf > 0 else 0.0
        
        # EV real só existe se houver uma Odd da Casa > Fair Odds. 
        # Como não temos API, cravamos como 0.0 (EV Teórico nulo em Odd Justa)
        ev = 0.0
        # 7. Statistical Analyzer (Legacy Restoration)
        # --------------------------------------------
        # Prepare Dataframes for Statistical Engine
        # It expects df_home and df_away. accurate history.
        h_history = df_history[df_history['home_team_id'] == home_id].sort_values('start_timestamp')
        a_history = df_history[df_history['away_team_id'] == away_id].sort_values('start_timestamp')
        
        # Neural Params for Hybrid Mode (Dynamic Calculation)
        # JointCornersModel provides 4 lambdas with HT/FT coherence.
        # Falls back to NeuralChallenger if Joint is unavailable.
        if self.joint_model is not None:
            joint_lambdas = self.joint_model.predict_lambda(features_vector)
            neural_dist_params = {
                'lambda_home': joint_lambdas['home_ft'],
                'lambda_away': joint_lambdas['away_ft'],
                'lambda_home_1h': joint_lambdas['home_1H'],
                'lambda_away_1h': joint_lambdas['away_1H'],
                'lambda_home_2h': joint_lambdas['home_2H'],
                'lambda_away_2h': joint_lambdas['away_2H'],
                'variance_factor': 1.0,
                'source': 'joint_corners_v1',
            }
        else:
            neural_dist_params = self.neural.get_neural_distributions(
                match_stats={'home_id': home_id, 'away_id': away_id, 'tournament_id': tourn_id},
                df_history=df_history
            )
        
        # Override with current inference if available (get_neural_distributions re-runs inference, 
        # but we already have neural_home/neural_away from step 4. 
        # Ideally we trust get_neural_distributions for the full params including variance.)
        
        # Audit Fix 4.1: Bivariate Poisson Wiring
        # Compute the historical covariance (lambda3) between home/away corners.
        # This is passed to StatisticalAnalyzer.simulate_match_event() which activates
        # the Bivariate Poisson model (Karlis & Ntzoufras, 2009) when lambda3 > 0.1.
        lambda3 = 0.0
        try:
            lambda3 = self.statistical.calculate_covariance(h_history, a_history)
        except Exception:
            pass

        advanced_metrics = {
            'neural_params': neural_dist_params,
            'lambda3': lambda3  # Bivariate Poisson coupling term
        }
        
        # Execute Analysis
        market_opportunities = []
        suggestions = {}
        
        try:
            # We pass match_name to trigger internal printing of the Legacy Tables
            # Returns: top_picks (List), suggestions (Dict), tactical_metrics (Dict)
            market_opportunities, suggestions, tactical_data = self.statistical.analyze_match(
                df_home=h_history,
                df_away=a_history,
                ml_prediction=champion_raw,
                match_name=f"{home_name} vs {away_name}",
                advanced_metrics=advanced_metrics
            )
            
        except Exception as e:
            print(f"Statistical Engine Error: {e}")

        # Text Feedback
        feedback = (
            f"Champion({self.runtime_champion_id}): {champion_raw:.1f} | "
            f"Challenger({self.runtime_challenger_id}): {challenger_raw:.1f}\n"
            f"Ensemble: {ensemble_raw:.1f} | Neural: {neural_total:.1f} | "
            f"Consensus: {'High' if agreement > 1 else 'Normal'}"
        )
        
        # [DEBUG PREDICT] - Restored
        print(f"\n[DEBUG PREDICT] Match: {home_name} vs {away_name}")
        print(f"[DEBUG PREDICT] Champion ID:      {self.runtime_champion_id}")
        print(f"[DEBUG PREDICT] Challenger ID:    {self.runtime_challenger_id}")
        if self.joint_model is not None:
            _jl = self.joint_model.predict_lambda(features_vector)
            print(f"[DEBUG PREDICT] Joint λ(h1H={_jl['home_1H']:.2f} a1H={_jl['away_1H']:.2f} "
                  f"h2H={_jl['home_2H']:.2f} a2H={_jl['away_2H']:.2f}) → FT={_jl['ft_total']:.2f}")
        print(f"[DEBUG PREDICT] Neural Output:   {neural_total:.4f}")
        print(f"[DEBUG PREDICT] Champion Output: {champion_raw:.4f}")
        print(f"[DEBUG PREDICT] Challenger Out:  {challenger_raw:.4f}")
        print(f"[DEBUG PREDICT] Line Selected:   {line_val} ({pick})")
        print(f"[DEBUG PREDICT] Confidence:      {final_conf:.1%}")
        print(f"[DEBUG PREDICT] CHAMPION_ONLY_MODE: {CHAMPION_ONLY_MODE}")

        # Shadow log do challenger (sempre registrado, nunca entra no output)
        self._log_shadow_challenger(
            match_id=match_id,
            champion_raw=champion_raw,
            challenger_raw=challenger_raw,
            champion_conf=champion_conf,
            challenger_conf=challenger_conf,
            line_val=line_val,
            is_over=is_over,
        )
        
        return PredictionResult(
            match_id=match_id,
            home_team=home_name,
            away_team=away_name,
            final_prediction=ensemble_raw,
            line_val=line_val,
            best_bet=f"{pick} {line_val}",
            is_over=is_over,
            ensemble_confidence=ensemble_conf,
            neural_confidence=neural_conf,
            consensus_confidence=final_conf,
            ensemble_raw=ensemble_raw,
            neural_raw=neural_total,
            fair_odds=fair_odds,
            ev_percentage=ev,
            features=features_vector,
            feedback_text=feedback,
            alternative_markets=market_opportunities, # Populated
            suggestions=suggestions # Populated
        )
        
    def _find_best_line(self, projected_mu: float, neural_mu: float) -> Tuple[float, str, bool, float]:
        """
        Determines the best market line based on Value and Probability.
        Fixes the 'Under' bias by explicitly checking for 'Safety Overs' (Value Bet).
        
        Returns:
            Tuple: (line_val, pick_type, is_over, probability)
        """
        import math
        from scipy.stats import poisson
        
        # Weighted projection for line selection
        # We trust the Ensemble for the 'center' but Neural for 'tails'
        weighted_mu = (projected_mu * 0.6 + neural_mu * 0.4)
        
        candidates = []
        
        # 1. Standard Line (Round closest) - The "Market Proxy"
        # e.g. 9.2 -> 9.5 | 9.7 -> 10.5
        standard_line = round(weighted_mu) + 0.5
        candidates.append({'line': standard_line, 'type': 'Standard'})
        
        # 2. Floor Line (Aggressive Over)
        # e.g. 9.8 -> 9.5
        floor_line = math.floor(weighted_mu) + 0.5
        candidates.append({'line': floor_line, 'type': 'Floor'})
        
        # 3. Safety Over (Value Bet) 
        # User Request: If pred is 9.2 (Under 10.5), check Over 7.5?
        # e.g. 9.2 -> Floor 9.5 -> Safety 7.5 or 8.5. 
        # Let's check 1.0 and 2.0 lines below floor.
        candidates.append({'line': floor_line - 1.0, 'type': 'Safety_1'})
        candidates.append({'line': floor_line - 2.0, 'type': 'Safety_2'})
        
        # 4. Ceiling Line (Aggressive Under)
        candidates.append({'line': math.ceil(weighted_mu) + 0.5, 'type': 'Ceil'})
        
        best_candidate = None
        best_score = -1.0 # Score = Probability * Utility
        
        for cand in candidates:
            line = cand['line']
            if line < 0.5: continue
            
            # Calculate Probabilities using Poisson
            # Over X.5 -> X >= Int(X.5) + 1. e.g. Over 9.5 -> >= 10
            k_over = int(line - 0.5) 
            prob_over = 1.0 - poisson.cdf(k_over, weighted_mu)
            
            # Under X.5 -> X <= Int(X.5). e.g. Under 9.5 -> <= 9
            k_under = int(line - 0.5)
            prob_under = poisson.cdf(k_under, weighted_mu)
            
            # Evaluate "Over"
            # We prefer Overs if Prob > 65% (Value)
            score_over = prob_over
            # Penalty for very low lines (odds too low, huge juice)
            if prob_over > 0.90: score_over *= 0.8  # Diminishing returns for ultra-safe lines (1.05 odds)
            
            if score_over > best_score:
                best_score = score_over
                best_candidate = (line, "Over", True, prob_over)
                
            # Evaluate "Under"
            # We prefer Unders if Prob > 60%
            score_under = prob_under
            if prob_under > 0.90: score_under *= 0.85
            
            if score_under > best_score:
                best_score = score_under
                best_candidate = (line, "Under", False, prob_under)
        
        # Fallback
        if not best_candidate:
            return standard_line, "Under", False, 0.5
            
        return best_candidate

    def _get_confidence(self, raw_val: float, line_val: float, is_over: bool) -> float:
        """Calculates calibrated probability."""
        prob = 0.5
        if self.calibrator and hasattr(self.calibrator, 'predict_proba'):
            # Check calibration signature
            try:
                # Assuming MultiThresholdCalibrator
                 prob = self.calibrator.predict_proba(raw_val, threshold=line_val, use_poisson=True)
            except:
                # Fallback TemperatureScaling (takes logits)
                # Raw val is usually logit-like in regression? No, it's counts.
                # Temperature Scaling usually works on classification logits.
                # For Regression: Poisson survival function is better.
                prob = self._poisson_prob(raw_val, line_val)
        else:
             prob = self._poisson_prob(raw_val, line_val)
             
        return prob if is_over else (1.0 - prob)

    def _poisson_prob(self, mu, k):
        from scipy.stats import poisson
        # P(X > k)
        return 1 - poisson.cdf(k, mu)

    def _log_shadow_challenger(
        self,
        match_id: int,
        champion_raw: float,
        challenger_raw: float,
        champion_conf: float,
        challenger_conf: float,
        line_val: float,
        is_over: bool,
    ) -> None:
        """
        Registra a predição do challenger em shadow mode.

        O challenger nunca entra no output de produção (veja CHAMPION_ONLY_MODE).
        Este log é usado para walk-forward comparativo offline.
        Falha silenciosa — nunca deve bloquear o output do champion.
        """
        try:
            import json
            from datetime import datetime, timezone

            shadow_dir = Path("data/shadow_logs")
            shadow_dir.mkdir(parents=True, exist_ok=True)
            log_path = shadow_dir / "challenger_shadow.jsonl"

            record = {
                "ts": datetime.now(timezone.utc).isoformat(),
                "match_id": match_id,
                "champion_raw": round(champion_raw, 4),
                "challenger_raw": round(challenger_raw, 4),
                "champion_conf": round(champion_conf, 4),
                "challenger_conf": round(challenger_conf, 4),
                "line_val": line_val,
                "is_over": is_over,
                "champion_only_mode": CHAMPION_ONLY_MODE,
            }
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
        except Exception as e:
            print(f"[SHADOW LOG] Warning: could not write shadow log: {e}")
