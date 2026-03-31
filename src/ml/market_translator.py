"""
src/ml/market_translator.py
============================
Projeção da Distribuição Latente Y → 9 Mercados Derivados.

Recebe o vetor λ = [λ_h1H, λ_a1H, λ_h2H, λ_a2H] do JointCornersModel
e retorna a distribuição de probabilidade para cada família de mercado.

Princípio central:
    TODOS os 9 mercados derivam da MESMA distribuição latente.
    Nunca são calculados de fontes independentes.

Coerência obrigatória:
    home_ft = home_1H + home_2H
    away_ft = away_1H + away_2H
    total_ft = home_ft + away_ft
    total_1H = home_1H + away_1H
    total_2H = home_2H + away_2H

Proibições:
    - NÃO usar 2T = FT - 1T como derivação preditiva
    - NÃO tratar Monte Carlo como automaticamente calibrado
    - Simulações devem ser comparadas com histórico para validar calibração

Referências:
    Karlis & Ntzoufras (2003) - Bivariate Poisson Models
    Gneiting & Raftery (2007) - "Strictly Proper Scoring Rules" — JASA
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy.stats import poisson, nbinom


# ---------------------------------------------------------------------------
# Famílias de Mercado
# ---------------------------------------------------------------------------

MARKET_FAMILIES = [
    "ft_total",   # total FT (soma dos 4 componentes)
    "ht_total",   # total 1H
    "ht2_total",  # total 2H
    "ft_home",    # home FT
    "ft_away",    # away FT
    "ht_home",    # home 1H
    "ht_away",    # away 1H
    "ht2_home",   # home 2H
    "ht2_away",   # away 2H
]


@dataclass
class MarketDistribution:
    """
    Distribuição de probabilidade para um mercado específico.

    Campos:
        family: Nome da família de mercado.
        line: Linha Over/Under (ex: 9.5).
        prob_over: P(X > line) — probabilidade bruta.
        prob_under: P(X <= line) — probabilidade bruta.
        expected: Valor esperado E[X].
        std: Desvio padrão do mercado.
        samples: Amostras Monte Carlo (para pós-calibração).
    """
    family: str
    line: float
    prob_over: float
    prob_under: float
    expected: float
    std: float
    samples: Optional[np.ndarray] = field(default=None, repr=False)

    @property
    def fair_odd_over(self) -> float:
        """Odd justa para Over — apenas informativa, nunca como critério de ranking."""
        return round(1.0 / self.prob_over, 3) if self.prob_over > 0 else 999.0

    @property
    def fair_odd_under(self) -> float:
        """Odd justa para Under — apenas informativa."""
        return round(1.0 / self.prob_under, 3) if self.prob_under > 0 else 999.0


# ---------------------------------------------------------------------------
# Translator
# ---------------------------------------------------------------------------

class MarketTranslator:
    """
    Projeta o vetor latente λ em 9 distribuições de mercado.

    Método:
        Monte Carlo sobre distribuições marginais (Poisson independente
        para cada componente, com λ3 de covariância quando disponível).

    Nota sobre calibração:
        As probabilidades retornadas são BRUTAS (não calibradas).
        A calibração é responsabilidade do PerMarketCalibrator.
        Monte Carlo com N > 50.000 converge bem, mas não garante calibração
        sem comparação com dados históricos.

    Args:
        n_simulations: Número de amostras MC por inferência.
        random_seed: Seed para reprodutibilidade (por padrão None = estocastico).
    """

    def __init__(self, n_simulations: int = 50_000, random_seed: Optional[int] = None) -> None:
        self.n_simulations = n_simulations
        self.random_seed = random_seed
        self._rng = np.random.default_rng(random_seed)

    def translate(
        self,
        lambda_dict: Dict[str, float],
        lines: Optional[Dict[str, List[float]]] = None,
        lambda3_1h: float = 0.0,
        lambda3_2h: float = 0.0,
    ) -> Dict[str, List[MarketDistribution]]:
        """
        Traduz o vetor λ em distribuições para cada família de mercado.

        Args:
            lambda_dict: Output do JointCornersModel.predict_lambda().
                         Deve conter: {home_1H, away_1H, home_2H, away_2H, ...}
            lines: Linhas Over/Under por família.
                   Default: linhas típicas por família.
            lambda3_1h: Covariância Bivariate Poisson — período 1H.
            lambda3_2h: Covariância Bivariate Poisson — período 2H.

        Returns:
            Dict[family_name → List[MarketDistribution]] (uma por linha).
        """
        if lines is None:
            lines = self._default_lines()

        # 1. Simular os 4 componentes latentes
        samples = self._simulate_joint(
            lambda_dict=lambda_dict,
            lambda3_1h=lambda3_1h,
            lambda3_2h=lambda3_2h,
        )

        # 2. Derivar séries por família
        family_samples = self._derive_family_samples(samples)

        # 3. Converter em MarketDistribution por linha
        result: Dict[str, List[MarketDistribution]] = {}
        for family in MARKET_FAMILIES:
            s = family_samples[family]
            family_lines = lines.get(family, self._default_lines()[family])
            result[family] = []
            for line in family_lines:
                prob_over = float(np.mean(s > line))
                prob_under = float(np.mean(s <= line))
                result[family].append(
                    MarketDistribution(
                        family=family,
                        line=line,
                        prob_over=max(1e-6, min(1.0 - 1e-6, prob_over)),
                        prob_under=max(1e-6, min(1.0 - 1e-6, prob_under)),
                        expected=float(np.mean(s)),
                        std=float(np.std(s)),
                        samples=s,
                    )
                )

        return result

    # ------------------------------------------------------------------
    # Simulação Monte Carlo
    # ------------------------------------------------------------------

    def _simulate_joint(
        self,
        lambda_dict: Dict[str, float],
        lambda3_1h: float,
        lambda3_2h: float,
    ) -> Dict[str, np.ndarray]:
        """
        Simula N amostras dos 4 componentes latentes com estrutura Bivariate Poisson.

        Bivariate Poisson (Karlis & Ntzoufras 2003):
            X = X' + Z,  Y = Y' + Z
            X' ~ Poisson(λ_h - λ3), Y' ~ Poisson(λ_a - λ3), Z ~ Poisson(λ3)

        Garante λ3 ≤ min(λ_h, λ_a) - ε para λ' ≥ 0.
        """
        N = self.n_simulations
        rng = self._rng

        def biv_poisson_pair(lh: float, la: float, l3: float) -> Tuple[np.ndarray, np.ndarray]:
            safe_l3 = min(l3, lh - 0.01, la - 0.01)
            safe_l3 = max(0.0, safe_l3)
            x = rng.poisson(lh - safe_l3, N)
            y = rng.poisson(la - safe_l3, N)
            z = rng.poisson(safe_l3, N)
            return x + z, y + z

        l_h1H = max(0.05, lambda_dict.get("home_1H", 0.05))
        l_a1H = max(0.05, lambda_dict.get("away_1H", 0.05))
        l_h2H = max(0.05, lambda_dict.get("home_2H", 0.05))
        l_a2H = max(0.05, lambda_dict.get("away_2H", 0.05))

        h1H, a1H = biv_poisson_pair(l_h1H, l_a1H, lambda3_1h)
        h2H, a2H = biv_poisson_pair(l_h2H, l_a2H, lambda3_2h)

        return {
            "home_1H": h1H,
            "away_1H": a1H,
            "home_2H": h2H,
            "away_2H": a2H,
        }

    def _derive_family_samples(
        self, samples: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """
        Deriva os 9 mercados das 4 séries simuladas.
        Coerência garantida: todos derivam da mesma simulação.
        """
        h1H = samples["home_1H"]
        a1H = samples["away_1H"]
        h2H = samples["home_2H"]
        a2H = samples["away_2H"]

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

    # ------------------------------------------------------------------
    # Linhas padrão por família
    # ------------------------------------------------------------------

    @staticmethod
    def _default_lines() -> Dict[str, List[float]]:
        """Linhas Over/Under típicas para cada família de mercado."""
        return {
            "ft_total":  [7.5, 8.5, 9.5, 10.5, 11.5, 12.5],
            "ht_total":  [3.5, 4.5, 5.5],
            "ht2_total": [3.5, 4.5, 5.5],
            "ft_home":   [3.5, 4.5, 5.5, 6.5],
            "ft_away":   [3.5, 4.5, 5.5, 6.5],
            "ht_home":   [1.5, 2.5, 3.5],
            "ht_away":   [1.5, 2.5, 3.5],
            "ht2_home":  [1.5, 2.5, 3.5],
            "ht2_away":  [1.5, 2.5, 3.5],
        }

    # ------------------------------------------------------------------
    # Fast analytical P(over) — para OOF calibration (sem MC)
    # ------------------------------------------------------------------

    @staticmethod
    def compute_prob_over_analytical(
        lambdas_df: pd.DataFrame,
        lines: Optional[Dict[str, float]] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Computa P(X > line) analiticamente via CDF Poisson para cada família.

        Válido quando lambda3=0 (Poisson independente), que é o default.
        Resultado: dict {family: array de probabilidades shape (n_samples,)}.

        Args:
            lambdas_df: DataFrame com colunas do predict_lambda_batch().
            lines: {family: line} — uma linha por família.
        """
        if lines is None:
            lines = {
                "ft_total": 9.5, "ht_total": 4.5, "ht2_total": 4.5,
                "ft_home": 4.5, "ft_away": 4.5,
                "ht_home": 2.5, "ht_away": 2.5,
                "ht2_home": 2.5, "ht2_away": 2.5,
            }

        # Mapping: family → lambda column in the batch dataframe
        family_to_lambda = {
            "ft_total": "ft_total",
            "ht_total": "total_1H",
            "ht2_total": "total_2H",
            "ft_home": "home_ft",
            "ft_away": "away_ft",
            "ht_home": "home_1H",
            "ht_away": "away_1H",
            "ht2_home": "home_2H",
            "ht2_away": "away_2H",
        }

        result = {}
        for family, line in lines.items():
            col = family_to_lambda.get(family)
            if col is None or col not in lambdas_df.columns:
                continue
            lam_arr = lambdas_df[col].values
            # P(X > line) = 1 - P(X <= floor(line)) = poisson.sf(floor(line), lambda)
            result[family] = poisson.sf(int(np.floor(line)), lam_arr).clip(1e-6, 1 - 1e-6)

        return result

    # ------------------------------------------------------------------
    # Comparação com odd do usuário (opcional)
    # ------------------------------------------------------------------

    @staticmethod
    def compare_with_user_odd(
        prob_calibrated: float, user_odd: float
    ) -> Dict[str, float]:
        """
        Compara probabilidade calibrada com odd fornecida pelo usuário.

        NOTA: Este método só deve ser chamado quando o usuário informa
        uma odd manualmente. Sem odd real, não exibir EV nem edge.

        Args:
            prob_calibrated: Probabilidade pós-calibração do modelo.
            user_odd: Odd decimal informada pelo usuário.

        Returns:
            Dict com fair_odd, implied_prob, edge_vs_market.
        """
        if prob_calibrated <= 0:
            return {}
        fair_odd = 1.0 / prob_calibrated
        implied_prob = 1.0 / user_odd if user_odd > 1.0 else 1.0
        edge = prob_calibrated - implied_prob
        return {
            "fair_odd": round(fair_odd, 3),
            "implied_prob_market": round(implied_prob, 4),
            "edge_vs_market": round(edge, 4),
            "is_value": edge > 0,
        }
