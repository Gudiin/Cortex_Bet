"""
Módulo de Integração: Odds Reais + Modelo de ML

Este módulo integra o scraper de odds da Superbet com o modelo de ML,
permitindo a validação do ROI com dados reais de apostas.
"""

import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OddsIntegrator:
    """Integra odds reais com previsões do modelo de ML."""

    def __init__(self, scraper=None):
        self.scraper = scraper
        self.odds_cache = {}

    async def fetch_odds_for_matches(self, match_urls: List[str]) -> pd.DataFrame:
        """Busca odds para uma lista de partidas."""
        if not self.scraper:
            from src.scrapers.superbet_odds_scraper import SuperbetOddsScraper

            self.scraper = SuperbetOddsScraper(headless=True)

        try:
            df_odds = await self.scraper.extract_odds_batch(match_urls)
            logger.info("✅ %s odds obtidas com sucesso.", len(df_odds))
            return df_odds
        except Exception as exc:
            logger.error("❌ Erro ao buscar odds: %s", exc)
            return pd.DataFrame()

    def validate_odds(self, df_odds: pd.DataFrame) -> pd.DataFrame:
        """Valida e limpa dados de odds."""
        df_clean = df_odds.copy()
        df_clean = df_clean[
            (df_clean["over_odds"] > 1.0)
            & (df_clean["under_odds"] > 1.0)
            & (df_clean["line"].notna())
        ]
        df_clean = df_clean[
            (df_clean["over_odds"] >= 1.01)
            & (df_clean["over_odds"] <= 50.0)
            & (df_clean["under_odds"] >= 1.01)
            & (df_clean["under_odds"] <= 50.0)
        ]
        df_clean = df_clean[(df_clean["line"] >= 0) & (df_clean["line"] <= 20)]

        logger.info(
            "✅ %s odds validadas (removidas %s inválidas).",
            len(df_clean),
            len(df_odds) - len(df_clean),
        )
        return df_clean

    def merge_predictions_with_odds(
        self,
        predictions: np.ndarray,
        df_odds: pd.DataFrame,
        y_true: Optional[np.ndarray] = None,
    ) -> pd.DataFrame:
        """Mescla previsões do modelo com dados de odds."""
        df_merged = df_odds.copy()
        df_merged["prediction"] = predictions[: len(df_merged)]
        if y_true is not None:
            df_merged["actual"] = y_true[: len(df_merged)]
        return df_merged

    def calculate_ev_bets(self, df_merged: pd.DataFrame, margin: float = 0.05) -> pd.DataFrame:
        """Identifica apostas com Valor Esperado Positivo (+EV)."""
        from scipy.stats import poisson

        df_ev = df_merged.copy()
        df_ev["model_prob_over"] = df_ev.apply(
            lambda row: 1 - poisson.cdf(k=row["line"], mu=row["prediction"]), axis=1
        )
        df_ev["model_prob_under"] = df_ev.apply(
            lambda row: poisson.cdf(k=row["line"], mu=row["prediction"]), axis=1
        )
        df_ev["bookie_prob_over"] = 1 / df_ev["over_odds"]
        df_ev["bookie_prob_under"] = 1 / df_ev["under_odds"]
        df_ev["is_over_ev"] = df_ev["model_prob_over"] > df_ev["bookie_prob_over"] * (1 + margin)
        df_ev["is_under_ev"] = df_ev["model_prob_under"] > df_ev["bookie_prob_under"] * (1 + margin)
        df_ev_only = df_ev[df_ev["is_over_ev"] | df_ev["is_under_ev"]].copy()

        logger.info("✅ %s apostas +EV identificadas de %s total.", len(df_ev_only), len(df_ev))
        return df_ev_only

    def simulate_betting(self, df_ev: pd.DataFrame, stake: float = 1.0) -> Dict:
        """Simula apostas baseadas nas oportunidades +EV."""
        results = {
            "total_bets": 0,
            "total_wins": 0,
            "total_losses": 0,
            "total_profit": 0.0,
            "win_rate": 0.0,
            "roi": 0.0,
            "roi_percent": 0.0,
            "avg_odds": 0.0,
            "bets": [],
        }
        if df_ev.empty:
            logger.warning("⚠️ Nenhuma aposta +EV para simular.")
            return results

        total_odds = 0.0
        for _, row in df_ev.iterrows():
            bet_type = "Over" if row["is_over_ev"] else "Under"
            odds = row["over_odds"] if row["is_over_ev"] else row["under_odds"]
            bet = {
                "match": row.get("match_name", "Unknown"),
                "type": bet_type,
                "line": row["line"],
                "odds": odds,
                "stake": stake,
            }
            if "actual" in row and pd.notna(row["actual"]):
                actual = row["actual"]
                line = row["line"]
                is_win = (actual > line) if bet_type == "Over" else (actual <= line)
                bet["result"] = "WIN" if is_win else "LOSS"
                bet_profit = (odds - 1) * stake if is_win else -stake
                bet["profit"] = bet_profit

                results["total_bets"] += 1
                total_odds += odds
                if is_win:
                    results["total_wins"] += 1
                else:
                    results["total_losses"] += 1
                results["total_profit"] += bet_profit
            results["bets"].append(bet)

        if results["total_bets"] > 0:
            results["win_rate"] = results["total_wins"] / results["total_bets"]
            total_staked = results["total_bets"] * stake
            results["roi"] = results["total_profit"] / total_staked if total_staked else 0.0
            results["roi_percent"] = results["roi"] * 100
            results["avg_odds"] = total_odds / results["total_bets"]

        return results

    def generate_report(
        self,
        df_merged: pd.DataFrame,
        betting_results: Dict,
        output_path: Optional[str] = None,
    ) -> str:
        """Gera um relatório detalhado de performance."""
        coverage = (betting_results["total_bets"] / len(df_merged) * 100) if len(df_merged) else 0.0
        report = f"""
╔════════════════════════════════════════════════════════════════╗
║           RELATÓRIO DE PERFORMANCE - ODDS REAIS               ║
╚════════════════════════════════════════════════════════════════╝

📊 ESTATÍSTICAS GERAIS
─────────────────────────────────────────────────────────────────
Total de Partidas Analisadas:  {len(df_merged)}
Apostas +EV Identificadas:     {betting_results['total_bets']}
Taxa de Cobertura:             {coverage:.1f}%

💰 RESULTADOS DE APOSTAS
─────────────────────────────────────────────────────────────────
Total de Apostas:              {betting_results['total_bets']}
Vitórias:                      {betting_results['total_wins']}
Derrotas:                      {betting_results['total_losses']}
Win Rate:                      {betting_results['win_rate']:.2%}

📈 RETORNO FINANCEIRO
─────────────────────────────────────────────────────────────────
Lucro Total:                   {betting_results['total_profit']:+.2f} unidades
ROI (Retorno):                 {betting_results['roi']:+.2f} unidades
ROI (%):                       {betting_results['roi_percent']:+.1f}%
Odd Média:                     {betting_results['avg_odds']:.2f}
"""
        if betting_results["roi_percent"] > 15:
            report += "\n✅ EXCELENTE! ROI acima de 15%. Modelo tem potencial lucrativo.\n"
        elif betting_results["roi_percent"] > 5:
            report += "\n🟡 BOM! ROI positivo. Modelo é viável com gestão de banca.\n"
        elif betting_results["roi_percent"] > 0:
            report += "\n🟠 MARGINAL. ROI positivo mas baixo. Requer otimização.\n"
        else:
            report += "\n🔴 NEGATIVO. Modelo não é lucrativo. Revisar features/parâmetros.\n"

        report += f"""

═════════════════════════════════════════════════════════════════
Relatório gerado em: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
═════════════════════════════════════════════════════════════════
"""
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as report_file:
                report_file.write(report)
            logger.info("✅ Relatório salvo em: %s", output_path)

        return report


async def main():
    from src.scrapers.superbet_odds_scraper import SuperbetOddsScraper

    scraper = SuperbetOddsScraper(headless=True)
    integrator = OddsIntegrator(scraper=scraper)
    try:
        urls = [
            "https://superbet.bet.br/odds/futebol/arsenal-x-wolverhampton-8627842/?t=offer-prematch-106&mdt=o",
        ]
        df_odds = await integrator.fetch_odds_for_matches(urls)
        if not df_odds.empty:
            df_odds_clean = integrator.validate_odds(df_odds)
            predictions = np.random.uniform(8, 12, len(df_odds_clean))
            df_merged = integrator.merge_predictions_with_odds(predictions, df_odds_clean)
            df_ev = integrator.calculate_ev_bets(df_merged)
            results = integrator.simulate_betting(df_ev)
            print(integrator.generate_report(df_merged, results))
    finally:
        await scraper.close()


if __name__ == "__main__":
    asyncio.run(main())
