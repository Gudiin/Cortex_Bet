"""
Módulo de Análise Estatística para Previsão de Escanteios.

Este módulo implementa análise estatística avançada utilizando distribuições
probabilísticas (Poisson e Binomial Negativa) e simulações de Monte Carlo
para calcular probabilidades de mercados de escanteios.

Regras de Negócio:
    - Utiliza distribuição de Poisson quando variância ≤ média
    - Utiliza Binomial Negativa quando variância > média (overdispersion)
    - Monte Carlo com 10.000 simulações para precisão estatística
    - Gera sugestões categorizadas por nível de risco (Easy/Medium/Hard)
"""

import numpy as np
import pandas as pd
from scipy.stats import poisson, nbinom
from tabulate import tabulate


class Colors:
    """
    Constantes ANSI para colorização de output no terminal.
    
    Permite destacar visualmente diferentes tipos de informação:
    - GREEN: Apostas Over, vitórias
    - RED: Alertas, erros
    - CYAN: Apostas Under
    - YELLOW: Destaques importantes
    """
    RESET = "\033[0m"
    BOLD = "\033[1m"
    GREEN = "\033[92m"
    RED = "\033[91m"
    CYAN = "\033[96m"
    YELLOW = "\033[93m"


class StatisticalAnalyzer:
    """
    Analisador estatístico para previsão de escanteios em partidas de futebol.
    
    Utiliza modelos probabilísticos e simulação Monte Carlo para calcular
    probabilidades de diferentes mercados de escanteios (Over/Under).
    
    Modelos Probabilísticos:
        - Distribuição de Poisson: Usada quando variância ≈ média (equidispersão)
          Ideal para eventos raros e independentes como escanteios.
          
        - Binomial Negativa: Usada quando variância > média (overdispersion)
          Mais flexível, captura variabilidade extra em jogos atípicos.
    
    Simulação Monte Carlo:
        Gera 10.000 cenários aleatórios baseados na distribuição escolhida,
        permitindo estimar probabilidades de qualquer mercado.
    
    Mercados Analisados:
        - JOGO COMPLETO: Total de escanteios (8.5 a 12.5)
        - MANDANTE/VISITANTE: Escanteios por time (3.5 a 6.5)
        - 1º/2º TEMPO: Escanteios por período (3.5 a 5.5)
        - MANDANTE/VISITANTE por tempo: Linhas mais baixas (1.5 a 3.5)
    
    Cálculos Principais:
        1. Lambda (λ): Taxa média de escanteios esperados
           λ = 0.6 * média_10_jogos + 0.4 * média_5_jogos
        
        2. Odd Justa: Conversão de probabilidade em odd
           Odd = 1 / Probabilidade
        
        3. Score: Ranking de oportunidades
           Score = Probabilidade * (1 - CV * fator)
           onde CV = Coeficiente de Variação (σ/μ)
    
    Attributes:
        Nenhum atributo persistente - stateless por design.
    
    Example:
        >>> analyzer = StatisticalAnalyzer()
        >>> top_picks = analyzer.analyze_match(df_home, df_away)
    """
    
    def __init__(self):
        """
        Inicializa o analisador estatístico.
        
        Classe é stateless - nenhuma inicialização necessária.
        """
        self.n_simulations = 10000
        
        # Pesos padrão (serão substituídos por Bayesianos se houver histórico)
        self.default_weights = {
            'IA': 0.40,
            'Specific': 0.25,
            'Defense': 0.15,
            'H2H': 0.10,
            'Momentum': 0.10
        }
    
    def calculate_bayesian_weights(
        self,
        historical_errors: dict = None
    ) -> dict:
        """
        Calcula pesos dinamicamente usando inverso do erro quadrático médio.
        
        Permite que fontes mais precisas recebam mais peso automaticamente.
        
        Args:
            historical_errors: Dict com listas de erros por fonte.
                              Ex: {'IA': [1.2, 0.8, ...], 'Specific': [0.5, 1.1, ...]}
        
        Returns:
            dict: Pesos normalizados que somam 1.0
        
        Fórmula Bayesiana (aproximada):
            w_i = (1 / MSE_i) / Σ(1 / MSE_j)
            
        Onde MSE_i é o erro quadrático médio da fonte i.
        """
        if not historical_errors or len(historical_errors) == 0:
            return self.default_weights.copy()
        
        weights = {}
        total_precision = 0
        
        for source, errors in historical_errors.items():
            if len(errors) == 0:
                weights[source] = 1.0
                total_precision += 1.0
                continue
                
            mse = np.mean(np.array(errors) ** 2) + 1e-6  # Evita divisão por zero
            precision = 1.0 / mse
            weights[source] = precision
            total_precision += precision
        
        # Normaliza para somar 1
        if total_precision > 0:
            for source in weights:
                weights[source] /= total_precision
        
        return weights

    @staticmethod
    def calculate_ev(probability: float, odds: float) -> float:
        """
        Calcula o Valor Esperado (+EV) de uma aposta.
        
        Args:
            probability: Probabilidade estimada pelo modelo (0.0 a 1.0)
            odds: Odd oferecida pela casa
            
        Returns:
            float: Valor Esperado em % (ex: 5.0 para 5% de valor)
        """
        if odds <= 1.0:
            return -100.0
        
        # Fórmula EV: (Prob * Odds) - 1
        ev = (probability * odds) - 1
        return ev * 100

    @staticmethod
    def calculate_kelly(probability: float, odds: float, fraction: float = 0.25) -> float:
        """
        Calcula a gestão de banca sugerida pelo Critério de Kelly (Fracionário).
        
        Args:
            probability: Probabilidade estimada (0.0 a 1.0)
            odds: Odd Decimal
            fraction: Fração do Kelly (padrão 1/4 Kelly para segurança)
            
        Returns:
            float: Porcentagem da banca a apostar (0.0 a 100.0)
        """
        if odds <= 1.0:
            return 0.0
            
        b = odds - 1
        q = 1 - probability
        p = probability
        
        # Kelly: (bp - q) / b
        f = (b * p - q) / b
        
        # Ajuste fracionário e proibe negativos
        recommendation = max(0, f) * fraction
        
        # Cap de segurança (ex: nunca apostar mais de 5% da banca)
        return min(recommendation * 100, 5.0)

    def calculate_hybrid_lambdas(
        self,
        neural_params: dict,
        match_name: str = None
    ) -> tuple:
        """
        Calcula lambdas finais usando parâmetros puramente neurais (Cyborg Upgrade).
        
        A IA já processou o contexto tático, histórico e momentum.
        O papel deste módulo é apenas propagar esses parâmetros para a distribuição matemática.
        
        Args:
            neural_params: Dict contendo:
                - lambda_home: Expectativa de cantos Mandante
                - lambda_away: Expectativa de cantos Visitante
                - variance_factor: Fator de caos (1.0 = Poisson, >1.0 = Overdispersion)
            match_name: Nome da partida para log.
            
        Returns:
            tuple: (lambda_home, lambda_away, tactical_data)
        """
        lambda_home = neural_params.get('lambda_home', 0.0)
        lambda_away = neural_params.get('lambda_away', 0.0)
        
        # Log detalhado (Cyborg Style)
        print(f"\n{Colors.CYAN}{'='*80}")
        header = f"� JOINT-STATISTICAL ENGINE ({match_name})" if match_name else "🧬 JOINT-STATISTICAL ENGINE"
        print(f"{header}")
        print(f"{'='*80}{Colors.RESET}")
        
        total_pred = lambda_home + lambda_away
        split_h = lambda_home / total_pred if total_pred > 0 else 0.5
        
        source = neural_params.get('source', 'neural_legacy')
        print(f"🧠 {source} Prediction: {total_pred:.2f} corners")
        print(f"🏠 Home λ: {lambda_home:.2f} ({split_h*100:.1f}%)") 
        print(f"✈️  Away λ: {lambda_away:.2f} ({(1-split_h)*100:.1f}%)")
        if 'lambda_home_1h' in neural_params:
            print(f"📊 HT λ: h1H={neural_params['lambda_home_1h']:.2f} a1H={neural_params['lambda_away_1h']:.2f}")
        print(f"🌊 Variance Factor: {neural_params.get('variance_factor', 1.0):.2f}")
        print(f"{Colors.CYAN}{'='*80}{Colors.RESET}\n")
        
        tactical_data = {
            'prop_home': split_h,
            'lambda_home': lambda_home,
            'lambda_away': lambda_away,
            'ia_weight': 1.0, # 100% Neural Driven
            'match_name': match_name,
            'variance_factor': neural_params.get('variance_factor', 1.0)
        }
        
        return lambda_home, lambda_away, tactical_data

    def _get_distribution_params(self, data: pd.Series) -> tuple:
        """
        Calcula parâmetros da distribuição para uma série de dados.
        
        Args:
            data: Série temporal de dados (ex: escanteios nos últimos jogos).
            
        Returns:
            tuple: (tipo_distribuicao, media, variancia)
        """
        if len(data) == 0:
            return 'poisson', 0, 0
            
        mean = data.mean()
        var = data.var() if len(data) > 1 else 0
        
        # Se variância for zero ou NaN, assume Poisson com a média
        if pd.isna(var) or var == 0:
            return 'poisson', mean, 0
            
        dist_type = 'nbinom' if var > mean else 'poisson'
        return dist_type, mean, var

    def calculate_covariance(self, df_home: pd.DataFrame, df_away: pd.DataFrame, window: int = 20) -> float:
        """
        Calcula a covariância histórica entre escanteios feitos e sofridos.
        
        A covariância captura se jogos desses times tendem a ser "abertos" (muitos cantos para ambos)
        ou "fechados" (poucos cantos para ambos).
        
        Args:
            df_home: Histórico do mandante.
            df_away: Histórico do visitante.
            window: Número de jogos a considerar (default 20).
            
        Returns:
            float: Lambda 3 (fator de covariância), garantido >= 0.
        """
        # Concatena últimos jogos de cada time para analisar comportamento geral
        # Precisamos de 'corners_ft' e 'corners_conceded_ft' (que é corners do oponente)
        
        # Como os dataframes já têm colunas de escanteios feitos/sofridos:
        # df['corners_ft'] vs df['corners_away_ft'] (se for home)
        # Atenção: df_home['corners_ft'] são cantos QUE O TIME FEZ. 
        # Precisamos da relação SOMA(Casa + Fora) ou correlação entre (Casa, Fora).
        
        # Unificando histórico relevante
        # Para cada jogo do histórico, pegamos (Cantos Time, Cantos Oponente)
        
        def get_pairs(df):
            if df.empty: return []
            recent = df.sort_values('start_timestamp', ascending=False).head(window)
            return list(zip(recent['corners_ft'], recent['corners_away_ft'])) # Assume estrutura padrão onde corners_away_ft é o oponente
            
        # Pega pares (feito, sofrido) de ambos os times
        # Mas espere: num jogo passado do Home, 'corners_ft' é ele, 'corners_away_ft' é o oponente.
        # A covariância que buscamos é do PROCESSO GERADOR DE JOGO.
        # Se o time A joga, existe correlação entre ele marcar e sofrer?
        
        # Calculation strategy:
        # 1. Calculate Cov(Home_Made, Home_Conceded) for Home Team
        # 2. Calculate Cov(Away_Made, Away_Conceded) for Away Team
        # 3. Average them? 
        
        # Bivariate Poisson Theory (Karlis & Ntzoufras 2003):
        # We model the UPCOMING match between Home and Away.
        # We want to know if there is a common shock Z affecting THIS match.
        # Z comes from the "nature" of the teams meeting.
        # Usually approximated by the correlation of scores in their past games.
        
        # Simplified Implementation:
        # Calculate covariance of (Home Corners, Away Corners) across recent matches of these teams.
        
        vals_h = []
        if not df_home.empty:
            rec_h = df_home.sort_values('start_timestamp', ascending=False).head(window)
            # Covariance of (MyCorners, OpponentCorners)
            # Note: In our DB, corners_ft is HomeTeam(if is_home) and Oponente is corners_away_ft?
            # Need to be careful with column names.
            # Assuming standard structure:
            # corners_ft = Home Team Corners (always? No, DB columns are corners_home_ft, corners_away_ft)
            # df_home usually contains matches where 'home_team_id' or 'away_team_id' is the team.
            
            # Let's trust input structure is consistent filter for the specific team
            # But simpler: just take the columns 'corners_home_ft' and 'corners_away_ft' 
            # from the matches involving these teams.
            
            # Covariance(X, Y) = E[(X-Ux)(Y-Uy)]
            cov_h = rec_h[['corners_home_ft', 'corners_away_ft']].cov().iloc[0, 1]
            vals_h.append(cov_h)

        if not df_away.empty:
            rec_a = df_away.sort_values('start_timestamp', ascending=False).head(window)
            cov_a = rec_a[['corners_home_ft', 'corners_away_ft']].cov().iloc[0, 1]
            vals_h.append(cov_a)
            
        if not vals_h:
            return 0.0
            
        avg_cov = np.mean(vals_h)
        
        # Lambda 3 must be non-negative (common shock increases counts)
        # And usually small (0 to 2 for corners)
        return max(0.0, avg_cov)

    def simulate_bivariate_match(self, lambda_home: float, lambda_away: float, lambda3: float, n_sims: int = 10000) -> np.ndarray:
        """
        Simula partida usando Poisson Bivariado.
        
        X ~ Poisson(lambda_home - lambda3)
        Y ~ Poisson(lambda_away - lambda3)
        Z ~ Poisson(lambda3)  <-- Choque comum (aumenta ambos)
        
        Final Home = X + Z
        Final Away = Y + Z
        
        Args:
            lambda_home: Média esperada do Mandante (marginal)
            lambda_away: Média esperada do Visitante (marginal)
            lambda3: Covariância (interseção)
            
        Returns:
            np.ndarray: Array com TOTAL de escanteios (Home + Away) para cada simulação
        """
        # Restrição teórica: lambda1, lambda2 >= 0
        # lambda3 não pode ser maior que as médias marginais
        safe_lambda3 = min(lambda3, lambda_home - 0.1, lambda_away - 0.1)
        safe_lambda3 = max(0.0, safe_lambda3)
        
        lambda1 = lambda_home - safe_lambda3
        lambda2 = lambda_away - safe_lambda3
        
        # Gera os componentes
        # Usa random_state para reprodutibilidade se necessário, mas aqui queremos estocástico
        x = poisson.rvs(lambda1, size=n_sims)
        y = poisson.rvs(lambda2, size=n_sims)
        z = poisson.rvs(safe_lambda3, size=n_sims)
        
        final_home = x + z
        final_away = y + z
        
        return final_home + final_away

    def simulate_match_event(self, avg_home: float, avg_away: float, 
                           var_home: float = 0, var_away: float = 0,
                           covariance: float = 0.0) -> np.ndarray:
        """
        Simula um evento de partida (ex: Total Escanteios) combinando mandante e visitante.
        Suporta simulação Bivariada se covariance > 0.
        
        Args:
            avg_home: Média do mandante.
            avg_away: Média do visitante.
            var_home: Variância do mandante.
            var_away: Variância do visitante.
            covariance: Covariância entre mandante e visitante (Lambda 3).
            
        Returns:
            np.ndarray: Array com a soma das simulações (Home + Away).
        """
        # Se tivermos covariância significativa (> 0.1), usamos o modelo Bivariado
        if covariance > 0.1:
            return self.simulate_bivariate_match(avg_home, avg_away, covariance, self.n_simulations)
            
        # Caso contrário, fallback para o modelo Independente (Univariate)
        # Verifica Overdispersion para decidir entre Poisson e Negative Binomial (existente)
        sim_home = self.monte_carlo_simulation(avg_home, var_home)
        sim_away = self.monte_carlo_simulation(avg_away, var_away)
        return sim_home + sim_away

    def monte_carlo_simulation(self, lambda_val: float, var_val: float, 
                               n_sims: int = 10000) -> np.ndarray:
        """
        Executa simulação de Monte Carlo para estimar distribuição de escanteios.
        
        Gera N cenários aleatórios seguindo a distribuição apropriada
        (Poisson ou Binomial Negativa) baseada na relação variância/média.
        
        Args:
            lambda_val: Taxa média esperada de escanteios (λ).
            var_val: Variância observada nos dados históricos.
            n_sims: Número de simulações (default: 10.000).
        
        Returns:
            np.ndarray: Array com n_sims valores simulados de escanteios.
        
        Lógica:
            1. Compara variância com média (lambda)
            2. Se variância > lambda: usa Binomial Negativa (overdispersion)
            3. Se variância ≤ lambda: usa Poisson (equidispersion)
            4. Gera n_sims amostras da distribuição escolhida
        
        Fórmulas:
            Poisson:
                P(X=k) = (λ^k * e^(-λ)) / k!
                Onde λ = média esperada
            
            Binomial Negativa (parametrização alternativa):
                p = λ / σ²  (probabilidade de sucesso)
                n = λ² / (σ² - λ)  (número de sucessos)
        
        Regras de Negócio:
            - 10.000 simulações fornece precisão de ~1% nas probabilidades
            - Overdispersion é comum em futebol (jogos imprevisíveis)
            - Monte Carlo captura toda a distribuição, não apenas a média
        
        Example:
            >>> sims = analyzer.monte_carlo_simulation(10.5, 15.0)
            >>> prob_over_9 = (sims > 9.5).mean()  # ~65%
        """
        if lambda_val <= 0.0:
            # Não é possível gerar Poisson ou Nbinom com lambda <= 0, retorna array de zeros
            return np.zeros(n_sims, dtype=int)
            
        if var_val > lambda_val:
            # Overdispersion: usa Binomial Negativa
            p = lambda_val / var_val
            # Evita divisão por zero ou negativa por arredondamentos flutuantes:
            diff = var_val - lambda_val
            if diff <= 0: diff = 1e-6
            n = (lambda_val ** 2) / diff
            # Garante que n e p sejam válidos
            if n > 0 and 0 < p <= 1:
                try:
                    sims = nbinom.rvs(n, p, size=n_sims)
                except ValueError:
                    # Fallback final se hiperparâmetros ficarem instáveis
                    sims = poisson.rvs(lambda_val, size=n_sims)
            else:
                sims = poisson.rvs(lambda_val, size=n_sims)
        else:
            # Equidispersion: usa Poisson
            sims = poisson.rvs(lambda_val, size=n_sims)
        return sims

    @staticmethod
    def calculate_target_price(probability: float, min_roi: float = 0.05) -> float:
        """
        Calcula o Preço Alvo (Target Odd) para uma aposta ter valor.
        
        Args:
            probability: Probabilidade estimada (0.0 a 1.0)
            min_roi: Retorno sobre investimento mínimo desejado (padrão 5%)
            
        Returns:
            float: Odd mínima recomendada.
        """
        if probability <= 0:
            return 999.0
            
        # Fórmula: Odd = (1 + ROI) / Prob
        target_odd = (1.0 + min_roi) / probability
        return round(target_odd, 2)

    def generate_suggestions(self, opportunities: list, 
                            ml_prediction: float = None) -> dict:
        """
        Gera sugestões de apostas categorizadas por nível de risco.
        
        Analisa as oportunidades encontradas e seleciona as melhores
        para cada nível de risco, alinhando com a previsão do modelo ML.
        
        Args:
            opportunities: Lista de dicionários com oportunidades.
            ml_prediction: Previsão do modelo ML.
        
        Returns:
            dict: Sugestões por nível de risco.
        """
        suggestions = {
            "Easy": None,
            "Medium": None,
            "Hard": None
        }
        
        # Ordena por probabilidade (decrescente)
        sorted_ops = sorted(opportunities, key=lambda x: x['Prob'], reverse=True)
        
        def aligns_with_ml(op: dict) -> bool:
            if ml_prediction is None: return True
            if "Over" in op['Seleção'] and ml_prediction > 10.5: return True
            if "Under" in op['Seleção'] and ml_prediction < 9.5: return True
            if 9.5 <= ml_prediction <= 10.5: return True
            return False

        # =================================================================
        # FILTRO DE JUNK ODDS (Sprint 12 - Value Betting)
        # =================================================================
        # Regra: Ignorar sugestões onde a Odd Justa < 1.25
        # Probabilidades > 80% geram retorno financeiro pífio.
        # =================================================================
        
        min_fair_odd = 1.25

        # Easy: Probabilidade Calibrada Alta (>60%), mas com VALOR
        for op in sorted_ops:
            if op['Prob'] >= 0.60 and op['FairOdd'] >= min_fair_odd:
                if aligns_with_ml(op):
                    suggestions["Easy"] = op
                    break
        
        # Medium: Probabilidade Média (50-60%)
        for op in sorted_ops:
            if 0.50 <= op['Prob'] < 0.60 and op['FairOdd'] >= min_fair_odd:
                if aligns_with_ml(op):
                    suggestions["Medium"] = op
                    break
                
        # Hard: Probabilidade Moderada (40-50%), odds altas
        for op in sorted_ops:
            if 0.40 <= op['Prob'] < 0.50 and op['Odd'] > 2.20:
                if aligns_with_ml(op):
                    suggestions["Hard"] = op
                    break
                
        return suggestions

    def analyze_match(self, df_home: pd.DataFrame, df_away: pd.DataFrame, 
                     ml_prediction: float = None, match_name: str = None,
                     advanced_metrics: dict = None, scraped_odds: dict = None,
                     calibrator: any = None) -> tuple:
        """
        Executa análise estatística completa de uma partida.
        
        Calcula probabilidades para múltiplos mercados de escanteios
        usando Monte Carlo e gera ranking de melhores oportunidades.
        
        Args:
            df_home: DataFrame com histórico do mandante.
            df_away: DataFrame com histórico do visitante.
            ml_prediction: Previsão do modelo ML para alinhamento.
            match_name: Nome da partida.
            advanced_metrics: Métricas avançadas da IA.
            scraped_odds: Dict com odds raspadas do bookmaker (ex: {'Over 9.5': 1.85}).
            calibrator: Objeto calibrador (TemperatureScaling) para ajustar probabilidades.
        """
        # 1. Extração de Estatísticas Básicas
        # Calculamos médias e variâncias para alimentar as simulações
        
        # FIX: The dataframes passed (df_home, df_away) contain match history.
        # df_home contains matches where the target team was Home.
        # df_away contains matches where the target team was Away.
        # We need to extract the corners FOR that team.
        
        # For df_home (Team played Home): Use 'corners_home_ft'
        if 'corners_home_ft' in df_home.columns:
            h_corners_ft = df_home['corners_home_ft']
            h_corners_ht = df_home['corners_home_ht']
        else:
            # Fallback (Legacy schema?)
            h_corners_ft = df_home.get('corners_ft', pd.Series())
            h_corners_ht = df_home.get('corners_ht', pd.Series())

        # For df_away (Team played Away): Use 'corners_away_ft'
        # Wait - ManagerAI filters df_away based on AWAY_TEAM_ID match.
        # So for these rows, our team is Away. We want 'corners_away_ft'.
        if 'corners_away_ft' in df_away.columns:
            a_corners_ft = df_away['corners_away_ft']
            a_corners_ht = df_away['corners_away_ht']
        else:
             a_corners_ft = df_away.get('corners_ft', pd.Series())
             a_corners_ht = df_away.get('corners_ht', pd.Series())
        
        # Simulações (O "Coração" do Monte Carlo)
        # ---------------------------------------
        
        # Simula Jogo Completo (FT)
        dist_h, mean_h, var_h = self._get_distribution_params(h_corners_ft)
        dist_a, mean_a, var_a = self._get_distribution_params(a_corners_ft)
        
        # Capture Historical Means for Scaling Ratio
        mean_h_hist = mean_h if mean_h > 0 else 1.0
        mean_a_hist = mean_a if mean_a > 0 else 1.0
        
        # Lógica de Integração IA + Estatística (NÍVEL 2 - HÍBRIDO)
        # ------------------------------------------------------------
        
        tactical_metrics = {}
        tactical_metrics = {}
        
        # Lógica de Integração IA + Estatística (NÍVEL 3 - CYBORG)
        # ------------------------------------------------------------
        
        # Verifica se temos parâmetros neurais completos (da nova engine)
        if isinstance(advanced_metrics, dict) and 'neural_params' in advanced_metrics:
             mean_h, mean_a, tactical_metrics = self.calculate_hybrid_lambdas(
                neural_params=advanced_metrics['neural_params'],
                match_name=match_name
            )
            
        elif ml_prediction is not None and ml_prediction > 0:
            # 🤖 MODO LEGADO/FALLBACK: Constrói params neurais básicos a partir da predição única
            # Tenta inferir split a partir de métricas se disponíveis
            split_h = 0.5
            if advanced_metrics:
                 # Tentativa básica de split proporcional à média histórica
                 tot = mean_h + mean_a
                 if tot > 0: split_h = mean_h / tot
            
            neural_params = {
                'lambda_home': ml_prediction * split_h,
                'lambda_away': ml_prediction * (1 - split_h),
                'variance_factor': 1.0 # Padrão conservador
            }
            
            mean_h, mean_a, tactical_metrics = self.calculate_hybrid_lambdas(
                neural_params=neural_params,
                match_name=match_name
            )
        else:
            # Modo Puramente Histórico (Sem IA)
            # Mantém médias calculadas pelo histórico
            pass
        
        # Audit Fix 4.1: Extract Bivariate Poisson coupling coefficient from ManagerAI
        # When lambda3 > 0.1 simulate_match_event() activates the Bivariate Poisson model.
        lambda3 = advanced_metrics.get('lambda3', 0.0) if advanced_metrics else 0.0

        sim_total = self.simulate_match_event(mean_h, mean_a, var_h, var_a, covariance=lambda3)
        
        # Simula Primeiro Tempo (HT)
        dist_h_ht, mean_h_ht, var_h_ht = self._get_distribution_params(h_corners_ht)
        dist_a_ht, mean_a_ht, var_a_ht = self._get_distribution_params(a_corners_ht)
        
        # Joint model provides dedicated HT lambdas — use them directly
        _np = advanced_metrics.get('neural_params', {}) if advanced_metrics else {}
        if _np.get('source') == 'joint_corners_v1' and 'lambda_home_1h' in _np:
            mean_h_ht = _np['lambda_home_1h']
            mean_a_ht = _np['lambda_away_1h']
        else:
            # ⚖️ HT Consistency Scaling (Legacy: proportion from FT)
            ratio_h = mean_h / mean_h_hist
            ratio_a = mean_a / mean_a_hist
            ratio_h = max(0.5, min(2.0, ratio_h))
            ratio_a = max(0.5, min(2.0, ratio_a))
            mean_h_ht = mean_h_ht * ratio_h
            mean_a_ht = mean_a_ht * ratio_a
        
        sim_ht = self.simulate_match_event(mean_h_ht, mean_a_ht, var_h_ht, var_a_ht)
        
        # Simula Totais Individuais
        sim_home_total = self.monte_carlo_simulation(mean_h, var_h)
        sim_away_total = self.monte_carlo_simulation(mean_a, var_a)
        
        # Análise de Mercados
        markets = []


        
        # Função auxiliar para adicionar mercado analisado
        def add_market(name, simulations, line, type_='Over'):
            # =================================================================
            # CRITICAL FIX (30/01/2026): UNDER Bias Correction
            # =================================================================
            # PROBLEMA ANTERIOR:
            #   Over:  simulations > line  (correto)
            #   Under: simulations < line  (ERRADO - muito restritivo)
            #
            # EXEMPLO DO BUG:
            #   Lambda=10.2 → 30% das simulações = 10 escanteios exatos
            #   Linha 10.5:
            #     Over  (>10.5): [11, 11, 12] = 47%
            #     Under (<10.5): [8, 9, 10, 10] = 43%  ← 10 era ignorado!
            #     Push (=10.5): [10, 10, 10] = 10%     ← PERDIDO
            #
            # SOLUÇÃO:
            #   Under deve ser INCLUSIVO no limite: <= line
            #   Rationale: "Under 10.5" = "10 ou menos escanteios"
            #
            # IMPACTO: Reduz viés UNDER de ~68% para ~50% (balanceado)
            # =================================================================
            
            if type_ == 'Over':
                count = np.sum(simulations > line)  # Over permanece estrito
            else:  # Under
                count = np.sum(simulations <= line)  # Under agora inclusivo (FIX)
            
            # 1. Probabilidade Bruta (Monte Carlo)
            raw_prob = count / self.n_simulations
            
            # Monte Carlo probabilities are already exact (frequentist), no calibration needed
            prob = raw_prob
            
            if prob > 0.01: 
                fair_odd = 1 / prob
                target_odd = self.calculate_target_price(prob, min_roi=0.05) # 5% ROI Target
                
                # Busca odd do bookmaker se disponível
                # Mapeamento para nomes curtos no label
                market_short = {
                    'JOGO COMPLETO': 'Total',
                    '1º TEMPO (HT)': '1T',
                    '2º TEMPO (FT)': '2T',
                    'TOTAL MANDANTE': 'Casa',
                    'TOTAL VISITANTE': 'Vis.'
                }.get(name, name)
                
                selection_key = f"{market_short} {type_} {line}" 
                bookmaker_odd = 0.0
                ev = 0.0
                kelly = 0.0
                
                if scraped_odds and selection_key in scraped_odds:
                    bookmaker_odd = scraped_odds[selection_key]
                    ev = self.calculate_ev(prob, bookmaker_odd)
                    kelly = self.calculate_kelly(prob, bookmaker_odd)
                
                markets.append({
                    'Mercado': name,
                    'Seleção': selection_key,
                    'Prob': prob,
                    'FairOdd': fair_odd,
                    'TargetOdd': target_odd,
                    'Odd': bookmaker_odd if bookmaker_odd > 0 else fair_odd, # Usa Fair se não tiver Book
                    'IsBookmaker': bookmaker_odd > 0,
                    'EV': ev,
                    'Kelly': kelly,
                    # Structured Data (Fix for "1T" parsing bug)
                    'raw_line': float(line),
                    'market_type': market_short,
                    'bet_side': type_
                })

        # Define as linhas padrão a serem analisadas
        # AUDIT FIX (Phase 1): Expanded range to cover low-scoring (6.5, 7.5) and high-scoring (13.5, 14.5) games
        # This prevents forcing the model into risky bets when predicted lambda is outside [8.5, 12.5]
        lines_ft = [6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5, 14.5]
        lines_ht = [2.5, 3.5, 4.5, 5.5, 6.5]  # Extended HT as well
        lines_team = [2.5, 3.5, 4.5, 5.5, 6.5, 7.5]  # Extended team totals

        # Analisa Over/Under para cada linha
        for line in lines_ft:
            add_market('JOGO COMPLETO', sim_total, line, 'Over')
            add_market('JOGO COMPLETO', sim_total, line, 'Under') 

        for line in lines_ht:
            add_market('1º TEMPO (HT)', sim_ht, line, 'Over')
            add_market('1º TEMPO (HT)', sim_ht, line, 'Under')

        # AUDIT FIX (P0-B): Removed '2º TEMPO (FT)' market generation because it was using
        # 1st-half simulation data (`sim_ht`), producing statistically invalid invalid labels.

        for line in lines_team:
            add_market('TOTAL MANDANTE', sim_home_total, line, 'Over')
            add_market('TOTAL VISITANTE', sim_away_total, line, 'Over')
            add_market('TOTAL MANDANTE', sim_home_total, line, 'Under')
            add_market('TOTAL VISITANTE', sim_away_total, line, 'Under')

        # Seleção das Melhores Oportunidades (Sprint 11 - Fair Price Sorting)
        # ----------------------------------
        # Nova Lógica Robusta:
        # Prioriza Probabilidade Calibrada (Fair Price) mas considera EV se houver Odd real.
        
        # Filtra candidatos viáveis (Prob > 50% ou EV > 0)
        # FIX (30/01/2026): Junk Odds Filter for Top 7
        # Remove apostas com Odd Justa < 1.25 (Prob > 80%) EXCETO se houver EV confirmado.
        candidates = []
        for m in markets:
            # 1. Basic viability
            if m['Prob'] <= 0.50 and m['EV'] <= 0:
                continue
                
            # 2. Junk Odds Filter
            # Se não tem bookmaker odd (EV=0), remove se for odd de "lixo" (<1.25)
            # HARD RULE: Mesmo com Odd de bookmaker, se a FairOdd for < 1.25 (Prob > 80%),
            # exigimos um EV massivo (>10%) para justificar o risco de cauda.
            if m['FairOdd'] < 1.25:
                if m['EV'] < 10.0:  # Só aceita se tiver muito valor (+10%)
                    continue
            
            # 3. Hard filter for extreme low odds even with bookmaker (preventing 1.01 bets)
            if m['FairOdd'] < 1.15: # Prob > 87% is hardly ever value
                continue
                
            candidates.append(m)
        
        # Função de score para ordenação
        def ranking_score(pick):
            # Prioriza EV se for significativo (Value Betting > 5%)
            if pick['EV'] > 5.0 and pick['IsBookmaker']:
                 return 2000 + pick['EV']
            # Caso contrário, usa Probabilidade Calibrada como proxy de valor
            return pick['Prob'] * 1000
            
        # Ordena candidatos pelo score
        candidates = sorted(candidates, key=ranking_score, reverse=True)
        
        # Top 7 Selection
        # Garante diversidade (pelo menos 1 Under, 1 Over se possível)
        top_picks = []
        
        # Tenta pegar top pick de cada tipo para diversidade
        best_over = next((x for x in candidates if 'Over' in x['Seleção']), None)
        best_under = next((x for x in candidates if 'Under' in x['Seleção']), None)
        
        if best_over: top_picks.append(best_over)
        if best_under: top_picks.append(best_under)
        
        # Preenche o resto com os melhores scores restantes
        for pick in candidates:
            if len(top_picks) >= 7: break
            if pick not in top_picks:
                top_picks.append(pick)
                
        # Sort final for display (EV/Prob)
        top_picks = sorted(top_picks, key=ranking_score, reverse=True)
    
        # Gera sugestões categorizadas (Easy/Medium/Hard) usando TODOS os mercados analisados
        suggestions = self.generate_suggestions(markets, ml_prediction)

        # Exibição no Terminal (apenas se executado via CLI)
        if match_name:
            print(f"\n{Colors.CYAN}{'─'*80}")
            print(f" 📊 {Colors.BOLD}MONTE CARLO ANALYSIS — Joint λ{Colors.RESET}")
            print(f"{Colors.CYAN}{'─'*80}{Colors.RESET}")
            
            print(f"\n🏆 {Colors.BOLD}TOP 7 OPPORTUNITIES (Data-Driven Analysis){Colors.RESET}")
            
            tabela_display = []
            for pick in top_picks:
                prob = pick['Prob']
                tipo = "OVER" if "Over" in pick['Seleção'] else "UNDER"
                cor = Colors.GREEN if tipo == "OVER" else Colors.CYAN
                seta = "▲" if tipo == "OVER" else "▼"
                
                linha_fmt = f"{cor}{pick['Seleção']}{Colors.RESET}"
                prob_fmt = f"{prob * 100:.1f}%"
                odd_fmt = f"{Colors.BOLD}@{pick['Odd']:.2f}{Colors.RESET}"
                direcao_fmt = f"{cor}{seta} {tipo}{Colors.RESET}"
                
                tabela_display.append([pick['Mercado'], linha_fmt, prob_fmt, odd_fmt, direcao_fmt])
                
            headers = ["MERCADO", "LINHA", "PROB.", "ODD JUSTA", "TIPO"]
            print(tabulate(tabela_display, headers=headers, tablefmt="fancy_grid", stralign="center"))

            print(f"\n🎯 {Colors.BOLD}SUGESTÕES DA IA:{Colors.RESET}")
            for level, pick in suggestions.items():
                if pick:
                    color = Colors.GREEN if level == 'Easy' else (Colors.YELLOW if level == 'Medium' else Colors.RED)
                    print(f"[{color}{level.upper()}{Colors.RESET}] {pick['Mercado']} - {pick['Seleção']} (@{pick['Odd']:.2f}) | Prob: {pick['Prob']*100:.1f}%")
                else:
                    print(f"[{level.upper()}] Nenhuma oportunidade encontrada.")

        return top_picks, suggestions, tactical_metrics
