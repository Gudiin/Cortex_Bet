import pandas as pd
import numpy as np
import warnings
from datetime import datetime

# Suprime avisos de performance do Pandas para melhorar a User Experience no CLI
warnings.simplefilter('ignore', pd.errors.PerformanceWarning)
# ============================================================================
# FUNÇÕES AUXILIARES - MELHORIAS V5 (Auditoria ML)
# ============================================================================

def exponential_decay_weight(days_ago: float, half_life: float = 14.0) -> float:
    """
    Peso exponencial baseado em física de decaimento radioativo.
    
    Jogos recentes têm mais peso que jogos antigos.
    
    Args:
        days_ago: Dias desde o jogo
        half_life: Tempo para peso reduzir à metade (default: 14 dias)
    
    Returns:
        Peso entre 0 e 1
    
    Fórmula:
        w(t) = exp(-λ * t)
        onde λ = ln(2) / half_life
    """
    decay_constant = np.log(2) / half_life
    return np.exp(-decay_constant * days_ago)


def calculate_entropy(values: pd.Series) -> float:
    """
    Calcula entropia de Shannon normalizada para série de resultados.
    
    Alta entropia = time imprevisível (resultados variam muito)
    Baixa entropia = time consistente (resultados similares)
    
    Args:
        values: Série de escanteios
    
    Returns:
        Entropia normalizada entre 0 e 1
    
    Fórmula:
        H = -Σ p(x) * log2(p(x))
        Normalizada: H / log2(n_bins)
    """
    if len(values) < 3:
        return 0.5  # Neutro sem dados
    
    # Discretiza em bins (0-5, 5-10, 10-15, 15+)
    bins = [0, 5, 10, 15, 100]
    hist, _ = np.histogram(values, bins=bins)
    
    # Remove zeros
    hist = hist[hist > 0]
    if len(hist) == 0:
        return 0.5
        
    probs = hist / hist.sum()
    
    # Entropia de Shannon
    entropy = -np.sum(probs * np.log2(probs))
    max_entropy = np.log2(len(bins) - 1)
    
    return entropy / max_entropy if max_entropy > 0 else 0

def create_advanced_features(df: pd.DataFrame, window_short: int = 3, window_long: int = 5) -> tuple:
    """
    Pipeline unificado de Feature Engineering (Vetorizado e Anti-Leakage) - V3 (Advanced).
    
    Gera:
    1. Médias móveis Gerais (Momentum)
    2. Médias móveis Específicas (Home/Away)
    3. Médias de Concessão (Defesa)
    4. H2H (Confronto Direto)
    5. Trend (Curto vs Longo Prazo) - NOVO
    6. Volatilidade (Desvio Padrão) - NOVO
    7. Rest Days (Cansaço) - NOVO
    8. EMA (Exponential Moving Average) - NOVO
    9. Força Relativa (Interações) - NOVO
    
    Returns:
        X (pd.DataFrame): Features
        y (pd.Series): Target (Total Corners)
        timestamps (pd.Series): Data do jogo (para validação temporal)
    """
    # 1. Ordenação Temporal Obrigatória
    df = df.sort_values('start_timestamp').copy()
    
    # --- CORREÇÃO DE COMPATIBILIDADE (Tournament ID) ---
    if 'tournament_id' not in df.columns:
        if 'tournament_name' in df.columns:
            df['tournament_id'] = df['tournament_name']
        else:
            df['tournament_id'] = 'Unknown'
    # ---------------------------------------------------
    
    # 2. Estratégia "Team-Centric" (Transforma Partida em Linhas de Time)
    # Added 'momentum_home' and 'momentum_away'
    cols_metrics = ['corners_ft', 'shots_ot_ft', 'goals_ft', 'corners_ht', 'dangerous_attacks_ft', 'momentum_home']
    
    # Home Stats
    # Home Stats
    # Colunas: match_id, start_ts, tourn_id, team_id, opp_id, corners, shots, goals, goals_conceded, corners_ht, corners_conceded_cantos, da, blocked, crosses, tackles, interc, clearances, recov, xg, possession, is_home
    home_cols = ['match_id', 'start_timestamp', 'tournament_id', 'home_team_id', 'away_team_id', 'corners_home_ft', 'shots_ot_home_ft', 'home_score', 'away_score', 'corners_home_ht', 'corners_away_ft', 'dangerous_attacks_home', 'blocked_shots_home', 'crosses_home', 'tackles_home', 'interceptions_home', 'clearances_home', 'recoveries_home']
    home_names = ['match_id', 'start_timestamp', 'tournament_id', 'team_id', 'opponent_id', 'corners', 'shots', 'goals', 'goals_conceded', 'corners_ht', 'corners_conceded', 'dangerous_attacks', 'blocked_shots', 'crosses', 'tackles', 'interceptions', 'clearances', 'recoveries']
    
    # Add xG and possession if available in DataFrame
    has_xg = 'expected_goals_home' in df.columns and 'expected_goals_away' in df.columns
    has_possession = 'possession_home' in df.columns and 'possession_away' in df.columns
    
    if has_xg:
        home_cols.append('expected_goals_home')
        home_names.append('xg')
    if has_possession:
        home_cols.append('possession_home')
        home_names.append('possession')
    
    df_home = df[home_cols].copy()
    df_home.columns = home_names
    df_home['is_home'] = 1
    
    # Away Stats
    away_cols = ['match_id', 'start_timestamp', 'tournament_id', 'away_team_id', 'home_team_id', 'corners_away_ft', 'shots_ot_away_ft', 'away_score', 'home_score', 'corners_away_ht', 'corners_home_ft', 'dangerous_attacks_away', 'blocked_shots_away', 'crosses_away', 'tackles_away', 'interceptions_away', 'clearances_away', 'recoveries_away']
    away_names = ['match_id', 'start_timestamp', 'tournament_id', 'team_id', 'opponent_id', 'corners', 'shots', 'goals', 'goals_conceded', 'corners_ht', 'corners_conceded', 'dangerous_attacks', 'blocked_shots', 'crosses', 'tackles', 'interceptions', 'clearances', 'recoveries']
    
    if has_xg:
        away_cols.append('expected_goals_away')
        away_names.append('xg')
    if has_possession:
        away_cols.append('possession_away')
        away_names.append('possession')
    
    df_away = df[away_cols].copy()
    df_away.columns = away_names
    df_away['is_home'] = 0
    
    # Stack de todos os jogos na visão do time
    team_stats = pd.concat([df_home, df_away]).sort_values(['team_id', 'start_timestamp'])
    
    # Robustez: Garante que start_timestamp é datetime para evitar erro .dt accessor
    if not np.issubdtype(team_stats['start_timestamp'].dtype, np.datetime64):
         team_stats['start_timestamp'] = pd.to_datetime(team_stats['start_timestamp'], unit='s')
         
    # --- ROBUST FEATURE ENGINEERING (Sprint 8) ---
    # Aplica mesma lógica de Winsorization (min=3.0) usada na inferência
    # Evita que o modelo aprenda padrões ruidosos de jogos com 1 escanteio
    if 'corners' in team_stats.columns:
        team_stats['corners'] = team_stats['corners'].clip(lower=3.0)
    if 'corners_conceded' in team_stats.columns:
        team_stats['corners_conceded'] = team_stats['corners_conceded'].clip(lower=3.0)
    
    # 3. Engenharia Vetorizada
    grouped = team_stats.groupby('team_id')
    
    feature_cols = ['corners', 'shots', 'goals', 'corners_conceded', 'dangerous_attacks', 'corners_ht', 'blocked_shots', 'crosses', 'tackles', 'interceptions', 'clearances', 'recoveries']
    # NOTE: 'momentum' and 'momentum_conceded' REMOVED (97.6% of raw DB values are 0.0,
    # causing EMA/std to produce nonsensical values >1000 when mixing zeros with real values).
    # See: match_stats table — only 299/12655 matches have non-zero momentum data.
    
    # V11: Add xG and possession if available (79.2% and 99.8% coverage respectively)
    if has_xg:
        feature_cols.append('xg')
    if has_possession:
        feature_cols.append('possession')
    
    # --- A. Features Temporais (Rest Days) ---
    team_stats['prev_timestamp'] = grouped['start_timestamp'].shift(1)
    # Fix: Convert Timedelta to float days
    team_stats['rest_days'] = (team_stats['start_timestamp'] - team_stats['prev_timestamp']).dt.total_seconds() / 86400
    # Imputação Inteligente: Mediana global (mais robusto que fixo 7)
    global_median_rest = team_stats['rest_days'].median()
    if pd.isna(global_median_rest):
        global_median_rest = 7.0 # Fallback se tudo for nan
    team_stats['rest_days'] = team_stats['rest_days'].fillna(global_median_rest)
    # Clip para evitar outliers (ex: 100 dias de pausa)
    team_stats['rest_days'] = team_stats['rest_days'].clip(0, 14)
    
    # --- A2. Resultado Anterior (Win, Draw, Loss) ---
    # Vectorized calculation of result
    team_stats['game_result'] = np.where(team_stats['goals'] > team_stats['goals_conceded'], 'win',
                                        np.where(team_stats['goals'] == team_stats['goals_conceded'], 'draw', 'loss'))
    
    # P1-C FIX: Usa o game_result agrupado e mapeado para (Win=1, Draw=0.5, Loss=0)
    # em vez da proxy incorreta de gols > 1.
    result_map = {'win': 1.0, 'draw': 0.5, 'loss': 0.0}
    team_stats['last_result'] = grouped['game_result'].transform(
        lambda x: x.map(result_map).shift(1)
    ).fillna(0.5)  # Neutro se não há histórico
    
    # --- B. Médias Móveis Dinâmicas (Dynamic Windows) ---
    # Especialista: Substituir janelas fixas por múltiplas janelas
    windows = [3, 5, 10]
    
    for w in windows:
        for col in feature_cols:
            # Rolling Mean (MANTIDO - Essenciais para compatibilidade com Model V2)
            team_stats[f'avg_{col}_{w}g'] = grouped[col].transform(
                lambda x: x.shift(1).rolling(window=w, min_periods=1).mean()
            )
            
            # EMA (Exponential Moving Average) - MANTIDO (SIGNAL)
            # FIX: Added shift(1) to prevent leakage
            team_stats[f'ema_{col}_{w}g'] = grouped[col].transform(
                lambda x: x.shift(1).ewm(span=w, adjust=False).mean()
            )
            # Volatilidade (Std Dev)
            # FIX: Added shift(1) to prevent leakage
            team_stats[f'std_{col}_{w}g'] = grouped[col].transform(
                lambda x: x.shift(1).rolling(window=w).std()
            ).fillna(0)
            
            # Resolve PerformanceWarning: Fragmented DataFrame
            # Em vez de counter, usamos uma cópia ocasional para desfragmentar
            if len(team_stats.columns) % 50 == 0:
                team_stats = team_stats.copy()

    # --- ALIASES DE RETROCOMPATIBILIDADE (ATUALIZADOS PARA EMA) ---
    # O código antigo espera 'avg_corners_general', mas agora usamos EMA
    for col in feature_cols:
        # mapeia avg -> ema (Noise Reduction)
        team_stats[f'avg_{col}_general'] = team_stats[f'ema_{col}_5g']
        team_stats[f'avg_{col}_short'] = team_stats[f'ema_{col}_3g']
        
        team_stats[f'ema_{col}_general'] = team_stats[f'ema_{col}_5g']
        team_stats[f'std_{col}_general'] = team_stats[f'std_{col}_5g']
        
        # Trend (Curto 3g - Longo 10g)
        # Necessário para compatibilidade com Model V2.1+
        team_stats[f'trend_{col}'] = team_stats[f'ema_{col}_3g'] - team_stats[f'ema_{col}_10g']

    # --- Derived Momentum Features: REMOVED ---
    # momentum_diff_5g and momentum_ratio_5g removed because raw momentum data
    # is 97.6% zeros in DB, making EMA derivatives unreliable.
    
    
    # --- E. FEATURES V5 (Auditoria ML - CORRIGIDO) ---
    # CORREÇÃO: Decaimento exponencial SEM LEAKAGE
    # Em vez de usar max_timestamp global (que inclui futuro), calculamos
    # o decaimento relativo ao timestamp do PRÓPRIO jogo atual
    
    # --- E. FEATURES V5 (Auditoria ML - CORRIGIDO VETORIZADO) ---
    # CORREÇÃO V2: Vectorized Exponential Decay usando Pandas EWM
    # Complexidade: de O(N^2) para O(N)
    
    def calculate_decay_weighted_avg_vectorized(group):
        # Garante ordenação temporal - group here is a DataFrame because we apply on grouped df
        group = group.sort_values('start_timestamp')
        
        try:
             times = pd.to_datetime(group['start_timestamp'], unit='s')
        except Exception:
             return group['corners'].ewm(halflife=5).mean().shift(1)
             
        weighted_avg = group['corners'].ewm(halflife='14 days', times=times).mean().shift(1)
        return weighted_avg
    
    # Revert to passing the full dataframe group to access 'start_timestamp'
    decay_result = grouped.apply(calculate_decay_weighted_avg_vectorized, include_groups=False)
    
    # Extract Series from result
    if isinstance(decay_result, pd.Series):
        decay_series = decay_result
    else:
        # Fallback if it returns a DataFrame (unlikely with this return)
        decay_series = decay_result.iloc[:, 0]

    # Garante alinhamento de índice
    if decay_series.index.nlevels > 1:
        decay_series = decay_series.reset_index(level=0, drop=True)
        
    team_stats['decay_weighted_corners'] = decay_series
    team_stats['decay_weighted_corners'] = team_stats['decay_weighted_corners'].fillna(team_stats['avg_corners_general'])
    
    # Entropia (imprevisibilidade) - alta = time instável
    # OPTIMIZATION: Rolling apply with custom python function is O(N*W) and very slow.
    # P1-B FIX: Removed dummy entropy_corners entirely since it was hardcoded and dropped.

    # --- C. Médias ESPECÍFICAS (Home/Away) ---
    home_games = team_stats[team_stats['is_home'] == 1].sort_values(['team_id', 'start_timestamp'])
    away_games = team_stats[team_stats['is_home'] == 0].sort_values(['team_id', 'start_timestamp'])
    
    for col in feature_cols:
        # Média EM CASA
        home_games[f'avg_{col}_home'] = home_games.groupby('team_id')[col].transform(
            lambda x: x.shift(1).rolling(window=window_long, min_periods=1).mean()
        )
        # Média FORA
        away_games[f'avg_{col}_away'] = away_games.groupby('team_id')[col].transform(
            lambda x: x.shift(1).rolling(window=window_long, min_periods=1).mean()
        )

    # Merge de volta
    team_stats = team_stats.merge(
        home_games[['match_id', 'team_id'] + [f'avg_{col}_home' for col in feature_cols]], 
        on=['match_id', 'team_id'], how='left'
    )
    team_stats = team_stats.merge(
        away_games[['match_id', 'team_id'] + [f'avg_{col}_away' for col in feature_cols]], 
        on=['match_id', 'team_id'], how='left'
    )
    
    # Fillna com média geral
    for col in feature_cols:
        team_stats[f'avg_{col}_home'] = team_stats[f'avg_{col}_home'].fillna(team_stats[f'avg_{col}_general'])
        team_stats[f'avg_{col}_away'] = team_stats[f'avg_{col}_away'].fillna(team_stats[f'avg_{col}_general'])

    # --- D. H2H (Confronto Direto) ---
    team_stats = team_stats.sort_values(['team_id', 'opponent_id', 'start_timestamp'])
    h2h_grouped = team_stats.groupby(['team_id', 'opponent_id'])
    
    for col in ['corners', 'corners_conceded']:
        team_stats[f'avg_{col}_h2h'] = h2h_grouped[col].transform(
            lambda x: x.shift(1).rolling(window=3, min_periods=1).mean()
        )
    
    team_stats['avg_corners_h2h'] = team_stats['avg_corners_h2h'].fillna(team_stats['avg_corners_general'])
    team_stats['avg_corners_conceded_h2h'] = team_stats['avg_corners_conceded_h2h'].fillna(team_stats['avg_corners_conceded_general'])
    
    # --- F. STRENGTH OF SCHEDULE (SoS) - Auditoria V6 ---
    # Mede a força média dos adversários enfrentados
    # Um time que faz 10 escanteios contra o lanterna != 10 contra o líder
    
    # 1. Força defensiva de cada time (proxy: escanteios que cede)
    team_stats = team_stats.sort_values(['team_id', 'start_timestamp'])
    grouped = team_stats.groupby('team_id')  # Regrouping after sort
    
    team_stats['own_defense_strength'] = grouped['corners_conceded'].transform(
        lambda x: x.shift(1).rolling(window=10, min_periods=3).mean()
    ).fillna(5.0)  # Média global como fallback
    
    # 2. Para cada jogo, traz a força defensiva do oponente
    # Cria mapeamento: match_id + team_id -> own_defense_strength
    defense_map = team_stats[['match_id', 'team_id', 'own_defense_strength']].copy()
    defense_map.columns = ['match_id', 'opponent_id', 'opponent_defense_strength']
    
    team_stats = team_stats.merge(
        defense_map,
        on=['match_id', 'opponent_id'],
        how='left'
    )
    team_stats['opponent_defense_strength'] = team_stats['opponent_defense_strength'].fillna(5.0)
    
    # 3. SoS Rolling = Média da força dos oponentes enfrentados recentemente
    grouped = team_stats.groupby('team_id')  # Regrouping
    team_stats['sos_rolling'] = grouped['opponent_defense_strength'].transform(
        lambda x: x.shift(1).rolling(window=5, min_periods=1).mean()
    ).fillna(5.0)
    
    # --- F2. LEAGUE RELATIVE STRENGTH (Ajuste por Nível da Liga) - V2.1 ---
    # Normaliza a força da defesa do oponente comparada à média da liga
    
    # CRITICAL FIX: Sort by time for expanding mean across tournament (Anti-Leakage)
    # Added deterministic tie-breakers to ensure stable expanding window
    team_stats = team_stats.sort_values(['start_timestamp', 'match_id', 'is_home'])
    
    # 1. Calcula médias da LIGA (expanding window shift=1)
    team_stats['league_avg_conceded'] = team_stats.groupby('tournament_id')['corners_conceded'].transform(
        lambda x: x.expanding().mean().shift(1)
    ).fillna(4.5)
    
    team_stats['league_avg_corners'] = team_stats.groupby('tournament_id')['corners'].transform(
        lambda x: x.expanding().mean().shift(1)
    ).fillna(4.5)

    # 2. Força Relativa
    team_stats['relative_opponent_defense'] = team_stats['opponent_defense_strength'] - team_stats['league_avg_conceded']
    team_stats['relative_opponent_defense'] = team_stats['relative_opponent_defense'].fillna(0.0)

    # RESTORE SORT: Team-centric for subsequent rolling ops
    team_stats = team_stats.sort_values(['team_id', 'start_timestamp'])

    
    # --- G. GAME STATE (Comportamento Histórico) - Auditoria V7 ---
    # Mede como o time se comporta quando está ganhando vs perdendo
    # Usa resultado PASSADO para prever comportamento futuro
    
    # 1. Determina resultado de cada jogo (gol do time vs gol do oponente)
    team_stats['game_result'] = team_stats.apply(
        lambda row: 'win' if row['goals'] > row.get('goals_conceded', 0) 
                    else ('loss' if row['goals'] < row.get('goals_conceded', 0) else 'draw'),
        axis=1
    )
    
    # 2. Média de escanteios quando PERDEU (histórico)
    # 2. Média de escanteios quando PERDEU (histórico)
    # VECTORIZED OPTIMIZATION (No loop)
    def calculate_conditional_avg(df_full, result_type):
        """Calcula média condicional vetorizada (ex: média de corners nos últimos 5 jogos onde perdeu)."""
        # Filtra apenas jogos do tipo desejado (ex: 'loss')
        subset = df_full[df_full['game_result'] == result_type].copy()
        subset = subset.sort_values(['team_id', 'start_timestamp'])
        
        # Calcula rolling mean nesse subset
        subset['rolling_avg'] = subset.groupby('team_id')['corners'].transform(
             lambda x: x.shift(1).rolling(window=5, min_periods=1).mean()
        )
        
        # Merge back to full dataset via merge_asof (time-travel safe)
        # Precisamos de 'asof' merge para pegar o valor mais recente 'as of' data do jogo atual
        # Mas para simplificar e ser robusto:
        # Vamos fazer um merge left on match_id? Não, porque o jogo atual pode não ser 'loss'.
        # O que queremos é: Para o jogo atual (tempo T), qual era a 'rolling_avg' do subset 'loss' em tempo < T?
        
        # Estratégia simples: Forward Fill (ffill) dos valores calculados
        # 1. Cria série vazia alinhada com df_full
        # 2. Preenche com valores nos timestamps onde houve jogo 'loss'
        # 3. Faz ffill por grupo
        
        res = df_full[['match_id', 'team_id', 'start_timestamp']].copy()
        subset_vals = subset[['match_id', 'rolling_avg']]
        
        res = res.merge(subset_vals, on='match_id', how='left')
        
        # Propaga o último valor conhecido para frente (dentro do grupo de time)
        res['rolling_avg'] = res.groupby('team_id')['rolling_avg'].ffill()
        
        return res['rolling_avg']

    # Corners quando perde
    team_stats['avg_corners_when_losing'] = calculate_conditional_avg(team_stats, 'loss')
    
    # Corners quando ganha
    team_stats['avg_corners_when_winning'] = calculate_conditional_avg(team_stats, 'win')
    
    # Fillna com média geral
    team_stats['avg_corners_when_losing'] = team_stats['avg_corners_when_losing'].fillna(team_stats['avg_corners_general'])
    team_stats['avg_corners_when_winning'] = team_stats['avg_corners_when_winning'].fillna(team_stats['avg_corners_general'])
    
    # 3. Desperation Index = corners quando perde - corners quando ganha
    # Positivo = time ataca mais quando perde (desesperado)
    # Negativo = time recua quando perde (defensivo)
    team_stats['desperation_index'] = team_stats['avg_corners_when_losing'] - team_stats['avg_corners_when_winning']

    # 4. Reconstrução do Dataset de Partidas
    # --- H. SCIENTIFIC FEATURES (2025) - PPG & Strength Ratings ---
    # Implementação baseada em literatura científica (Dixon-Coles, Poisson)
    
    # 1. Points Per Game (PPG) - Proxy para posição na tabela
    # Mapeia: Win=3, Draw=1, Loss=0
    points_map = {'win': 3, 'draw': 1, 'loss': 0}
    team_stats['match_points'] = team_stats['game_result'].map(points_map).fillna(1)
    
    team_stats['rolling_ppg_20g'] = team_stats.groupby('team_id')['match_points'].transform(
        lambda x: x.shift(1).rolling(window=10, min_periods=3).mean()
    ).fillna(1.3) # 1.3 é uma média global razoável (~45 pontos em 38 jogos)

    # 2. League Averages (Contexto da Liga)
    # Already calculated in F2 block (sorted by time) to prevent leakage.
    pass

    # 3. Strength Ratings (Normalizados pela Liga)
    # Ratio > 1.0 = Acima da média da liga
    # Avoid div by zero
    team_stats['attack_strength'] = team_stats['avg_corners_general'] / team_stats['league_avg_corners'].replace(0, 4.5)
    team_stats['defense_strength'] = team_stats['avg_corners_conceded_general'] / team_stats['league_avg_conceded'].replace(0, 4.5)

    stats_home = team_stats[team_stats['is_home'] == 1].add_prefix('home_')
    stats_away = team_stats[team_stats['is_home'] == 0].add_prefix('away_')
    
    stats_home = stats_home.rename(columns={'home_match_id': 'match_id'})
    stats_away = stats_away.rename(columns={'away_match_id': 'match_id'})
    
    df_features = df[['match_id', 'start_timestamp', 'tournament_id', 'corners_home_ft', 'corners_away_ft']].merge(
        stats_home, on='match_id', how='inner'
    ).merge(
        stats_away, on='match_id', how='inner'
    )
    
    # 5. Features de Interação (Força Relativa) - NOVO
    # Ataque Casa vs Defesa Visitante
    df_features['home_attack_adv'] = df_features['home_avg_corners_home'] - df_features['away_avg_corners_conceded_away']
    
    # 6. Scientific Poisson Models (Dixon-Coles Proxy)
    # Expected Corners = Attack Strength (Home) * Defense Strength (Away) * League Avg
    df_features['expected_corners_poisson_home'] = (
        df_features['home_attack_strength'] * 
        df_features['away_defense_strength'] * 
        df_features['home_league_avg_corners']
    )
    
    df_features['expected_corners_poisson_away'] = (
        df_features['away_attack_strength'] * 
        df_features['home_defense_strength'] * 
        df_features['away_league_avg_corners']
    )
    
    # Diferencial de PPG (Table Mismatch)
    # Positivo = Mandante é muito superior na tabela
    df_features['ppg_differential'] = df_features['home_rolling_ppg_20g'] - df_features['away_rolling_ppg_20g']
    # Ataque Visitante vs Defesa Casa
    df_features['away_attack_adv'] = df_features['away_avg_corners_away'] - df_features['home_avg_corners_conceded_home']
    
    # H2H Dominance (Quem domina o confronto direto comparado à média geral)
    df_features['home_h2h_dominance'] = df_features['home_avg_corners_h2h'] - df_features['home_avg_corners_general']
    df_features['away_h2h_dominance'] = df_features['away_avg_corners_h2h'] - df_features['away_avg_corners_general']
    
    # Diferença de Momentum (Quem está em melhor fase?)
    # Usamos avg_corners_general (EMA 5g) como proxy de força recente
    df_features['momentum_diff'] = df_features['home_avg_corners_general'] - df_features['away_avg_corners_general']
    
    # Diferença de Cansaço
    df_features['rest_diff'] = df_features['home_rest_days'] - df_features['away_rest_days']
    
    # --- 7. Feature de Pressão (Pressure Ratio) - NOVO ---
    # Mede a eficiência em converter pressão (chutes) em escanteios
    df_features['home_pressure_ratio'] = df_features['home_avg_corners_general'] / (df_features['home_avg_shots_general'] + 1.0)
    df_features['away_pressure_ratio'] = df_features['away_avg_corners_general'] / (df_features['away_avg_shots_general'] + 1.0)

    # --- 8. QUANTITATIVE FEATURES V9 (Dangerous Attacks) ---
    # DA Efficiency: Quantos Dangerous Attacks precisa para gerar 1 escanteio?
    # Valor alto = ineficiente. Valor baixo = letal.
    # Invertendo (Corners / DA) para que maior seja melhor
    # Smooth de +1.0 para evitar explosão em casos de 0 ataques (dados faltantes)
    df_features['home_da_efficiency'] = df_features['home_avg_corners_general'] / (df_features['home_avg_dangerous_attacks_general'] + 1.0)
    df_features['away_da_efficiency'] = df_features['away_avg_corners_general'] / (df_features['away_avg_dangerous_attacks_general'] + 1.0)
    
    # Pressure Index V2: (DA + Shots) Agregados
    df_features['home_pressure_index'] = df_features['home_avg_dangerous_attacks_general'] + (df_features['home_avg_shots_general'] * 2)
    df_features['away_pressure_index'] = df_features['away_avg_dangerous_attacks_general'] + (df_features['away_avg_shots_general'] * 2)
    
    # --- 9. V11 FEATURES (xG & Possession — Scientific Enhancement) ---
    # xG: strong monotonic correlation with corners (+16% across quartiles)
    # Possession: non-linear relationship — moderate possession (~55-60%) correlates with highest corners
    if has_xg:
        # xG differential: which team creates more quality chances?
        df_features['xg_differential'] = df_features['home_ema_xg_5g'] - df_features['away_ema_xg_5g']
        # xG per corner: how many expected goals per corner? (efficiency proxy)
        df_features['home_xg_per_corner'] = df_features['home_ema_xg_5g'] / (df_features['home_avg_corners_general'] + 0.1)
        df_features['away_xg_per_corner'] = df_features['away_ema_xg_5g'] / (df_features['away_avg_corners_general'] + 0.1)
    
    if has_possession:
        # Possession differential
        df_features['possession_diff'] = df_features['home_ema_possession_5g'] - df_features['away_ema_possession_5g']
    
    df_features['tournament_id'] = df_features['tournament_id'].astype('category')
    
    # --- 6. Novas Features V4 ---
    # Fase da Temporada (0=início, 0.5=meio, 1=fim)
    # Assume 38 rodadas padrão, ajusta baseado no timestamp relativo ao torneio
    if 'round' in df.columns:
        df_features = df_features.merge(
            df[['match_id', 'round']], on='match_id', how='left'
        )
        df_features['season_stage'] = df_features['round'].fillna(19) / 38
        df_features['season_stage'] = df_features['season_stage'].clip(0, 1)
    else:
        df_features['season_stage'] = 0.5  # Fallback neutro
    
    # Último Resultado
    df_features['home_last_result'] = df_features['home_last_result'] if 'home_last_result' in df_features.columns else 0.5
    df_features['away_last_result'] = df_features['away_last_result'] if 'away_last_result' in df_features.columns else 0.5
    
    # Posição na Tabela (Real reconstruída)
    if 'home_league_position' in df_features.columns and 'away_league_position' in df_features.columns:
        # Preenche Nulos com 0 ou média (se for início de campeonato, assumimos meio de tabela simulado)
        df_features['home_league_pos'] = df_features['home_league_position'].fillna(10)
        df_features['away_league_pos'] = df_features['away_league_position'].fillna(10)
        
        # Diferença de posição (Visitante - Mandante). 
        # Ex: Home=1, Away=10 -> Diff = 9 (Positivo = Vantagem Home)
        # Ex: Home=10, Away=1 -> Diff = -9 (Negativo = Desvantagem Home)
        df_features['position_diff'] = df_features['away_league_pos'] - df_features['home_league_pos']
    else:
        # Fallback para proxy antigo se reconstrução falhar
        df_features['home_form_score'] = df_features['home_avg_goals_general'] * 3
        df_features['away_form_score'] = df_features['away_avg_goals_general'] * 3
        df_features['position_diff'] = df_features['home_form_score'] - df_features['away_form_score']
        df_features['home_league_pos'] = 10
        df_features['away_league_pos'] = 10
    
    # Limpeza
    # df_features = df_features.dropna() # REMOVED: Preserve early games (with NaNs) for alignment and test validity
    
    # Definição de X e y
    # Definição de X e y
    
    # 1. Features Dinâmicas (Windows: 3, 5, 10)
    # ACADEMIC OVERHAUL: Seleção estrita para reduzir dimensionalidade (264 -> ~122)
    # Remove: SMA (substituído por EMA), Janelas redundantes, Métricas de baixo sinal
    
    dynamic_features = []
    
    # Core Metrics (high-signal metrics for dynamic EMA/std features)
    core_metrics = [
        'corners', 'shots', 'goals', 'corners_conceded', 'dangerous_attacks',
        'blocked_shots', 'crosses'
    ]
    # NOTE: 'momentum' and 'momentum_conceded' REMOVED from core_metrics
    # (same reason as feature_cols: 97.6% zeros in DB)
    
    # V11: Add xG and possession to core metrics (79.2% and 99.8% DB coverage)
    # xG: strong monotonic correlation with corners (8.84→10.26 avg corners, +16%)
    # Possession: 99.8% coverage, captures tactical dominance patterns
    if has_xg:
        core_metrics.append('xg')
    if has_possession:
        core_metrics.append('possession')
    
    for team in ['home', 'away']:
        for metric in core_metrics:
            # EMA (Exponential Moving Average) - Sinal Primário [3, 5, 10]
            for w in [3, 5, 10]:
                dynamic_features.append(f'{team}_ema_{metric}_{w}g')
            
            # Volatilidade (Std Dev) - Risco [5, 10] (3 jogos é muito ruidoso para std)
            for w in [5, 10]:
                dynamic_features.append(f'{team}_std_{metric}_{w}g')

    # 2. Features de Contexto e Específicas
    static_features = [
        # Trend & Volatilidade (Legacy aliases mantidos por segurança)
        'home_trend_corners', 'away_trend_corners',
        
        # Específicas (Home/Away - continuam fixas no longo prazo por enquanto)
        'home_avg_corners_home', 'away_avg_corners_away',
        'home_avg_corners_conceded_home', 'away_avg_corners_conceded_away',
        
        # H2H
        'home_avg_corners_h2h', 'away_avg_corners_h2h',
        
        # Interações (Força Relativa)
        'home_attack_adv', 'away_attack_adv',
        'home_h2h_dominance', 'away_h2h_dominance',
        'momentum_diff', 'rest_diff',
        
        # Contexto
        'home_rest_days', 'away_rest_days',
        'tournament_id',
        
        # V4 Features
        'season_stage',
        'position_diff',
        'home_league_pos', 'away_league_pos', # Novas features explícitas
        
        # V5 Features
        'home_decay_weighted_corners', 'away_decay_weighted_corners',
        
        # V6 Features
        'home_sos_rolling', 'away_sos_rolling',
        'home_opponent_defense_strength', 'away_opponent_defense_strength',
        'home_relative_opponent_defense', 'away_relative_opponent_defense', # V2.1 New Feature
        
        # V7 Features
        'home_desperation_index', 'away_desperation_index',
        
        # V8 Features (Pressure Ratio)
        'home_pressure_ratio', 'away_pressure_ratio',
        
        # V9 Features (Quant)
        'home_da_efficiency', 'away_da_efficiency',
        'home_pressure_index', 'away_pressure_index',
        
        # V10 Features (Scientific 2025)
        'home_rolling_ppg_20g', 'away_rolling_ppg_20g',
        'home_attack_strength', 'away_attack_strength',
        'home_defense_strength', 'away_defense_strength',
        'expected_corners_poisson_home', 'expected_corners_poisson_away',
        'ppg_differential',
    ]
    
    # V11 Features (xG & Possession — conditionally added)
    if has_xg:
        static_features.extend([
            'xg_differential',
            'home_xg_per_corner', 'away_xg_per_corner',
        ])
    if has_possession:
        static_features.extend([
            'possession_diff',
        ])

    # 3. Métricas de Exibição (Separate from Model Features)
    # These are used for UI/Feedback explanation, NOT for training the model.
    # We explicitly extract them before filtering X to ensure availability.
    display_metrics_cols = [
        'home_avg_corners_general', 'away_avg_corners_general',
        'home_std_corners_general', 'away_std_corners_general',
        'home_avg_dangerous_attacks_general', 'away_avg_dangerous_attacks_general',
        'home_avg_shots_general', 'away_avg_shots_general',
        'home_ema_crosses_5g', 'away_ema_crosses_5g',
        'home_ema_corners_5g', 'away_ema_corners_5g', # Ensure aliased columns are available
    ]
    
    # Ensure they exist (aliasing or fillna)
    for col in display_metrics_cols:
        if col not in df_features.columns:
            # Fallback logic identical to original to ensure no key errors
            if 'avg_corners_general' in col:
                team = 'home' if 'home' in col else 'away'
                alt = f'{team}_ema_corners_5g'
                df_features[col] = df_features[alt] if alt in df_features.columns else 0.0
            elif 'std_corners_general' in col:
                 team = 'home' if 'home' in col else 'away'
                 alt = f'{team}_std_corners_5g'
                 df_features[col] = df_features[alt] if alt in df_features.columns else 0.0
            elif 'avg_dangerous_attacks_general' in col:
                 team = 'home' if 'home' in col else 'away'
                 alt = f'{team}_ema_dangerous_attacks_5g'
                 df_features[col] = df_features[alt] if alt in df_features.columns else 0.0
            elif 'avg_shots_general' in col:
                 team = 'home' if 'home' in col else 'away'
                 alt = f'{team}_ema_shots_5g'
                 df_features[col] = df_features[alt] if alt in df_features.columns else 0.0
            else:
                df_features[col] = 0.0

    # Create distinct DataFrames
    feature_columns = dynamic_features + static_features
    X = df_features[feature_columns].copy()
    
    # =========================================================================
    # SPRINT 9.5 (Audit): FEATURE SELECTION - STATIONARITY
    # =========================================================================
    # AUDIT FIX (Phase 1): Removed 'home_league_pos' and 'away_league_pos' from drop list.
    # Reason: League position is a CAUSAL factor (team quality), not just temporal drift.
    # Solution: We keep position_diff (already computed above) which IS stationary.
    NON_STATIONARY_FEATURES = [
        # Dangerous Attacks (Trend/Scale issues)
        'away_ema_dangerous_attacks_3g', 'away_std_dangerous_attacks_10g', 
        'away_ema_dangerous_attacks_5g', 'away_ema_dangerous_attacks_10g',
        'away_std_dangerous_attacks_5g', 'home_ema_dangerous_attacks_3g',
        'home_std_dangerous_attacks_5g', 'home_ema_dangerous_attacks_5g', 
        'home_std_dangerous_attacks_10g'
    ]
    
    # Safe Drop
    X = X.drop(columns=[c for c in NON_STATIONARY_FEATURES if c in X.columns], errors='ignore')
    # =========================================================================

    df_display = df_features[display_metrics_cols].copy()
    
    y = df_features['corners_home_ft'] + df_features['corners_away_ft']
    timestamps = df_features['start_timestamp']
    
    return X, y, timestamps, df_display

def prepare_features_for_prediction(home_id, away_id, db_manager, window_long=5):
    """
    Prepares features for a single match prediction (inference mode).

    .. deprecated::
        This function is a thin compatibility wrapper.
        New code should use ``FeatureStore.build_match_features()`` directly,
        which is the single source of truth for inference feature generation
        (Task 2 — Feature Store Centralization).

    Returns:
        pd.DataFrame: Features ready for model.predict() (single-row DataFrame).
    """
    import warnings
    warnings.warn(
        "prepare_features_for_prediction() is deprecated. "
        "Use FeatureStore.build_match_features(home_id, away_id, df_history) instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    # Delegate to the canonical implementation in FeatureStore.
    # Importing here to avoid a circular import at module level
    # (features_v2 is imported by feature_store, so we import feature_store lazily).
    from src.features.feature_store import FeatureStore
    df = db_manager.get_historical_data()
    return FeatureStore.build_match_features(home_id, away_id, df, window_long=window_long)


def generate_professional_feedback(ml_prediction, confidence, features_df, home_name, away_name):
    """
    Gera um feedback profissional e tático em português para a interface.
    Versão aprimorada com análises mais ricas e formatação premium.
    """
    lines = []
    
    # ═══════════════════════════════════════════════════════════════
    # SEÇÃO 1: PREVISÃO PRINCIPAL
    # ═══════════════════════════════════════════════════════════════
    conf_level = "Alta" if confidence >= 0.75 else "Média" if confidence >= 0.55 else "Baixa"
    lines.append(f"🎯 **PREVISÃO:** {ml_prediction:.1f} escanteios totais")
    lines.append(f"📊 **CONFIANÇA:** {confidence*100:.0f}% ({conf_level})")
    lines.append("")
    
    # ═══════════════════════════════════════════════════════════════
    # SEÇÃO 2: CONTEXTO TÁTICO
    # ═══════════════════════════════════════════════════════════════
    tactical = []
    
    try:
        # Médias recentes de escanteios
        h_avg = features_df['home_avg_corners_home'].iloc[0] if 'home_avg_corners_home' in features_df else 0
        a_avg = features_df['away_avg_corners_away'].iloc[0] if 'away_avg_corners_away' in features_df else 0
        tactical.append(f"• {home_name} (casa): {h_avg:.1f} escanteios/jogo")
        tactical.append(f"• {away_name} (fora): {a_avg:.1f} escanteios/jogo")
        
        # Cruzamentos (indicador de pressão lateral)
        h_crosses = features_df.get('home_avg_crosses_5g', features_df.get('home_avg_crosses_home', pd.Series([0]))).iloc[0]
        a_crosses = features_df.get('away_avg_crosses_5g', features_df.get('away_avg_crosses_away', pd.Series([0]))).iloc[0]
        if h_crosses > 12 or a_crosses > 12:
            tactical.append(f"• Volume de cruzamentos acima da média ({max(h_crosses, a_crosses):.0f}/jogo)")
        
        # Ataques Perigosos
        h_da = features_df.get('home_avg_dangerous_attacks_5g', features_df.get('home_avg_dangerous_attacks', pd.Series([0]))).iloc[0]
        a_da = features_df.get('away_avg_dangerous_attacks_5g', features_df.get('away_avg_dangerous_attacks', pd.Series([0]))).iloc[0]
        if h_da > 45 or a_da > 45:
            tactical.append(f"• Alta pressão ofensiva detectada ({max(h_da, a_da):.0f} ataques perigosos)")
            
        # Chutes Bloqueados (convertem em escanteios)
        h_blocked = features_df.get('home_avg_blocked_shots_5g', features_df.get('home_avg_blocked_shots', pd.Series([0]))).iloc[0]
        a_blocked = features_df.get('away_avg_blocked_shots_5g', features_df.get('away_avg_blocked_shots', pd.Series([0]))).iloc[0]
        if h_blocked > 3 or a_blocked > 3:
            tactical.append(f"• Defesa bloqueadora ativa (rebotes → escanteios)")
            
    except Exception:
        pass
    
    if tactical:
        lines.append("📈 **CONTEXTO TÁTICO:**")
        lines.extend(tactical)
        lines.append("")
    
    # ═══════════════════════════════════════════════════════════════
    # SEÇÃO 3: ODD JUSTA (baseada na confiança do modelo)
    # ═══════════════════════════════════════════════════════════════
    if confidence > 0:
        fair_odd = 1.0 / confidence
        lines.append("💰 **ODD JUSTA CALCULADA:**")
        lines.append(f"• Probabilidade: {confidence*100:.0f}%")
        lines.append(f"• Odd correspondente: {fair_odd:.2f}")
        if confidence >= 0.65:
            lines.append("• ✅ Alta convicção do modelo")
        elif confidence >= 0.50:
            lines.append("• ⚠️ Convicção moderada")
        else:
            lines.append("• ⚡ Baixa convicção - cautela recomendada")
    
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Joint Targets -- refactor/multimercado-cientifico (Fase 2)
# ---------------------------------------------------------------------------

def create_joint_targets(df):
    """
    Gera os 4 targets do vetor latente [home_1H, away_1H, home_2H, away_2H].

    Complementar a create_advanced_features() -- usado pelo JointTrainer.
    Nao modifica os targets existentes (backward compatibility).

    Requisitos:
        - corners_home_ft, corners_away_ft  (sempre presentes)
        - corners_home_ht, corners_away_ht  (HT -- podem estar ausentes)

    Nota sobre 2H:
        2H de treino = corners_ft - corners_ht sobre dados REAIS observados.
        Derivar previsoes de 2H subtraindo previsoes (FT_pred - 1H_pred)
        continua PROIBIDO -- esta funcao gera targets, nao predicoes.

    Returns:
        pd.DataFrame com colunas [home_1H, away_1H, home_2H, away_2H]
        preservando o indice de df, ou None se dados insuficientes.
    """
    required_ft = ["corners_home_ft", "corners_away_ft"]
    required_ht = ["corners_home_ht", "corners_away_ht"]

    for col in required_ft:
        if col not in df.columns:
            return None

    has_ht = all(c in df.columns for c in required_ht)
    if not has_ht:
        return None

    mask = (
        df["corners_home_ht"].notna()
        & df["corners_away_ht"].notna()
        & df["corners_home_ft"].notna()
        & df["corners_away_ft"].notna()
        & (df["corners_home_ht"] >= 0)
        & (df["corners_away_ht"] >= 0)
        & (df["corners_home_ft"] >= df["corners_home_ht"])
        & (df["corners_away_ft"] >= df["corners_away_ht"])
    )

    if mask.sum() < 50:
        return None

    import pandas as pd
    sub = df.loc[mask].copy()
    result = pd.DataFrame(index=sub.index)
    result["home_1H"] = sub["corners_home_ht"].astype(float)
    result["away_1H"] = sub["corners_away_ht"].astype(float)
    result["home_2H"] = (sub["corners_home_ft"] - sub["corners_home_ht"]).clip(lower=0).astype(float)
    result["away_2H"] = (sub["corners_away_ft"] - sub["corners_away_ht"]).clip(lower=0).astype(float)
    return result
