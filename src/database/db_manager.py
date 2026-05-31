from __future__ import annotations
"""
Módulo de Gerenciamento de Banco de Dados.

Padrão Facade: DBManager delega operações especializadas para repositórios.
Os repositórios compartilham a conexão via referência ao DBManager.
"""

import sqlite3
import pandas as pd
from datetime import datetime
import json
from typing import Optional, Dict, Any, List
from werkzeug.security import generate_password_hash, check_password_hash

from src.database.match_repository import MatchRepository
from src.database.prediction_repository import PredictionRepository
from src.database.user_repository import UserRepository

class DBManager:
    """
    Gerenciador de banco de dados SQLite.

    Facade que inicializa repositórios especializados:
        - self.matches → MatchRepository (CRUD de partidas e estatísticas)
        - self.predictions → PredictionRepository (CRUD de previsões e feedback loop)
        - self.users → UserRepository (CRUD de usuários e apostas)

    Todos os métodos públicos originais são mantidos para backward compatibility
    e delegam internamente para os repositórios correspondentes.
    """
    
    def __init__(self, db_path: str = "data/football_data.db"):
        self.db_path = db_path
        self.conn = None
        self.create_tables()

        # Repositórios especializados (Facade pattern)
        self.matches = MatchRepository(self)
        self.predictions = PredictionRepository(self)
        self.users = UserRepository(self)


    def connect(self) -> sqlite3.Connection:
        if self.conn is None:
            self.conn = sqlite3.connect(self.db_path, timeout=30.0, check_same_thread=False)
        return self.conn

    def close(self) -> None:
        if self.conn:
            self.conn.close()
            self.conn = None

    def create_tables(self) -> None:
        """
        Cria as tabelas necessárias no banco de dados se não existirem.
        
        Regra de Negócio:
            Garante a estrutura do banco para armazenar partidas, estatísticas e predições.
            Executa migrações automáticas para manter compatibilidade com versões anteriores.
        """
        conn = self.connect()
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS matches (
                match_id INTEGER PRIMARY KEY,
                tournament_name TEXT,
                tournament_id INTEGER,
                season_id INTEGER,
                round INTEGER,
                status TEXT,
                start_timestamp INTEGER,
                home_team_id INTEGER,
                home_team_name TEXT,
                away_team_id INTEGER,
                away_team_name TEXT,
                home_score INTEGER,
                away_score INTEGER,
                odds_home REAL,
                odds_draw REAL,
                odds_away REAL,
                match_minute TEXT,
                home_league_position INTEGER,
                away_league_position INTEGER,
                last_updated TEXT
            )
        ''')
        
        # Migração: Adiciona colunas de odds se não existirem
        try:
             cursor.execute("ALTER TABLE matches ADD COLUMN odds_home REAL")
             cursor.execute("ALTER TABLE matches ADD COLUMN odds_draw REAL")
             cursor.execute("ALTER TABLE matches ADD COLUMN odds_away REAL")
             print("✅ Migração de schema: Colunas de odds adicionadas.")
        except sqlite3.OperationalError:
             pass # Colunas já existem
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS match_stats (
                match_id INTEGER PRIMARY KEY,
                corners_home_ft INTEGER,
                corners_away_ft INTEGER,
                corners_home_ht INTEGER,
                corners_away_ht INTEGER,
                shots_ot_home_ft INTEGER,
                shots_ot_away_ft INTEGER,
                shots_ot_home_ht INTEGER,
                shots_ot_away_ht INTEGER,
                possession_home INTEGER,
                possession_away INTEGER,
                total_shots_home INTEGER,
                total_shots_away INTEGER,
                fouls_home INTEGER,
                fouls_away INTEGER,
                yellow_cards_home INTEGER,
                yellow_cards_away INTEGER,
                red_cards_home INTEGER,
                red_cards_away INTEGER,
                big_chances_home INTEGER,
                big_chances_away INTEGER,
                dangerous_attacks_home INTEGER,
                dangerous_attacks_away INTEGER,
                expected_goals_home REAL,
                expected_goals_away REAL,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (match_id) REFERENCES matches (match_id)
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                match_id INTEGER,
                model_version TEXT,
                prediction_value REAL,
                prediction_label TEXT,
                confidence REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                is_correct BOOLEAN,
                category TEXT,
                market_group TEXT,
                odds REAL,
                status TEXT DEFAULT 'PENDING',
                FOREIGN KEY (match_id) REFERENCES matches (match_id)
            )
        ''')

        # --- NEW TABLES FOR BET REGISTRY ---
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS bets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                timestamp INTEGER,
                status TEXT DEFAULT 'PENDING', -- PENDING, GREEN, RED, VOID
                stake REAL,
                total_odds REAL,
                possible_win REAL,
                bet_type TEXT, -- SINGLE, MULTIPLE
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS bet_items (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                bet_id INTEGER,
                match_id INTEGER,
                prediction_label TEXT,
                odds REAL,
                status TEXT DEFAULT 'PENDING',
                FOREIGN KEY (bet_id) REFERENCES bets (id),
                FOREIGN KEY (match_id) REFERENCES matches (match_id)
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                role TEXT DEFAULT 'user',
                initial_bankroll REAL DEFAULT 0.0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # --- BANKROLL HISTORY WITH USER_ID ---
        # Note: bankroll_history might exist from legacy. We will migrate it.
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS bankroll_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                balance REAL,
                transaction_type TEXT, -- BET_PLACED, WIN, REFUND, DEPOSIT, WITHDRAW
                amount REAL,
                bet_id INTEGER,
                description TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')

        # Migrations
        try:
            cursor.execute("SELECT tournament_id FROM matches LIMIT 1")
        except sqlite3.OperationalError:
            cursor.execute("ALTER TABLE matches ADD COLUMN tournament_id INTEGER")
            conn.commit()

        # Multi-User Migrations
        try:
            cursor.execute("SELECT user_id FROM bets LIMIT 1")
        except sqlite3.OperationalError:
            cursor.execute("ALTER TABLE bets ADD COLUMN user_id INTEGER")
            print("✅ Migração: user_id adicionado em bets.")
            conn.commit()

        try:
            cursor.execute("SELECT user_id FROM bankroll_history LIMIT 1")
        except sqlite3.OperationalError:
            cursor.execute("ALTER TABLE bankroll_history ADD COLUMN user_id INTEGER")
            print("✅ Migração: user_id adicionado em bankroll_history.")
            conn.commit()
            
        # Ensure bankroll_history table exists (if not created by CREATE stmt above due to legacy table name clash?)
        # Legacy might have been created without user_id. The ALTER above handles it.

        # Robust Migration using PRAGMA to avoid exception-driven logic
        cursor.execute("PRAGMA table_info(predictions)")
        columns_info = cursor.fetchall()
        columns = [info[1] for info in columns_info]

        if 'predicted_value' in columns:
            cursor.execute("ALTER TABLE predictions RENAME COLUMN predicted_value TO prediction_value")
            conn.commit()
            print("✅ Migração: predicted_value renomeado para prediction_value.")

        if 'prediction_type' in columns:
            cursor.execute("ALTER TABLE predictions RENAME COLUMN prediction_type TO model_version")
            conn.commit()
            print("✅ Migração: prediction_type renomeado para model_version.")
        
        if 'market' in columns:
            cursor.execute("ALTER TABLE predictions RENAME COLUMN market TO prediction_label")
            conn.commit()
            print("✅ Migração: market renomeado para prediction_label.")

        if 'probability' in columns:
            cursor.execute("ALTER TABLE predictions RENAME COLUMN probability TO confidence")
            conn.commit()
            print("✅ Migração: probability renomeado para confidence.")

        # Check for new columns to add
        if 'is_correct' not in columns:
             cursor.execute("ALTER TABLE predictions ADD COLUMN is_correct BOOLEAN")
             conn.commit()
        
        if 'status' not in columns:
             cursor.execute("ALTER TABLE predictions ADD COLUMN status TEXT DEFAULT 'PENDING'")
             conn.commit()

        try:
            cursor.execute("SELECT feedback_text FROM predictions LIMIT 1")
        except sqlite3.OperationalError:
            cursor.execute("ALTER TABLE predictions ADD COLUMN feedback_text TEXT")
            conn.commit()

        try:
            cursor.execute("SELECT fair_odds FROM predictions LIMIT 1")
        except sqlite3.OperationalError:
            cursor.execute("ALTER TABLE predictions ADD COLUMN fair_odds REAL")
            conn.commit()

        # ✨ MIGRATION: raw_model_score (Lambda puro antes de arredondamento)
        try:
            cursor.execute("SELECT raw_model_score FROM predictions LIMIT 1")
        except sqlite3.OperationalError:
            cursor.execute("ALTER TABLE predictions ADD COLUMN raw_model_score REAL")
            print("✅ Migração: Coluna raw_model_score adicionada (Bug Fix #1).")
            conn.commit()

        try:
            cursor.execute("SELECT home_league_position FROM matches LIMIT 1")
        except sqlite3.OperationalError:
            cursor.execute("ALTER TABLE matches ADD COLUMN home_league_position INTEGER")
            cursor.execute("ALTER TABLE matches ADD COLUMN away_league_position INTEGER")
            conn.commit()

        # Dangerous Attacks Migration
        try:
            cursor.execute("SELECT dangerous_attacks_home FROM match_stats LIMIT 1")
        except sqlite3.OperationalError:
            cursor.execute("ALTER TABLE match_stats ADD COLUMN dangerous_attacks_home INTEGER DEFAULT 0")
            cursor.execute("ALTER TABLE match_stats ADD COLUMN dangerous_attacks_away INTEGER DEFAULT 0")
            print("[OK] Migracao de schema: Colunas de Dangerous Attacks adicionadas.")
            conn.commit()

        # Odds Migration
        try:
            cursor.execute("SELECT odds_home FROM matches LIMIT 1")
        except sqlite3.OperationalError:
            cursor.execute("ALTER TABLE matches ADD COLUMN odds_home REAL")
            cursor.execute("ALTER TABLE matches ADD COLUMN odds_draw REAL")
            cursor.execute("ALTER TABLE matches ADD COLUMN odds_away REAL")
            cursor.execute("ALTER TABLE matches ADD COLUMN odds_provider TEXT")
            conn.commit()

        # xG Migration
        try:
            cursor.execute("SELECT expected_goals_home FROM match_stats LIMIT 1")
        except sqlite3.OperationalError:
            cursor.execute("ALTER TABLE match_stats ADD COLUMN expected_goals_home REAL")
            cursor.execute("ALTER TABLE match_stats ADD COLUMN expected_goals_away REAL")
            print("✅ Migração: Colunas xG adicionadas.")
            conn.commit()

        # Match Minute Migration (Live Data)
        try:
            cursor.execute("SELECT match_minute FROM matches LIMIT 1")
        except sqlite3.OperationalError:
            cursor.execute("ALTER TABLE matches ADD COLUMN match_minute TEXT")
            print("✅ Migração: Coluna match_minute adicionada.")
            conn.commit()

        # Tactical Metrics Migration (Gap Analysis)
        tactical_cols = [
            'blocked_shots', 'crosses', 'tackles', 'interceptions', 'clearances', 'recoveries'
        ]
        try:
            cursor.execute("SELECT blocked_shots_home FROM match_stats LIMIT 1")
        except sqlite3.OperationalError:
            for col in tactical_cols:
                cursor.execute(f"ALTER TABLE match_stats ADD COLUMN {col}_home INTEGER DEFAULT 0")
                cursor.execute(f"ALTER TABLE match_stats ADD COLUMN {col}_away INTEGER DEFAULT 0")
            print("✅ Migração: Métricas Táticas (Blocked Shots, Crosses, etc) adicionadas.")
            conn.commit()

        # Momentum Metrics Migration (Attack Momentum)
        # Regra de Negócio: Armazena a pressão acumulada (Area Under Curve) extraída do gráfico de momentum.
        try:
            cursor.execute("SELECT momentum_home FROM match_stats LIMIT 1")
        except sqlite3.OperationalError:
            cursor.execute("ALTER TABLE match_stats ADD COLUMN momentum_home REAL DEFAULT 0.0")
            cursor.execute("ALTER TABLE match_stats ADD COLUMN momentum_away REAL DEFAULT 0.0")
            cursor.execute("ALTER TABLE match_stats ADD COLUMN momentum_peak_home INTEGER DEFAULT 0")
            cursor.execute("ALTER TABLE match_stats ADD COLUMN momentum_peak_away INTEGER DEFAULT 0")
            print("✅ Migração: Métricas de Attack Momentum adicionadas.")
            conn.commit()

        # Migração bet_items: garante que a coluna bet_id exista
        try:
            cursor.execute("SELECT bet_id FROM bet_items LIMIT 1")
        except sqlite3.OperationalError:
            # Tabela antiga sem bet_id - recria com schema correto
            cursor.execute("DROP TABLE IF EXISTS bet_items")
            cursor.execute('''
                CREATE TABLE bet_items (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    bet_id INTEGER,
                    match_id INTEGER,
                    prediction_label TEXT,
                    odds REAL,
                    status TEXT DEFAULT 'PENDING',
                    FOREIGN KEY (bet_id) REFERENCES bets (id),
                    FOREIGN KEY (match_id) REFERENCES matches (match_id)
                )
            ''')
            print("✅ Migração: Tabela bet_items recriada com schema correto.")
            conn.commit()

        conn.commit()


    def save_match(self, match_data: dict) -> None:
        """
        Salva ou atualiza os dados básicos de uma partida.
        
        Args:
            match_data (dict): Dicionário com dados da partida (id, times, placar, etc).
            
        Regra de Negócio:
            Centraliza a persistência de dados brutos das partidas para histórico e feature engineering.
        """
        conn = self.connect()
        cursor = conn.cursor()

        # --- AUTO-MIGRATE IDS (FIREWALL) ---
        # Garante que IDs legados do SofaScore sejam convertidos para o ID Unificado do nosso banco.
        unified_ids = {
            1: 17,   # Premier League
            42: 35,  # Bundesliga
            36: 8,   # LaLiga
            33: 23,  # Serie A
            4: 34    # Ligue 1
        }
        
        original_id = match_data.get('tournament_id')
        if original_id in unified_ids:
            # print(f"🔄 Auto-Corrigindo Liga: ID {original_id} -> {unified_ids[original_id]}")
            match_data['tournament_id'] = unified_ids[original_id]
        # -----------------------------------

        # 🛡️ DEFENSIVE LOGIC: Prevent overwriting live minute with None
        # If scraper fails to get minute but we already have it, keep it.
        if match_data.get('status') == 'inprogress' and match_data.get('match_minute') is None:
            try:
                exist = cursor.execute("SELECT match_minute FROM matches WHERE match_id=?", (match_data['id'],)).fetchone()
                if exist and exist[0]:
                    match_data['match_minute'] = exist[0]
            except:
                pass

        try:
            # Usamos INSERT ... ON CONFLICT para não apagar colunas que não estamos enviando (ex: league_positions)
            # O REPLACE antigo deletava a linha e inseria de novo, perdendo dados manuais.
            cursor.execute('''
                INSERT INTO matches (
                    match_id, tournament_name, tournament_id, season_id, round, status, 
                    start_timestamp, home_team_id, home_team_name, 
                    away_team_id, away_team_name, home_score, away_score, match_minute,
                    home_league_position, away_league_position, last_updated
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(match_id) DO UPDATE SET
                    status=excluded.status,
                    home_score=excluded.home_score,
                    away_score=excluded.away_score,
                    match_minute=excluded.match_minute,
                    round=excluded.round,
                    start_timestamp=excluded.start_timestamp,
                    home_league_position=COALESCE(excluded.home_league_position, matches.home_league_position),
                    away_league_position=COALESCE(excluded.away_league_position, matches.away_league_position),
                    last_updated=CURRENT_TIMESTAMP
            ''', (
                match_data['id'], match_data['tournament'], match_data.get('tournament_id'),
                match_data['season_id'], match_data.get('round'), match_data['status'], 
                match_data['timestamp'], match_data['home_id'], match_data['home_name'],
                match_data['away_id'], match_data['away_name'],
                match_data['home_score'], match_data['away_score'],
                match_data.get('match_minute'),
                match_data.get('home_position'), 
                match_data.get('away_position')
            ))
            conn.commit()
        except Exception as e:
            print(f"Erro ao salvar jogo {match_data.get('id')}: {e}")

    def save_stats(self, match_id: int, stats_data: dict) -> None:
        """
        Salva as estatísticas detalhadas de uma partida.
        
        Args:
            match_id (int): ID da partida.
            stats_data (dict): Dicionário com estatísticas (escanteios, chutes, etc).
            
        Regra de Negócio:
            Armazena métricas profundas usadas para calcular médias e tendências dos times.
        """
        conn = self.connect()
        cursor = conn.cursor()
        try:
            cursor.execute('''
                INSERT OR REPLACE INTO match_stats (
                    match_id, corners_home_ft, corners_away_ft, corners_home_ht, corners_away_ht,
                    shots_ot_home_ft, shots_ot_away_ft, shots_ot_home_ht, shots_ot_away_ht,
                    possession_home, possession_away, total_shots_home, total_shots_away,
                    fouls_home, fouls_away, yellow_cards_home, yellow_cards_away,
                    red_cards_home, red_cards_away, big_chances_home, big_chances_away,
                    dangerous_attacks_home, dangerous_attacks_away,
                    expected_goals_home, expected_goals_away,
                    blocked_shots_home, blocked_shots_away,
                    crosses_home, crosses_away,
                    tackles_home, tackles_away,
                    interceptions_home, interceptions_away,
                    clearances_home, clearances_away,
                    recoveries_home, recoveries_away,
                    momentum_home, momentum_away,
                    momentum_peak_home, momentum_peak_away
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                match_id,
                stats_data.get('corners_home_ft', 0), stats_data.get('corners_away_ft', 0),
                stats_data.get('corners_home_ht', 0), stats_data.get('corners_away_ht', 0),
                stats_data.get('shots_ot_home_ft', 0), stats_data.get('shots_ot_away_ft', 0),
                stats_data.get('shots_ot_home_ht', 0), stats_data.get('shots_ot_away_ht', 0),
                stats_data.get('possession_home', 0), stats_data.get('possession_away', 0),
                stats_data.get('total_shots_home', 0), stats_data.get('total_shots_away', 0),
                stats_data.get('fouls_home', 0), stats_data.get('fouls_away', 0),
                stats_data.get('yellow_cards_home', 0), stats_data.get('yellow_cards_away', 0),
                stats_data.get('red_cards_home', 0), stats_data.get('red_cards_away', 0),
                stats_data.get('big_chances_home', 0), stats_data.get('big_chances_away', 0),
                stats_data.get('dangerous_attacks_home', 0), stats_data.get('dangerous_attacks_away', 0),
                stats_data.get('expected_goals_home', 0.0), stats_data.get('expected_goals_away', 0.0),
                stats_data.get('blocked_shots_home', 0), stats_data.get('blocked_shots_away', 0),
                stats_data.get('crosses_home', 0), stats_data.get('crosses_away', 0),
                stats_data.get('tackles_home', 0), stats_data.get('tackles_away', 0),
                stats_data.get('interceptions_home', 0), stats_data.get('interceptions_away', 0),
                stats_data.get('clearances_home', 0), stats_data.get('clearances_away', 0),
                stats_data.get('recoveries_home', 0), stats_data.get('recoveries_away', 0),
                stats_data.get('momentum_home', 0.0), stats_data.get('momentum_away', 0.0),
                stats_data.get('momentum_peak_home', 0), stats_data.get('momentum_peak_away', 0)
            ))
            conn.commit()
            # print(f"DEBUG: Stats saved for match {match_id}") # Uncomment for deeper debug if needed
        except Exception as e:
            print(f"Erro ao salvar stats do jogo {match_id}: {e}")

    def get_historical_data(self) -> pd.DataFrame:
        """
        Recupera todo o histórico de partidas finalizadas com estatísticas.
        
        Returns:
            pd.DataFrame: DataFrame contendo dados de partidas e estatísticas unificadas.
            
        Regra de Negócio:
            Fornece a base de dados completa para o treinamento do modelo de Machine Learning.
        """
        conn = self.connect()
        query = '''
            SELECT m.*, s.corners_home_ft, s.corners_away_ft, s.corners_home_ht, s.corners_away_ht,
                   s.shots_ot_home_ft, s.shots_ot_away_ft, s.shots_ot_home_ht, s.shots_ot_away_ht,
                   s.big_chances_home, s.big_chances_away,
                   s.dangerous_attacks_home, s.dangerous_attacks_away,
                   s.blocked_shots_home, s.blocked_shots_away,
                   s.crosses_home, s.crosses_away,
                   s.tackles_home, s.tackles_away,
                   s.interceptions_home, s.interceptions_away,
                   s.clearances_home, s.clearances_away,
                   s.recoveries_home, s.recoveries_away,
                   COALESCE(s.expected_goals_home, 0.0) as expected_goals_home,
                   COALESCE(s.expected_goals_away, 0.0) as expected_goals_away,
                   COALESCE(s.possession_home, 0) as possession_home,
                   COALESCE(s.possession_away, 0) as possession_away
            FROM matches m
            JOIN match_stats s ON m.match_id = s.match_id
            WHERE m.status = 'finished'
            ORDER BY m.start_timestamp ASC
        '''
        return pd.read_sql_query(query, conn)

    def get_season_stats(self, season_id: int) -> dict:
        """
        Retorna estatísticas resumidas de uma temporada.
        
        Args:
            season_id (int): ID da temporada.
            
        Returns:
            dict: {'total_matches': int, 'last_round': int}
            
        Regra de Negócio:
            Permite controle incremental de atualizações, evitando re-processar temporadas completas.
        """
        conn = self.connect()
        cursor = conn.cursor()
        cursor.execute('''
            SELECT COUNT(*), MAX(round) 
            FROM matches 
            WHERE season_id = ? AND status = 'finished'
        ''', (season_id,))
        row = cursor.fetchone()
        return {
            'total_matches': row[0] if row else 0,
            'last_round': row[1] if row and row[1] else 0
        }

    def save_prediction(self, match_id: int, model_version: str, value: float, label: str, confidence: float, category: str = None, market_group: str = None, odds: float = 0.0, feedback_text: str = None, fair_odds: float = 0.0, raw_model_score: float = None, verbose: bool = False) -> None:
        """
        Salva uma predição gerada pelo modelo ou análise estatística.
        
        Args:
            match_id (int): ID da partida.
            model_version (str): Identificador do modelo (ex: 'Professional V2', 'Statistical').
            value (float): Valor numérico da predição (ex: 9.5 escanteios).
            label (str): Rótulo legível (ex: 'Over 9.5').
            confidence (float): Grau de confiança ou probabilidade (0.0 a 1.0).
            category (str, optional): Categoria de risco (ex: 'Top7', 'Suggestion_Easy').
            market_group (str, optional): Grupo de mercado (ex: 'Escanteios Totais').
            odds (float, optional): Odd no momento da análise.
            feedback_text (str, optional): Feedback textual da análise.
            fair_odds (float, optional): Odd justa calculada pela IA.
            raw_model_score (float, optional): Lambda puro do modelo (ex: 12.5) antes de arredondar para linha de mercado.
            verbose (bool): Se deve imprimir confirmação no console.
            
        Regra de Negócio:
            Registra as previsões para posterior validação (backtesting) e exibição ao usuário.
            Evita duplicatas verificando se já existe previsão igual.
        """
        conn = self.connect()
        cursor = conn.cursor()
        try:
            # Verifica se já existe previsão idêntica (Bug Fix: mais rígor na verificação)
            # Consideramos duplicata se: mesmo match_id, mesmo label E categoria
            cursor.execute('''
                SELECT id FROM predictions 
                WHERE match_id = ? AND prediction_label = ? AND category = ?
            ''', (match_id, label, category))
            
            existing = cursor.fetchone()
            if existing:
                # Atualiza em vez de duplicar
                cursor.execute('''
                    UPDATE predictions 
                    SET prediction_value = ?, confidence = ?, odds = ?, model_version = ?, 
                        feedback_text = ?, fair_odds = ?, raw_model_score = ?, status = 'PENDING'
                    WHERE id = ?
                ''', (value, confidence, odds, model_version, feedback_text, fair_odds, raw_model_score, existing[0]))
            else:
                # Insere nova previsão
                cursor.execute('''
                    INSERT INTO predictions (
                        match_id, model_version, prediction_value, prediction_label, 
                        confidence, category, market_group, odds, feedback_text, fair_odds, raw_model_score, status
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'PENDING')
                ''', (match_id, model_version, value, label, confidence, category, market_group, odds, feedback_text, fair_odds, raw_model_score))
            
            conn.commit()
            if verbose:
                print(f"✅ Predição salva/atualizada: {label} ({category})")
        except Exception as e:
            print(f"Erro ao salvar predição: {e}")

    def delete_predictions(self, match_id: int) -> None:
        """
        Remove predições existentes para uma partida.
        
        Args:
            match_id (int): ID da partida.
            
        Regra de Negócio:
            Garante que ao re-analisar um jogo, não fiquem predições duplicadas ou obsoletas.
        """
        conn = self.connect()
        cursor = conn.cursor()
        try:
            cursor.execute("DELETE FROM predictions WHERE match_id = ?", (match_id,))
            conn.commit()
        except Exception as e:
            print(f"Erro ao limpar predições antigas: {e}")

    def check_predictions(self) -> None:
        """
        Verifica se predições passadas acertaram (Feedback Loop).
        
        Regra de Negócio:
            - Total Mandante → usa corners_home_ft
            - Total Visitante → usa corners_away_ft  
            - Jogo Completo / outros → usa soma total
            - 1º Tempo → usa corners_*_ht
            - 2º Tempo → usa corners_*_ft - corners_*_ht
        """
        conn = self.connect()
        cursor = conn.cursor()
        
        # Busca predictions com market_group para determinar qual valor usar
        # NOTA: Não filtra por status para permitir re-verificação de predições com bug anterior
        query = '''
            SELECT p.id, p.match_id, p.prediction_value, p.prediction_label, p.market_group,
                   s.corners_home_ft, s.corners_away_ft, s.corners_home_ht, s.corners_away_ht
            FROM predictions p
            JOIN matches m ON p.match_id = m.match_id
            LEFT JOIN match_stats s ON m.match_id = s.match_id
            WHERE m.status = 'finished'
              AND s.corners_home_ft IS NOT NULL
              AND p.prediction_label != 'Tactical Analysis Data'
        '''
        
        cursor.execute(query)
        rows = cursor.fetchall()
        
        if not rows:
            return

        print(f"Verificando {len(rows)} predições pendentes...")
        
        for row in rows:
            pred_id, match_id, pred_val, pred_label, market_group, h_corners_ft, a_corners_ft, h_corners_ht, a_corners_ht = row
            
            # Garante valores numéricos (fallback para 0 se None)
            h_corners_ft = h_corners_ft or 0
            a_corners_ft = a_corners_ft or 0
            h_corners_ht = h_corners_ht or 0
            a_corners_ht = a_corners_ht or 0
            
            # Determina qual valor usar baseado no market_group E no prediction_label
            # market_group pode ser: "Vis.", "Casa", "1T", "2T", "Total", etc.
            # prediction_label pode ser: "Vis. Over 3.5", "Casa Under 5.5", "1T Over 2.5", etc.
            import re
            market_group_lower = (market_group or '').lower()
            label_lower = (pred_label or '').lower()
            combined = f"{market_group_lower} {label_lower}"
            
            if re.search(r'\b(mandante|casa|home)\b', combined):
                # Total Mandante: usa apenas escanteios do time da casa
                corners_value = h_corners_ft
            elif re.search(r'\b(visitante|away)\b', combined) or 'vis.' in combined or 'vis ' in combined:
                # Total Visitante: usa apenas escanteios do visitante
                corners_value = a_corners_ft
            elif re.search(r'\b(1t|1h|ht|primeiro)\b', combined):
                # 1º Tempo: usa soma dos escanteios do 1º tempo
                corners_value = h_corners_ht + a_corners_ht
            elif re.search(r'\b(2t|2h|segundo)\b', combined):
                # 2º Tempo: usa diferença (FT - HT)
                corners_value = (h_corners_ft - h_corners_ht) + (a_corners_ft - a_corners_ht)
            else:
                # Jogo Completo ou outros: usa soma total
                corners_value = h_corners_ft + a_corners_ft
            
            is_over = 'over' in pred_label.lower() if pred_label else False
            is_under = 'under' in pred_label.lower() if pred_label else False
            
            # 🐛 FIX: Extrai a LINHA do prediction_label (ex: "1T Under 5.5" → 5.5)
            # Busca número APÓS a palavra Over/Under para evitar pegar "1" de "1T"
            line = None
            if pred_label:
                # Primeiro tenta pegar número após Over/Under
                match = re.search(r'(?:over|under)\s*(\d+\.?\d*)', pred_label.lower())
                if match:
                    line = float(match.group(1))
                else:
                    # Fallback: pega o último número no label
                    matches = re.findall(r'(\d+\.?\d*)', pred_label)
                    if matches:
                        line = float(matches[-1])
            
            is_correct = False
            if line is not None:
                if is_over:
                    is_correct = corners_value > line
                elif is_under:
                    is_correct = corners_value < line
            
            status = 'GREEN' if is_correct else 'RED'
            cursor.execute("UPDATE predictions SET is_correct = ?, status = ? WHERE id = ?", (is_correct, status, pred_id))
            
        conn.commit()
        print("Verificacao de predicoes concluida.")
        
        # Chama verificação de apostas após atualizar predições
        self.check_bets_debug()
    
    def get_win_rate_stats(self) -> dict:
        """
        Calcula estatísticas de Win Rate das predições.
        
        Returns:
            dict com: total, correct, win_rate, win_rate_top7, pending
        """
        conn = self.connect()
        cursor = conn.cursor()
        
        # Total de predições finalizadas (is_correct definido ou status final)
        cursor.execute("""
            SELECT COUNT(*) FROM predictions 
            WHERE is_correct IS NOT NULL OR status IN ('GREEN', 'RED')
        """)
        total = cursor.fetchone()[0] or 0
        
        # Predições corretas (is_correct=1 ou status='GREEN')
        cursor.execute("""
            SELECT COUNT(*) FROM predictions 
            WHERE is_correct = 1 OR status = 'GREEN'
        """)
        correct = cursor.fetchone()[0] or 0
        
        # Win Rate das Top 7 (confiança > 0.75 - NOTE: table stores 0.XX mostly, check if it stores 75)
        # In quick_scan it prints 82%. In match_card it shows (conf*100).
        # DB usually stores 0.82.
        # But existing code checked `confidence > 75`. This might be a BUG if confidence is stored as 0.82.
        # Let's check the schema or data. In match_card we saw `row[4]` (confidence).
        # In visual feedback we did `conf > 0.75`.
        # I suspect the DB has 0.82 but this query checks > 75.
        # I will change it to `confidence > 0.75 OR confidence > 75` to be safe.
        
        cursor.execute("""
            SELECT COUNT(*) FROM predictions 
            WHERE (is_correct IS NOT NULL OR status IN ('GREEN', 'RED')) 
            AND (confidence > 0.75 OR confidence > 75)
        """)
        total_top7 = cursor.fetchone()[0] or 0
        
        cursor.execute("""
            SELECT COUNT(*) FROM predictions 
            WHERE (is_correct = 1 OR status = 'GREEN') 
            AND (confidence > 0.75 OR confidence > 75)
        """)
        correct_top7 = cursor.fetchone()[0] or 0
        
        # Predições pendentes
        cursor.execute("""
            SELECT COUNT(*) FROM predictions 
            WHERE status = 'PENDING' OR status IS NULL
        """)
        pending = cursor.fetchone()[0] or 0
        
        # Calcula win rates
        win_rate = correct / total if total > 0 else 0.0
        win_rate_top7 = correct_top7 / total_top7 if total_top7 > 0 else 0.0
        
        return {
            'total': total,
            'correct': correct,
            'win_rate': win_rate,
            'total_top7': total_top7,
            'correct_top7': correct_top7,
            'win_rate_top7': win_rate_top7,
            'pending': pending
        }
    
    def check_bets_debug(self) -> None:
        print("DEBUG: Entered check_bets function")
        """
        Verifica o status das apostas registradas baseado no status dos seus itens.
        Uma aposta é GREEN se todos os itens forem GREEN.
        Uma aposta é RED se algum item for RED.
        """
        conn = self.connect()
        cursor = conn.cursor()
        
        # 1. Atualiza primeiro os status dos bet_items baseado nas predictions verificadas
        query_items = '''
            UPDATE bet_items
            SET status = (
                SELECT p.status 
                FROM predictions p 
                WHERE p.match_id = bet_items.match_id 
                  AND p.prediction_label = bet_items.prediction_label
                LIMIT 1
            )
            WHERE status = 'PENDING'
              AND EXISTS (
                SELECT 1 FROM predictions p 
                WHERE p.match_id = bet_items.match_id 
                  AND p.prediction_label = bet_items.prediction_label
                  AND p.status IN ('GREEN', 'RED')
              )
        '''
        cursor.execute(query_items)
        
        # 1.1 Robust Verification Fallback: Check items that are STILL pending
        # or didn't find a matching prediction record.
        cursor.execute('''
            SELECT bi.id, bi.match_id, bi.prediction_label, 
                   s.corners_home_ft, s.corners_away_ft, s.corners_home_ht, s.corners_away_ht
            FROM bet_items bi
            JOIN matches m ON bi.match_id = m.match_id
            JOIN match_stats s ON m.match_id = s.match_id
            WHERE bi.status = 'PENDING' AND m.status = 'finished'
        ''')
        pending_items = cursor.fetchall()
        print(f"DEBUG: check_bets pending_items count = {len(pending_items)}")
        
        for item in pending_items:
            item_id, m_id, label, h_ft, a_ft, h_ht, a_ht = item
            h_ft, a_ft, h_ht, a_ht = h_ft or 0, a_ft or 0, h_ht or 0, a_ht or 0
            
            # Improved extractor for "X Over/Under Y"
            try:
                label_lower = label.lower()
                
                import re
                # First try to find number explicitly after Over/Under/Mais/Menos
                match_val = re.search(r'(?:over|under|mais|menos)\s*(\d+\.?\d*)', label_lower)
                
                if match_val:
                    line = float(match_val.group(1))
                else:
                    # Fallback: take the last number found in the string
                    all_nums = re.findall(r'(\d+\.?\d*)', label)
                    if not all_nums: continue
                    line = float(all_nums[-1])
                
                # Define flags for Team and Period
                is_home = any(k in label_lower for k in ['casa', 'home', 'mandante'])
                is_away = any(k in label_lower for k in ['vis.', 'vis ', 'away', 'visitante'])
                is_1t = any(k in label_lower for k in ['1t', 'ht'])
                is_2t = any(k in label_lower for k in ['2t', '2st', ' st'])
                
                # Selection logic (Team x Period)
                if is_home:
                    val_to_check = h_ht if is_1t else (h_ft - h_ht if is_2t else h_ft)
                elif is_away:
                    val_to_check = a_ht if is_1t else (a_ft - a_ht if is_2t else a_ft)
                else:
                    # Default: Total Match
                    val_to_check = (h_ht + a_ht) if is_1t else ((h_ft - h_ht) + (a_ft - a_ht) if is_2t else h_ft + a_ft)
                
                is_correct = False
                if 'over' in label_lower: is_correct = val_to_check > line
                elif 'under' in label_lower: is_correct = val_to_check < line
                
                item_status = 'GREEN' if is_correct else 'RED'
                print(f"DEBUG: Label='{label}', Line={line}, Val={val_to_check}, Correct={is_correct}")
                cursor.execute("UPDATE bet_items SET status = ? WHERE id = ?", (item_status, item_id))
            except Exception as e:
                print(f"DEBUG ERROR: {e}")
                continue

        # 2. Atualiza status da aposta principal (bets)
        # Se algum item é RED -> Bet RED
        cursor.execute('''
            UPDATE bets
            SET status = 'RED'
            WHERE status = 'PENDING'
              AND id IN (SELECT bet_id FROM bet_items WHERE status = 'RED')
        ''')
        
        # Se todos os itens de uma aposta PENDING agora são GREEN -> Bet GREEN
        cursor.execute('''
            UPDATE bets
            SET status = 'GREEN'
            WHERE status = 'PENDING'
              AND NOT EXISTS (SELECT 1 FROM bet_items WHERE bet_id = bets.id AND status != 'GREEN')
              AND EXISTS (SELECT 1 FROM bet_items WHERE bet_id = bets.id)
        ''')
        
        conn.commit()
        print("✅ Verificação de apostas concluída.")

    def delete_bet(self, bet_id: int) -> bool:
        """
        Deleta uma aposta e seus itens relacionados.
        """
        conn = self.connect()
        cursor = conn.cursor()
        try:
            cursor.execute("DELETE FROM bet_items WHERE bet_id = ?", (bet_id,))
            cursor.execute("DELETE FROM bets WHERE id = ?", (bet_id,))
            conn.commit()
            print(f"✅ Aposta {bet_id} deletada com sucesso.")
            return True
        except Exception as e:
            print(f"❌ Erro ao deletar aposta {bet_id}: {e}")
            return False

    def reset_all_betting_history(self) -> bool:
        """
        Remove todo o histórico de apostas e itens, resetando a banca dos usuários para zero.
        
        Regra de Negócio:
            - Limpa a tabela bet_items (FK cascade não é garantido em todos os ambientes SQLite).
            - Limpa a tabela bets.
            - Reseta o initial_bankroll de todos os usuários cadastrados para 0.0, conforme pedido do usuário.
            
        Returns:
            bool: True se a operação foi concluída com sucesso.
        """
        conn = self.connect()
        cursor = conn.cursor()
        try:
            # 1. Limpa itens de aposta
            cursor.execute("DELETE FROM bet_items")
            # 2. Limpa apostas (boletins)
            cursor.execute("DELETE FROM bets")
            # 3. Reseta bankroll inicial de todos os usuários para 0.0
            cursor.execute("UPDATE users SET initial_bankroll = 0.0")
            
            conn.commit()
            print("✨ Histórico de apostas resetado com sucesso (Banca = R$ 0.00).")
            return True
        except Exception as e:
            print(f"❌ Erro ao resetar histórico de apostas: {e}")
            conn.rollback()
            return False

    def clear_finished_predictions(self) -> int:
        """
        Remove todas as predições com status 'GREEN' ou 'RED' do banco de dados.
        
        Regra de Negócio:
            Utilizado para limpeza de histórico durante fases de depuração ou recalibração,
            mantendo apenas os registros 'PENDING' que ainda precisam de acompanhamento.
            Isso ajuda a focar a análise em jogos ativos e reduzir o ruído visual em dashboards.
            
        Returns:
            int: Número de registros removidos.
        """
        conn = self.connect()
        cursor = conn.cursor()
        try:
            cursor.execute("DELETE FROM predictions WHERE status IN ('GREEN', 'RED')")
            count = cursor.rowcount
            conn.commit()
            print(f"🧹 Limpeza concluída: {count} predições removidas.")
            return count
        except Exception as e:
            print(f"❌ Erro ao limpar predições: {e}")
            return 0
    
    def fix_existing_predictions_values(self) -> int:
        """
        Corrige previsões antigas que foram salvas com prediction_value=0.
        Extrai o valor da linha do prediction_label (ex: 'Over 3.5' -> 3.5).
        
        Returns:
            int: Número de previsões corrigidas.
        """
        import re
        
        conn = self.connect()
        cursor = conn.cursor()
        
        # Busca predictions com valor 0 mas que têm label
        cursor.execute('''
            SELECT id, prediction_label 
            FROM predictions 
            WHERE (prediction_value IS NULL OR prediction_value = 0) 
              AND prediction_label IS NOT NULL
        ''')
        
        rows = cursor.fetchall()
        
        if not rows:
            print("✅ Nenhuma previsão precisa de correção.")
            return 0
        
        fixed_count = 0
        for pred_id, label in rows:
            # Extrai número do label (ex: "Over 3.5" -> 3.5)
            match = re.search(r'(\d+\.?\d*)', label or '')
            if match:
                line_value = float(match.group(1))
                cursor.execute("UPDATE predictions SET prediction_value = ? WHERE id = ?", (line_value, pred_id))
                fixed_count += 1
        
        conn.commit()
        print(f"✅ {fixed_count} previsões corrigidas.")
        return fixed_count

    def get_match_teams(self, match_id: int) -> tuple:
        """Busca os IDs dos times de uma partida."""
        conn = self.connect()
        cursor = conn.cursor()
        cursor.execute("SELECT home_team_id, away_team_id FROM matches WHERE match_id = ?", (match_id,))
        row = cursor.fetchone()
        return row if row else (None, None)

    def get_pending_matches(self) -> list:
        """
        Retorna lista de jogos pendentes (agendados ou em andamento).
        
        Returns:
            list: Lista de dicionários com dados dos jogos.
            
        Regra de Negócio:
            Identifica jogos que precisam de monitoramento ou atualização de status.
            Para "Todos os pendentes (histórico)", retorna jogos finalizados que ainda têm
            predições em aberto (GREEN/RED/PENDING), sem limite de tempo.
        """
        conn = self.connect()
        cursor = conn.cursor()
        
        import time
        now = int(time.time())
        
        query = '''
            SELECT DISTINCT m.match_id, m.home_team_name, m.away_team_name, m.status, m.start_timestamp
            FROM matches m
            WHERE (m.status = 'scheduled' AND m.start_timestamp < ?)
               OR (m.status = 'inprogress')
               OR (m.status = 'notstarted' AND m.start_timestamp < ?)
               OR (m.status = 'finished' AND EXISTS (
                    SELECT 1 FROM predictions p 
                    WHERE p.match_id = m.match_id 
                    AND (p.status = 'PENDING' OR p.status IS NULL)
               ))
            ORDER BY m.start_timestamp ASC
        '''
        
        cursor.execute(query, (now, now))
        rows = cursor.fetchall()
        
        matches = []
        for row in rows:
            matches.append({
                'match_id': row[0],
                'home_team': row[1],
                'away_team': row[2],
                'status': row[3],
                'start_timestamp': row[4]
            })
            
        return matches

    def get_predictions_by_date(self, date_str: str) -> list:
        """
        Busca jogos e suas predições para uma data específica (YYYY-MM-DD).
        Agora retorna estrutura agrupada e enriquecida por match_id.
        """
        from datetime import datetime
        import sqlite3
        conn = self.connect()
        cursor = conn.cursor()
        
        # Converte YYYY-MM-DD para range de timestamp (UTC-3 Fix)
        from datetime import timezone, timedelta
        brt = timezone(timedelta(hours=-3))
        dt_start = datetime.strptime(date_str, '%Y-%m-%d').replace(tzinfo=brt)
        start_ts = int(dt_start.timestamp())
        dt_end = dt_start.replace(hour=23, minute=59, second=59)
        end_ts = int(dt_end.timestamp())
        
        query = '''
            SELECT m.match_id, m.home_team_name, m.away_team_name, m.tournament_name,
                   p.prediction_label, p.confidence, p.fair_odds, p.prediction_value,
                   p.feedback_text, m.start_timestamp, m.home_score, m.away_score, 
                   m.round, p.model_version, p.market_group, m.status, p.status,
                   m.match_minute, p.raw_model_score, p.category,
                   m.home_league_position, m.away_league_position
            FROM matches m
            LEFT JOIN predictions p ON m.match_id = p.match_id
            WHERE m.start_timestamp BETWEEN ? AND ?
            ORDER BY m.match_id, p.confidence DESC
        '''
        
        cursor.execute(query, (start_ts, end_ts))
        rows = cursor.fetchall()
        
        # Agrupamento por match_id
        matches_dict = {}
        for row in rows:
            match_id = row[0]
            if match_id not in matches_dict:
                matches_dict[match_id] = {
                    "id": match_id,
                    "match_name": f"{row[1]} vs {row[2]}",
                    "tournament": row[3],
                    "timestamp": row[9],
                    "home_score": row[10],
                    "away_score": row[11],
                    "round": row[12],
                    "status": row[15],
                    "match_minute": row[17],  # <-- FIXED: Live status time
                    "ml_score": None, # Será extraído do model_version Professional
                    "predictions": [],
                    "max_confidence": 0,
                    "home_position": row[20],
                    "away_position": row[21]
                }
            
            # Dados da predição (Apenas se existir)
            if row[4]: # prediction_label
                pred_data = {
                    "prediction_label": row[4],
                    "confidence": int(row[5] * 100) if row[5] is not None and row[5] <= 1 else int(row[5] or 0),
                    "fair_odds": row[6],
                    "prediction_value": row[7],
                    "feedback_text": row[8] or "Nenhum detalhe disponível.",
                    "model_version": row[13],
                    "market_group": row[14] or "Geral",
                    "status": row[16],
                    "raw_model_score": row[18],
                    "category": row[19]
                }
                
                # Atualiza nota ML principal (Padronizado: usa raw_model_score se disponível)
                if row[13] == 'CORTEX_V2.1_CALIBRATED' and matches_dict[match_id]["ml_score"] is None:
                    # PRIORIDADE: raw_model_score (Lambda real) > prediction_value (linha de mercado)
                    matches_dict[match_id]["ml_score"] = row[18] or row[7]
                
                # REMOVED: Fallback problemático que pegava valores de linha (10.5, 12.5) como ml_score
    
                # Adiciona esta predição à lista
                matches_dict[match_id]["predictions"].append(pred_data)
                
                # Atualiza confiança máxima do jogo para ordenação
                matches_dict[match_id]["max_confidence"] = max(matches_dict[match_id]["max_confidence"], pred_data["confidence"])
            
    
        # Fallback para ml_score (Garante que mini-cards não fiquem 0.0 se não houver explicitamente CORTEX_V2.1_CALIBRATED)
        for m_id in matches_dict:
            if matches_dict[m_id]["ml_score"] is None and matches_dict[m_id]["predictions"]:
                # Tenta pegar Professional V2 ou o primeiro disponível
                pref = next((p for p in matches_dict[m_id]["predictions"] if p['model_version'] == 'CORTEX_V2.1_CALIBRATED'), None)
                if pref:
                    matches_dict[m_id]["ml_score"] = pref.get('raw_model_score') or pref['prediction_value']
                else:
                    matches_dict[m_id]["ml_score"] = matches_dict[m_id]["predictions"][0].get('raw_model_score') or matches_dict[m_id]["predictions"][0]['prediction_value']

        # Converte para lista e ordena por relevância (confiança máxima)
        results = list(matches_dict.values())
        results.sort(key=lambda x: x["max_confidence"], reverse=True)
        
        return results

    def get_match_analysis(self, match_id):
        """
        Busca dados detalhados de um jogo e suas predições pelo match_id.
        """
        conn = self.connect()
        cursor = conn.cursor()
        
        query = '''
            SELECT m.match_id, m.home_team_name, m.away_team_name, m.tournament_name,
                   p.prediction_label, p.confidence, p.fair_odds, p.prediction_value,
                   p.feedback_text, m.start_timestamp, m.home_score, m.away_score, 
                   m.round, p.model_version, p.market_group, m.status, p.status,
                   s.corners_home_ft, s.corners_away_ft, s.corners_home_ht, s.corners_away_ht,
                   m.home_team_id, m.away_team_id, m.home_league_position, m.away_league_position,
                   p.category
            FROM matches m
            LEFT JOIN predictions p ON m.match_id = p.match_id
            LEFT JOIN match_stats s ON m.match_id = s.match_id
            WHERE m.match_id = ?
            ORDER BY p.confidence DESC
        '''
        
        cursor.execute(query, (match_id,))
        rows = cursor.fetchall()
        
        if not rows:
            return None
            
        # Pega dados do primeiro row para info básica do match
        first = rows[0]
        match_data = {
            "id": match_id,
            "match_name": f"{first[1]} vs {first[2]}",
            "home_team": first[1],     # Nome do time casa (para Raio-X)
            "away_team": first[2],     # Nome do time visitante (para Raio-X)
            "home_team_id": first[21], # ID do time casa (CRITICAL para Raio-X)
            "away_team_id": first[22], # ID do time visitante (CRITICAL para Raio-X)
            "home_position": first[23],     # Posição na tabela (casa)
            "away_position": first[24],     # Posição na tabela (visitante)
            "tournament": first[3],
            "timestamp": first[9],
            "home_score": first[10],
            "away_score": first[11],
            "round": first[12],
            "status": first[15],
            "ml_score": None,
            "predictions": [],
            "stats": {
                "corners_home_ft": first[17],
                "corners_away_ft": first[18],
                "corners_home_ht": first[19],
                "corners_away_ht": first[20]
            }
        }
        
        for row in rows:
            # Dados da predição (Apenas se existir)
            if row[4]: # prediction_label
                pred_data = {
                    "prediction_label": row[4],
                    "confidence": int(row[5] * 100) if row[5] is not None and row[5] <= 1 else int(row[5] or 0),
                    "fair_odds": row[6],
                    "prediction_value": row[7],
                    "feedback_text": row[8] or "Nenhum detalhe disponível.",
                    "model_version": row[13],
                    "market_group": row[14] or "Geral",
                    "status": row[16] or 'PENDING',
                    "category": row[25]
                }
                
                # Seleção do score principal (Big Digit)
                if row[13] == 'CORTEX_V2.1_CALIBRATED' and match_data["ml_score"] is None and row[7] is not None:
                    match_data["ml_score"] = row[7]
                
                match_data["predictions"].append(pred_data)
            
        # Fallback se ml_score ainda for None
        if match_data["ml_score"] is None and match_data["predictions"]:
            match_data["ml_score"] = match_data["predictions"][0]["prediction_value"]
            
        return match_data

    def get_dashboard_stats(self, user_id: int = None) -> dict:
        """
        Retorna estatísticas agregadas para a Dashboard "Visão Estratégica".
        
        Args:
            user_id (int, optional): ID do usuário para filtrar o lucro.
        
        Regra de Negócio:
            - Lucro Líquido: Calculado a partir das apostas do usuário (ou total se None)
            - Assertividade: % de acertos nas previsões TOP7 (Global, pois é IA)
            - GREENs/REDs/Aguardando: Contagem de previsões TOP7 (Global, pois é IA)
            - Acurácia ML: Erro absoluto médio (Global)
            - Jogos Analisados: Total de jogos no banco
            
        Returns:
            dict: Dicionário com métricas para o dashboard.
        """
        conn = self.connect()
        cursor = conn.cursor()
        
        stats = {
            'total_matches': 0,
            'greens_top7': 0,
            'reds_top7': 0,
            'pending_top7': 0,
            'assertivity': 0.0,
            'net_profit': 0.0,
            'ml_accuracy': 0.0,
            'model_version': 'CORTEX_V2.1_CALIBRATED',
            'total_features': 45
        }
        
        try:
            # 1. Total de Jogos (TODOS, não apenas finished)
            cursor.execute('SELECT COUNT(*) FROM matches')
            stats['total_matches'] = cursor.fetchone()[0] or 0
            
            # 2. Contagem TOP7 por Status (Isso deve ser Global pois é performance da IA?)
            # REGRA: Performance da IA (TOP 7) é global. Apenas Lucro Líquido é pessoal.
            cursor.execute('''
                SELECT 
                    SUM(CASE WHEN status = 'GREEN' THEN 1 ELSE 0 END) as greens,
                    SUM(CASE WHEN status = 'RED' THEN 1 ELSE 0 END) as reds,
                    SUM(CASE WHEN status = 'PENDING' THEN 1 ELSE 0 END) as pending
                FROM predictions 
                WHERE category = 'Top7'
            ''')
            row = cursor.fetchone()
            if row:
                stats['greens_top7'] = row[0] or 0
                stats['reds_top7'] = row[1] or 0
                stats['pending_top7'] = row[2] or 0
            
            # 3. Assertividade (% de acerto TOP7)
            total_resolved = stats['greens_top7'] + stats['reds_top7']
            if total_resolved > 0:
                stats['assertivity'] = (stats['greens_top7'] / total_resolved) * 100
            
            # 4. Lucro Líquido (Baseado em apostas PESSOAIS)
            query_profit = '''
                SELECT 
                    SUM(CASE WHEN status = 'GREEN' THEN possible_win - stake ELSE 0 END) -
                    SUM(CASE WHEN status = 'RED' THEN stake ELSE 0 END) as net,
                    SUM(CASE WHEN status IN ('GREEN', 'RED') THEN stake ELSE 0 END) as invested
                FROM bets
                WHERE (? IS NULL OR user_id = ?)
            '''
            cursor.execute(query_profit, (user_id, user_id))
            profit_row = cursor.fetchone()
            stats['net_profit'] = profit_row[0] if profit_row and profit_row[0] else 0.0
            stats['total_invested'] = profit_row[1] if profit_row and profit_row[1] else 0.0
            
            # 5. Acurácia ML (MAE - Erro Absoluto Médio)
            cursor.execute('''
                SELECT AVG(ABS(p.prediction_value - (s.corners_home_ft + s.corners_away_ft))) as mae
                FROM predictions p
                JOIN match_stats s ON p.match_id = s.match_id
                WHERE p.model_version = 'ML_V10' 
                AND p.status IN ('GREEN', 'RED')
            ''')
            mae_row = cursor.fetchone()
            if mae_row and mae_row[0]:
                stats['ml_accuracy'] = round(mae_row[0], 2)
                
        except Exception as e:
            print(f"Erro ao calcular stats do dashboard: {e}")
            
        return stats
    
    def get_betting_statistics(self, user_id=None) -> Dict[str, Any]:
        """
        Retorna estatísticas completas de apostas para a UI 'Minhas Apostas'.
        
        Regra de Negócio:
            Centraliza todos os dados necessários para a interface de gestão de banca:
            - Saldo atual, ROI e taxa de acerto
            - Lista de apostas com status e items
            - Breakdown por mercado e evolução temporal
        
        Returns:
            Dict com todas as estatísticas de apostas.
        """
        conn = self.connect()
        cursor = conn.cursor()
        
        stats = {
            'saldo': 0.0,
            'roi': 0.0,
            'taxa_acerto': 0.0,
            'total_apostas': 0,
            'ganhas': 0,
            'perdidas': 0,
            'pendentes': 0,
            'bets': [],
            'weekly_stats': [],
            'market_stats': {}
        }
        
        try:
            # 1. Contagem de Apostas por Status
            cursor.execute('''
                SELECT 
                    COUNT(*) as total,
                    SUM(CASE WHEN status = 'GREEN' THEN 1 ELSE 0 END) as ganhas,
                    SUM(CASE WHEN status = 'RED' THEN 1 ELSE 0 END) as perdidas,
                    SUM(CASE WHEN status = 'PENDING' THEN 1 ELSE 0 END) as pendentes
                FROM bets
                WHERE (? IS NULL OR user_id = ?)
            ''', (user_id, user_id))
            row = cursor.fetchone()
            if row:
                stats['total_apostas'] = row[0] or 0
                stats['ganhas'] = row[1] or 0
                stats['perdidas'] = row[2] or 0
                stats['pendentes'] = row[3] or 0
            
            # 2. Taxa de Acerto
            total_resolved = stats['ganhas'] + stats['perdidas']
            if total_resolved > 0:
                stats['taxa_acerto'] = (stats['ganhas'] / total_resolved) * 100
            
            # 3. Saldo e ROI
            cursor.execute('''
                SELECT 
                    SUM(CASE WHEN status = 'GREEN' THEN possible_win ELSE 0 END) as ganhos,
                    SUM(CASE WHEN status IN ('GREEN', 'RED') THEN stake ELSE 0 END) as investido,
                    SUM(stake) as total_stake
                FROM bets
                WHERE (? IS NULL OR user_id = ?)
            ''', (user_id, user_id))
            row = cursor.fetchone()
            if row:
                ganhos = row[0] or 0
                investido = row[1] or 0
                total_stake = row[2] or 0
                stats['saldo'] = ganhos - investido + total_stake  # Saldo = ganhos - perdas
                if investido > 0:
                    stats['roi'] = ((ganhos - investido) / investido) * 100
            
            # 3.1 Weekly Stats (Win/Loss)
            cursor.execute('''
                SELECT 
                    strftime('%W', datetime(timestamp, 'unixepoch')) as week,
                    SUM(CASE WHEN status = 'GREEN' THEN 1 ELSE 0 END) as wins,
                    SUM(CASE WHEN status = 'RED' THEN 1 ELSE 0 END) as losses
                FROM bets
                WHERE (? IS NULL OR user_id = ?) AND status IN ('GREEN', 'RED')
                GROUP BY week
                ORDER BY week ASC
                LIMIT 12
            ''', (user_id, user_id))
            
            stats['weekly_stats'] = [
                {'week': f"Semana {row[0]}", 'wins': row[1], 'losses': row[2]}
                for row in cursor.fetchall()
            ]
            
            # 4. Lista de Apostas com Items
            cursor.execute('''
                SELECT id, timestamp, status, stake, total_odds, possible_win, bet_type
                FROM bets
                WHERE (? IS NULL OR user_id = ?)
                ORDER BY timestamp DESC
                LIMIT 50
            ''', (user_id, user_id))
            bets = cursor.fetchall()
            
            for bet in bets:
                bet_id = bet[0]
                cursor.execute('''
                    SELECT bi.match_id, bi.prediction_label, bi.odds, bi.status, m.home_team_name, m.away_team_name
                    FROM bet_items bi
                    LEFT JOIN matches m ON bi.match_id = m.match_id
                    WHERE bi.bet_id = ?
                ''', (bet_id,))
                items = cursor.fetchall()
                
                stats['bets'].append({
                    'id': bet_id,
                    'timestamp': bet[1],
                    'status': bet[2],
                    'stake': bet[3],
                    'total_odds': bet[4],
                    'possible_win': bet[5],
                    'bet_type': bet[6],
                    'items': [
                        {
                            'match_id': item[0],
                            'prediction_label': item[1],
                            'odds': item[2],
                            'status': item[3],
                            'match_name': f"{item[4]} vs {item[5]}" if item[4] and item[5] else "Jogo Desconhecido"
                        }
                        for item in items
                    ]
                })
            
            # 5. Performance por Mercado (baseada em predictions TOP7)
            cursor.execute('''
                SELECT 
                    market_group,
                    COUNT(*) as total,
                    SUM(CASE WHEN status = 'GREEN' THEN 1 ELSE 0 END) as wins
                FROM predictions
                WHERE category = 'Top7' AND status IN ('GREEN', 'RED')
                GROUP BY market_group
            ''')
            for row in cursor.fetchall():
                market = row[0] or 'Outro'
                stats['market_stats'][market] = {
                    'total': row[1],
                    'wins': row[2],
                    'rate': (row[2] / row[1] * 100) if row[1] > 0 else 0
                }
                
        except Exception as e:
            print(f"Erro ao calcular betting stats: {e}")
            
        return stats

    def verify_login(self, username, password):
        """Verifica credenciais de login."""
        conn = self.connect()
        cursor = conn.cursor()
        cursor.execute("SELECT id, username, password_hash, role, initial_bankroll FROM users WHERE username = ?", (username,))
        user = cursor.fetchone()
        
        if user and check_password_hash(user[2], password):
            return {
                "id": user[0],
                "username": user[1],
                "role": user[3],
                "bankroll": user[4]
            }
        return None

    def get_user_by_username(self, username):
        conn = self.connect()
        cursor = conn.cursor()
        cursor.execute("SELECT id, username, role, initial_bankroll FROM users WHERE username = ?", (username,))
        row = cursor.fetchone()
        if row:
            return {"id": row[0], "username": row[1], "role": row[2], "bankroll": row[3]}
        return None

    def create_user(self, username, password, role='user', initial_bankroll=1000.0):
        """Cria um novo usuário (para seed/admin)."""
        conn = self.connect()
        cursor = conn.cursor()
        try:
            password_hash = generate_password_hash(password)
            cursor.execute(
                "INSERT INTO users (username, password_hash, role, initial_bankroll) VALUES (?, ?, ?, ?)",
                (username, password_hash, role, initial_bankroll)
            )
            conn.commit()
            return True
        except sqlite3.IntegrityError:
            return False
            
    def update_user_password(self, username, password):
        conn = self.connect()
        cursor = conn.cursor()
        password_hash = generate_password_hash(password)
        cursor.execute("UPDATE users SET password_hash = ? WHERE username = ?", (password_hash, username))
        conn.commit()

    def get_all_users_stats(self):
        """
        Retorna estatísticas de todos os usuários para o Social Ranking.
        """
        conn = self.connect()
        cursor = conn.cursor()
        
        # Pega todos os usuários
        cursor.execute("SELECT id, username, initial_bankroll FROM users")
        users = cursor.fetchall()
        
        stats = []
        for u in users:
            uid, uname, ubank = u
            
            # Calcula stats baseado na tabela bets filtrando por user_id
            cursor.execute('''
                SELECT 
                    COUNT(*),
                    SUM(CASE WHEN status='GREEN' THEN 1 ELSE 0 END),
                    SUM(CASE WHEN status='GREEN' THEN possible_win - stake ELSE 
                        CASE WHEN status='RED' THEN -stake ELSE 0 END END) as profit,
                    SUM(stake)
                FROM bets
                WHERE user_id = ?
            ''', (uid,))
            
            bet_stats = cursor.fetchone()
            total_bets = bet_stats[0] or 0
            wins = bet_stats[1] or 0
            profit = bet_stats[2] or 0.0
            total_staked = bet_stats[3] or 0.0
            
            roi = (profit / total_staked * 100) if total_staked > 0 else 0.0
            win_rate = (wins / total_bets * 100) if total_bets > 0 else 0.0
            
            stats.append({
                "username": uname,
                "total_bets": total_bets,
                "win_rate": win_rate,
                "profit": profit,
                "roi": roi,
                "bankroll": ubank + profit # Saldo atual estimado
            })
            
        return sorted(stats, key=lambda x: x['profit'], reverse=True)

    def get_bets_by_user(self, username, limit=10):
        """Retorna as últimas apostas de um usuário específico."""
        user = self.get_user_by_username(username)
        if not user: return []
        
        user_id = user['id']
        
        conn = self.connect()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT id, timestamp, status, stake, total_odds, possible_win, bet_type
            FROM bets
            WHERE user_id = ?
            ORDER BY timestamp DESC
            LIMIT ?
        ''', (user_id, limit))
        
        bets = []
        for row in cursor.fetchall():
             bets.append({
                'id': row[0],
                'timestamp': row[1],
                'status': row[2],
                'stake': row[3],
                'total_odds': row[4],
                'possible_win': row[5],
                'bet_type': row[6]
            })
        return bets

    def get_bet_items(self, bet_id: int) -> list:
        """
        Retorna os itens (seleções) de uma aposta específica.
        
        Regra de Negócio:
            Usado para mostrar detalhes da aposta no Social Ranking.
            Inclui nome do jogo via JOIN com matches.
        """
        conn = self.connect()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT bi.prediction_label, bi.odds, bi.status,
                   m.home_team_name, m.away_team_name, m.match_id
            FROM bet_items bi
            LEFT JOIN matches m ON bi.match_id = m.match_id
            WHERE bi.bet_id = ?
        ''', (bet_id,))
        
        items = []
        for row in cursor.fetchall():
            match_name = f"{row[3] or '?'} vs {row[4] or '?'}"
            items.append({
                'prediction_label': row[0],
                'odds': row[1],
                'status': row[2],
                'match_name': match_name,
                'match_id': row[5]
            })
        return items

    def get_user_h2h(self, username1: str, username2: str) -> dict:
        """
        Compara estatísticas de dois usuários (Head-to-Head).
        
        Regra de Negócio:
            Permite comparação direta entre apostadores no Social Ranking.
        """
        all_stats = self.get_all_users_stats()
        
        user1_stats = next((u for u in all_stats if u['username'] == username1), None)
        user2_stats = next((u for u in all_stats if u['username'] == username2), None)
        
        return {
            'user1': user1_stats,
            'user2': user2_stats
        }

    def save_bet(self, user_id, stake, total_odds, possible_win, bet_type, items):
        """Salva uma nova aposta no banco."""
        conn = self.connect()
        cursor = conn.cursor()
        import time
        
        try:
            cursor.execute('''
                INSERT INTO bets (user_id, timestamp, status, stake, total_odds, possible_win, bet_type)
                VALUES (?, ?, 'PENDING', ?, ?, ?, ?)
            ''', (user_id, int(time.time()), stake, total_odds, possible_win, bet_type))
            
            bet_id = cursor.lastrowid
            
            for item in items:
                cursor.execute('''
                    INSERT INTO bet_items (bet_id, match_id, prediction_label, odds, status)
                    VALUES (?, ?, ?, ?, 'PENDING')
                ''', (bet_id, item['match_id'], item['label'], item['odds']))
                
            conn.commit()
            return True, "Aposta registrada com sucesso!"
        except Exception as e:
            return False, f"Erro ao salvar aposta: {e}"

    def list_users(self):
        """Retorna lista de todos os usuários."""
        conn = self.connect()
        cursor = conn.cursor()
        cursor.execute("SELECT id, username, role, initial_bankroll FROM users")
        return cursor.fetchall()

    def delete_user(self, username):
        """Deleta um usuário pelo nome."""
        conn = self.connect()
        cursor = conn.cursor()
        try:
            cursor.execute("DELETE FROM users WHERE username = ?", (username,))
            if cursor.rowcount > 0:
                conn.commit()
                return True
            return False
        except Exception as e:
            return False

    def check_bets(self):
        """
        Verifica e atualiza o status das apostas pendentes.
        """
        conn = self.connect()
        cursor = conn.cursor()
        
        # 1. Busca apostas pendentes
        cursor.execute("SELECT id, bet_type FROM bets WHERE status = 'PENDING'")
        pending_bets = cursor.fetchall()
        
        if not pending_bets:
            return 0
            
        resolved_count = 0
        
        for bet_id, bet_type in pending_bets:
            # 2. Busca itens da aposta
            cursor.execute('''
                SELECT bi.id, bi.match_id, bi.prediction_label, bi.status
                FROM bet_items bi
                WHERE bi.bet_id = ?
            ''', (bet_id,))
            items = cursor.fetchall()
            
            items_statuses = []
            
            # 3. Verifica status de cada item
            item_updates_needed = False
            
            for item in items:
                item_id, match_id, label, status = item
                
                # Se item já resolvido, mantém
                if status in ('GREEN', 'RED', 'VOID'):
                    items_statuses.append(status)
                    continue
                    
                # Busca status do jogo e estatísticas
                cursor.execute("SELECT status, home_score, away_score FROM matches WHERE match_id = ?", (match_id,))
                match_row = cursor.fetchone()
                
                if not match_row:
                    items_statuses.append(status) # Mantém pendente se jogo sumiu
                    continue
                    
                match_status = match_row[0]
                
                # Se jogo cancelado/adiado -> VOID
                if match_status in ('postponed', 'canceled', 'adiado', 'cancelado'):
                    new_status = 'VOID'
                    cursor.execute("UPDATE bet_items SET status = ? WHERE id = ?", (new_status, item_id))
                    items_statuses.append(new_status)
                    item_updates_needed = True
                    continue

                # Se jogo não terminou, item continua pendente
                if match_status != 'finished':
                    items_statuses.append('PENDING')
                    continue
                    
                # JOGO FINALIZADO: Verificar resultado
                cursor.execute("SELECT corners_home_ft, corners_away_ft FROM match_stats WHERE match_id = ?", (match_id,))
                stats_row = cursor.fetchone()
                
                if not stats_row:
                    # Tenta buscar stats se não existirem (fallback básico ou mantém pendente)
                    items_statuses.append('PENDING') 
                    continue
                    
                corners_ft = (stats_row[0] or 0) + (stats_row[1] or 0)
                
                # Lógica de Resolução (Parser de Labels)
                new_status = 'RED'
                label_lower = label.lower()
                
                import re
                # Extrai linha numérica
                line_match = re.search(r'(\d+\.?\d*)', label)
                line = float(line_match.group(1)) if line_match else 0.0
                
                # Over/Under
                if 'over' in label_lower or 'mais' in label_lower:
                    if corners_ft > line: new_status = 'GREEN'
                elif 'under' in label_lower or 'menos' in label_lower:
                    if corners_ft < line: new_status = 'GREEN'
                
                cursor.execute("UPDATE bet_items SET status = ? WHERE id = ?", (new_status, item_id))
                items_statuses.append(new_status)
                item_updates_needed = True
            
            if item_updates_needed:
                conn.commit()
                
            # 4. Determina status final da aposta (Bet)
            final_status = 'PENDING'
            
            if 'PENDING' in items_statuses:
                final_status = 'PENDING'
            elif 'RED' in items_statuses:
                final_status = 'RED'
            elif all(s == 'GREEN' or s == 'VOID' for s in items_statuses):
                # Se todos green/void, é GREEN (com possible_win ajustado se void, mas simplificado aqui)
                # Se TODAS forem VOID, a aposta é VOID.
                if all(s == 'VOID' for s in items_statuses):
                    final_status = 'VOID'
                else:
                    final_status = 'GREEN'
            
            if final_status != 'PENDING':
                cursor.execute("UPDATE bets SET status = ? WHERE id = ?", (final_status, bet_id))
                resolved_count += 1
                
        conn.commit()
        return resolved_count