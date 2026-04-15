"""
MatchRepository - Repositório de Partidas.

Regra de Negócio:
    Encapsula todo o CRUD de partidas e estatísticas,
    extraído do monolítico DBManager para Single Responsibility.
"""

import sqlite3
import pandas as pd
from typing import Optional, Dict, Any, List


class MatchRepository:
    """Repositório especializado em operações de partidas e estatísticas."""

    def __init__(self, db_manager):
        """
        Args:
            db_manager: Instância do DBManager (para acesso à conexão compartilhada).
        """
        self._db = db_manager

    def save_match(self, match_data: dict) -> None:
        """
        Salva ou atualiza os dados básicos de uma partida.

        Args:
            match_data: Dicionário com dados da partida (id, times, placar, etc).

        Regra de Negócio:
            Centraliza a persistência de dados brutos das partidas para histórico e feature engineering.
        """
        conn = self._db.connect()
        cursor = conn.cursor()

        # AUTO-MIGRATE IDS (FIREWALL)
        unified_ids = {
            1: 17,   # Premier League
            42: 35,  # Bundesliga
            36: 8,   # LaLiga
            33: 23,  # Serie A
            4: 34    # Ligue 1
        }

        original_id = match_data.get('tournament_id')
        if original_id in unified_ids:
            match_data['tournament_id'] = unified_ids[original_id]

        # Defensive Logic: Prevent overwriting live minute with None
        if match_data.get('status') == 'inprogress' and match_data.get('match_minute') is None:
            try:
                exist = cursor.execute("SELECT match_minute FROM matches WHERE match_id=?", (match_data['id'],)).fetchone()
                if exist and exist[0]:
                    match_data['match_minute'] = exist[0]
            except:
                pass

        try:
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
            match_id: ID da partida.
            stats_data: Dicionário com estatísticas (escanteios, chutes, etc).

        Regra de Negócio:
            Armazena métricas profundas usadas para calcular médias e tendências dos times.
        """
        conn = self._db.connect()
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
        except Exception as e:
            print(f"Erro ao salvar stats do jogo {match_id}: {e}")

    def get_historical_data(self) -> pd.DataFrame:
        """
        Recupera todo o histórico de partidas finalizadas com estatísticas.

        Returns:
            DataFrame contendo dados de partidas e estatísticas unificadas.

        Regra de Negócio:
            Fornece a base de dados completa para o treinamento do modelo de ML.
        """
        conn = self._db.connect()
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
            season_id: ID da temporada.

        Returns:
            dict: {'total_matches': int, 'last_round': int}
        """
        conn = self._db.connect()
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

    def get_match_teams(self, match_id: int) -> tuple:
        """Busca os IDs dos times de uma partida."""
        conn = self._db.connect()
        cursor = conn.cursor()
        cursor.execute("SELECT home_team_id, away_team_id FROM matches WHERE match_id = ?", (match_id,))
        row = cursor.fetchone()
        return row if row else (None, None)

    def get_pending_matches(self) -> list:
        """
        Retorna lista de jogos pendentes (agendados ou em andamento).

        Returns:
            Lista de dicionários com dados dos jogos.

        Regra de Negócio:
            Identifica jogos que precisam de monitoramento ou atualização de status.
        """
        conn = self._db.connect()
        cursor = conn.cursor()

        import time
        now = int(time.time())

        query = '''
            SELECT match_id, home_team_name, away_team_name, status, start_timestamp
            FROM matches 
            WHERE (status = 'scheduled' AND start_timestamp < ?)
               OR (status = 'inprogress')
               OR (status = 'notstarted' AND start_timestamp < ?)
               OR (status = 'finished' AND start_timestamp > ? - 10800)
            ORDER BY start_timestamp ASC
        '''

        cursor.execute(query, (now, now, now))
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
