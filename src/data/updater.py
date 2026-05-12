"""
Updater Module - Cortex ML V2.1
Handles database updates via scraping (SofaScore).
"""

import os
import sys
import re
import json
import traceback
from datetime import datetime
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.database.db_manager import DBManager
from src.scrapers.sofascore import SofaScoreScraper
from src.analysis.statistical import Colors

def load_leagues_config() -> list:
    """Carrega a lista de ligas configuradas do arquivo 'clubes_sofascore.json'."""
    try:
        config_path = os.path.join(os.path.dirname(__file__), '..', '..', 'clubes_sofascore.json')
        with open(config_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return data.get('competicoes', [])
    except Exception as e:
        print(f"Erro ao carregar config de ligas: {e}")
        return []

def update_database(
    league_name: str = "Brasileirão Série A",
    season_year: str = "2026",
    start_date: str | None = None,
    end_date: str | None = None,
) -> None:
    """Atualiza o banco de dados com inteligência incremental."""
    db = DBManager()
    
    # Check for feedback loop updates first
    print("Verificando resultados de previsões anteriores...")
    db.check_predictions()
    
    scraper = SofaScoreScraper(headless=True)
    
    try:
        scraper.start()
        
        # 1. Get Tournament/Season IDs
        t_id = scraper.get_tournament_id(league_name)
        if not t_id:
            print("Torneio não encontrado.")
            return
            
        s_id = scraper.get_season_id(t_id, season_year)
        if not s_id:
            print("Temporada não encontrada.")
            return
            
        print(f"ID Torneio: {t_id}, ID Temporada: {s_id}")
        
        # --- VERIFICAÇÃO DE INTEGRIDADE ---
        stats = db.get_season_stats(s_id)
        total_matches_db = stats['total_matches']
        last_round_db = stats['last_round']
        
        print(f"Status Atual no DB: {total_matches_db} jogos, Última Rodada: {last_round_db}")
        
        # Lógica: Se já tem +370 jogos e não é temporada atual, considera completo
        is_current_season = "2026" in season_year or "26/27" in season_year
        if total_matches_db > 370 and not is_current_season:
            print(f"✅ Temporada {season_year} já está completa no banco ({total_matches_db} jogos). Pulando...")
            return

        # Define rodada inicial (Incremental)
        start_round = 1
        if last_round_db > 0:
            start_round = last_round_db
            print(f"⏩ Retomando atualização a partir da rodada {start_round}...")
        
        # 2. Get Matches
        matches = scraper.get_matches(t_id, s_id, start_round=start_round)
        print(f"Encontrados {len(matches)} jogos novos/atualizados.")
        
        # Optional date-range filter (YYYY-MM-DD) based on match startTimestamp
        if start_date or end_date:
            start_ts = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp()) if start_date else None
            end_ts = int(datetime.strptime(end_date, "%Y-%m-%d").timestamp()) if end_date else None
            filtered_matches = []
            for m in matches:
                ts = m.get("startTimestamp")
                if ts is None:
                    continue
                if start_ts is not None and ts < start_ts:
                    continue
                if end_ts is not None and ts > end_ts + 86399:
                    continue
                filtered_matches.append(m)
            matches = filtered_matches
            print(f"🔎 Filtro por data ativo: {start_date or '-'} até {end_date or '-'} | {len(matches)} jogos")

        # 3. Process Matches & Stats
        for i, m in enumerate(matches):
            if m['status']['type'] == 'finished':
                print(f"[{i+1}/{len(matches)}] Processando {m['homeTeam']['name']} vs {m['awayTeam']['name']}...")
                
                # Save Match Info
                match_data = {
                    'id': m['id'],
                    'tournament': m['tournament']['name'],
                    'tournament_id': m['tournament']['id'],
                    'season_id': s_id,
                    'round': m.get('roundInfo', {}).get('round', 0),
                    'status': m.get('status', {}).get('type', 'finished'),
                    'timestamp': m['startTimestamp'],
                    'home_id': m['homeTeam']['id'],
                    'home_name': m['homeTeam']['name'],
                    'away_id': m['awayTeam']['id'],
                    'away_name': m['awayTeam']['name'],
                    'home_score': m['homeScore']['display'],
                    'away_score': m['awayScore']['display']
                }
                db.save_match(match_data)
                
                # Get & Save Stats
                stats = scraper.get_match_stats(m['id'])
                db.save_stats(m['id'], stats)
                
    except Exception as e:
        print(f"Erro: {e}")
    finally:
        scraper.stop()
        db.close()

def update_match_by_url() -> None:
    """Atualiza dados de uma partida específica via URL."""
    url = input("Cole a URL do jogo do SofaScore: ")
    match_id_search = re.search(r'id:(\d+)', url)
    
    if not match_id_search:
        print("ID do jogo não encontrado na URL.")
        return

    match_id = match_id_search.group(1)
    print(f"Atualizando jogo ID: {match_id}...")
    
    scraper = SofaScoreScraper(headless=True)
    db = DBManager()
    
    try:
        scraper.start()
        
        api_url = f"https://www.sofascore.com/api/v1/event/{match_id}"
        ev_data = scraper._fetch_api(api_url)
        
        if not ev_data or 'event' not in ev_data:
            print("Erro ao buscar dados do jogo.")
            return
            
        ev = ev_data['event']
        match_name = f"{ev['homeTeam']['name']} vs {ev['awayTeam']['name']}"
        print(f"Jogo: {match_name} (Status: {ev['status']['type']})")
        
        match_data = {
            'id': match_id,
            'tournament': ev.get('tournament', {}).get('name', 'Unknown'),
            'tournament_id': ev.get('tournament', {}).get('id', 0),
            'season_id': ev.get('season', {}).get('id', 0),
            'round': ev.get('roundInfo', {}).get('round', 0),
            'status': ev.get('status', {}).get('type', 'unknown'),
            'timestamp': ev.get('startTimestamp', 0),
            'home_id': ev['homeTeam']['id'],
            'home_name': ev['homeTeam']['name'],
            'away_id': ev['awayTeam']['id'],
            'away_name': ev['awayTeam']['name'],
            'home_score': ev.get('homeScore', {}).get('display', 0),
            'away_score': ev.get('awayScore', {}).get('display', 0)
        }
        db.save_match(match_data)
        print("✅ Dados da partida atualizados.")
        
        if ev['status']['type'] == 'finished':
            print("Coletando estatísticas finais...")
            stats = scraper.get_match_stats(match_id)
            db.save_stats(match_id, stats)
            print("✅ Estatísticas salvas.")
            
            print("\nVerificando apostas pendentes...")
            db.check_predictions()
        else:
            print("⚠️ Jogo não finalizado. Estatísticas completas podem não estar disponíveis.")
            
    except Exception as e:
        print(f"Erro ao atualizar jogo: {e}")
    finally:
        scraper.stop()
        db.close()

def update_specific_league() -> None:
    """Atualiza dados completos de uma liga específica."""
    league_name = input("Nome da Liga (ex: 'Brasileirão Série A'): ")
    years = ["2023", "2024", "2025", "2026"]
    
    print(f"Atualizando {league_name} para os anos: {years}")
    for year in years:
        print(f"\n📅 Processando Temporada {year}...")
        update_database(league_name, year)

def update_all_leagues() -> None:
    """Executa atualização em lote de todas as ligas configuradas."""
    leagues = load_leagues_config()
    years = ["2023", "2024", "2025", "2026"]
    
    print(f"🚀 Iniciando atualização em lote de {len(leagues)} ligas...")
    
    for league in leagues:
        league_name = league['torneio']
        print(f"\n🏆 Liga: {league_name}")
        for year in years:
            print(f"   📅 Temporada {year}...")
            update_database(league_name, year)
            
    print("\n✅ Atualização em lote concluída!")


def update_all_leagues_by_date_range() -> None:
    """Atualiza todas as ligas configuradas aplicando filtro por intervalo de datas."""
    start_date = input("Data inicial (AAAA-MM-DD): ").strip()
    end_date = input("Data final   (AAAA-MM-DD): ").strip()

    try:
        datetime.strptime(start_date, "%Y-%m-%d")
        datetime.strptime(end_date, "%Y-%m-%d")
    except ValueError:
        print("❌ Datas inválidas. Use o formato AAAA-MM-DD.")
        return

    if start_date > end_date:
        print("❌ Data inicial maior que data final.")
        return

    update_all_leagues_by_date_range_values(start_date, end_date)


def update_all_leagues_by_date_range_values(start_date: str, end_date: str) -> None:
    """Versão não-interativa para atualização por faixa de datas."""
    leagues = load_leagues_config()
    years = ["2023", "2024", "2025", "2026"]

    print(f"🚀 Atualização em lote por intervalo [{start_date} -> {end_date}] para {len(leagues)} ligas...")
    for league in leagues:
        league_name = league["torneio"]
        print(f"\n🏆 Liga: {league_name}")
        for year in years:
            print(f"   📅 Temporada {year} com filtro de datas...")
            update_database(
                league_name=league_name,
                season_year=year,
                start_date=start_date,
                end_date=end_date,
            )

    print("\n✅ Atualização em lote por intervalo concluída!")
