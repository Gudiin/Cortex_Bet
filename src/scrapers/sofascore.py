import time
import random
import requests
from playwright.sync_api import sync_playwright

class SofaScoreScraper:
    def __init__(self, headless: bool = True, verbose: bool = True):
        self.headless = headless
        self.verbose = verbose
        self.playwright = None
        self.browser = None
        self.page = None
        self.last_momentum_data = []  # Store intercepted graph points for DA fallback
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "*/*",
            "Accept-Language": "pt-BR,pt;q=0.9,en-US;q=0.8,en;q=0.7",
            "Origin": "https://www.sofascore.com",
            "Referer": "https://www.sofascore.com/"
        })

    def start(self) -> None:
        self.playwright = sync_playwright().start()
        self.browser = self.playwright.chromium.launch(headless=self.headless)
        self.page = self.browser.new_page()
        
        # Intercept Network for Momentum Graph (Attack Momentum)
        def handle_response(response):
            try:
                # Look for graph API or JSON containing graphPoints
                if "graph" in response.url and response.status == 200:
                    data = response.json()
                    if "graphPoints" in data:
                        self.last_momentum_data = data["graphPoints"]
                    elif isinstance(data, dict) and "graphPoints" in str(data):
                        self.last_momentum_data = data.get("graphPoints", [])
            except:
                pass

        self.page.on("response", handle_response)

        # Suppress noisy third-party JS errors (mapbox-gl, performance API in workers)
        self.page.on("pageerror", lambda _: None)
        self.page.on("console", lambda msg: None if msg.type == "error" else None)

        self.page.set_extra_http_headers(self.session.headers)
        
        # Retry logic for initial connection
        max_retries = 3
        for attempt in range(max_retries):
            try:
                if self.verbose: print(f"Tentativa de conexão {attempt + 1}/{max_retries}...")
                self.page.goto("https://www.sofascore.com", timeout=60000, wait_until='domcontentloaded')
                break
            except Exception as e:
                if self.verbose: print(f"Erro na conexão (tentativa {attempt + 1}): {e}")
                if attempt == max_retries - 1:
                    raise e
                time.sleep(5)

    def stop(self) -> None:
        if self.browser:
            try: self.browser.close()
            except: pass
        if self.playwright:
            try: self.playwright.stop()
            except: pass

    def _fetch_api(self, url: str, retries: int = 2) -> dict | None:
        time.sleep(random.uniform(0.5, 1.5))
        script = f"""
            async () => {{
                try {{
                    const r = await fetch('{url}');
                    if (r.status !== 200) return null;
                    return await r.json();
                }} catch {{ return null; }}
            }}
        """
        
        for attempt in range(retries + 1):
            try:
                return self.page.evaluate(script)
            except Exception as e:
                if attempt < retries:
                    if "context" in str(e).lower() or "destroyed" in str(e).lower():
                        # Context lost - try to recover by reloading page
                        try:
                            if self.verbose: print(f"⚠️ Contexto perdido, tentando recuperar ({attempt + 1}/{retries})...")
                            self.page.reload(timeout=30000, wait_until='domcontentloaded')
                            time.sleep(2)
                            continue
                        except:
                            pass
                    time.sleep(1)
                else:
                    if self.verbose: print(f"❌ Erro no evaluation após {retries + 1} tentativas: {e}")
                    return None
        return None

    def get_tournament_id(self, query: str = "Brasileirão") -> int | None:
        tournament_mapping = {
            "Brasileirão Série A": "Brasileirão",
            "Brasileirão Série B": "Brasileirão Série B",
            "Serie A 25/26": "Serie A",
            "Serie A (Itália)": "Serie A",
            "La Liga": "LaLiga",
            "Ligue 1": "Ligue 1",
            "Bundesliga": "Bundesliga",
            "Premier League": "Premier League",
            "Liga Profesional (Argentina)": "Liga Profesional"
        }
        search_query = tournament_mapping.get(query, query)
        url = f"https://www.sofascore.com/api/v1/search/{search_query}"
        if self.verbose: print(f"Buscando torneio: {query}...")
        
        data = self._fetch_api(url)
        if data and 'results' in data:
            for item in data['results']:
                if item['type'] == 'uniqueTournament':
                    entity = item['entity']
                    if self.verbose: print(f"Encontrado: {entity['name']} (ID: {entity['id']})")
                    if search_query.lower() in entity['name'].lower() or entity['name'].lower() in search_query.lower():
                        return entity['id']
        return None

    def get_season_id(self, tournament_id: int, year: str = "2024") -> int | None:
        url = f"https://www.sofascore.com/api/v1/unique-tournament/{tournament_id}/seasons"
        data = self._fetch_api(url)
        if data and 'seasons' in data:
            for s in data['seasons']:
                if s['year'] == year:
                    return s['id']
                # Check for "24/25" format if year is "2025"
                if year == "2025" and s['year'] == "24/25":
                    return s['id']
                if year == "2024" and s['year'] == "23/24":
                    return s['id']
            
            try:
                year_int = int(year)
                prev_year = year_int - 1
                euro_format = f"{str(prev_year)[-2:]}/{str(year_int)[-2:]}"
                for s in data['seasons']:
                    if s['year'] == euro_format:
                        if self.verbose: print(f"Temporada encontrada (formato europeu): {euro_format}")
                        return s['id']
            except ValueError:
                pass
        return None

    def get_current_round(self, tournament_id: int, season_id: int) -> int | None:
        """
        Obtém o número da rodada atual de um torneio.
        
        Regra de Negócio:
            - Economia de API: Em vez de baixar todas as rodadas, perguntamos à API
              qual é a rodada atual (currentRound) para baixar apenas ela.
              
        Args:
            tournament_id: ID do torneio.
            season_id: ID da temporada.
            
        Returns:
            int: Número da rodada atual ou None se não encontrar.
        """
        url = f"https://www.sofascore.com/api/v1/unique-tournament/{tournament_id}/season/{season_id}/rounds"
        if self.verbose: print(f"Buscando rodada atual (Torneio {tournament_id})...")
        
        data = self._fetch_api(url)
        if data and 'currentRound' in data:
            return data['currentRound']['round']
        return None

    def get_scheduled_matches(self, date_str: str, league_ids: list = None) -> list:
        """
        Busca jogos agendados para uma data específica (Scanner de Oportunidades).
        Usa o endpoint global de eventos agendados (Lógica do CLI).
        """
        if not league_ids:
            league_ids = [325, 390, 17, 8, 35, 34, 23]
            
        if self.verbose: print(f"Iniciando Scanner para {date_str}...")
        
        api_url = f"https://www.sofascore.com/api/v1/sport/football/scheduled-events/{date_str}"
        data = self._fetch_api(api_url)
        
        if not data or 'events' not in data:
            if self.verbose: print(f"❌ Nenhum evento encontrado para {date_str}.")
            return []
            
        # Filtra por Top Ligas
        matches = []
        ignored_tournaments = ['Liigacup', 'Copa da Liga da Finlândia', 'Club Friendly']
        
        for event in data['events']:
            t_name = event.get('tournament', {}).get('name', '')
            if any(ignored in t_name for ignored in ignored_tournaments):
                 # Silently ignore
                 continue

            t_id = event.get('tournament', {}).get('uniqueTournament', {}).get('id')
            if t_id in league_ids:
                # Extrai dados relevantes
                evt_ts = event.get('startTimestamp', 0)
                # UTC-3 fixo para o Brasil
                import datetime
                tz_offset = datetime.timezone(datetime.timedelta(hours=-3))
                dt_obj = datetime.datetime.fromtimestamp(evt_ts, tz=tz_offset)
                start_time = dt_obj.strftime('%Y-%m-%d %H:%M')
                
                # STRICT FILTER (Fix Regra "Jogos de Ontem"):
                # A API retorna em UTC. Um jogo 22:00 BRT (do dia 4) é 01:00 UTC (do dia 5).
                # Se o user pediu dia 5, devemos ignorar esse jogo do dia 4.
                match_date_local = dt_obj.strftime('%Y-%m-%d')
                if match_date_local != date_str:
                    continue
                
                # Extrai status e calcula minuto se necessário
                status_type = event['status']['type']
                status_description = event['status'].get('description', '')
                
                # FILTER: REMOVED to allow database cleanup of these matches
                # ignored_statuses = ['canceled', 'postponed', 'interrupted', 'abandoned', 'coverage_canceled', 'delayed']
                # if status_type in ignored_statuses:
                #     if self.verbose: print(f"   ⚠️ Ignorando jogo {status_type}: {event['homeTeam']['name']} vs {event['awayTeam']['name']}")
                #     continue
                
                calculated_minute = status_description
                
                # 🚀 LIVE MINUTE CALCULATION (same logic as get_match_details)
                if status_type == 'inprogress' and status_description in ["1st half", "2nd half", "1º Tempo", "2º Tempo", ""]:
                    try:
                        current_period_start = event.get('currentPeriodStartTimestamp') or event.get('time', {}).get('currentPeriodStartTimestamp')
                        if current_period_start:
                            import time as t_module
                            now = int(t_module.time())
                            diff_seconds = now - current_period_start
                            minute_calc = int(diff_seconds / 60)
                            
                            if "2nd" in status_description or "2º" in status_description:
                                minute_calc += 45
                                
                            calculated_minute = f"{minute_calc}'"
                    except Exception:
                        pass
                
                match_info = {
                    'match_id': event['id'],
                    'tournament': event['tournament']['name'],
                    'tournament_id': t_id,
                    'season_id': event.get('season', {}).get('id', 0),  # Extract season_id for standings
                    'timestamp': evt_ts,
                    'home_team': event['homeTeam']['name'],
                    'home_id': event['homeTeam']['id'],
                    'away_team': event['awayTeam']['name'],
                    'away_id': event['awayTeam']['id'],
                    'start_time': start_time,
                    'status': status_type,
                    'status_description': calculated_minute,  # Now contains calculated minute for live games
                    'home_score': event.get('homeScore', {}).get('display', 0),
                    'away_score': event.get('awayScore', {}).get('display', 0),
                    'home_position': event.get('homeTeam', {}).get('ranking', None),
                    'away_position': event.get('awayTeam', {}).get('ranking', None)
                }
                matches.append(match_info)
                        
        if self.verbose: print(f"Scanner finalizado. {len(matches)} jogos encontrados nas ligas monitoradas para {date_str}.")
        return matches

    def get_matches(self, tournament_id: int, season_id: int, start_round: int = 1) -> list:
        """
        Coleta partidas com suporte a início customizado (start_round).
        """
        matches = []
        round_num = start_round
        max_rounds = 50 
        empty_rounds_limit = 3 
        empty_rounds = 0
        
        print(f"Iniciando coleta de partidas (Torneio {tournament_id}, Season {season_id})")
        print(f"🚀 Começando da Rodada {start_round}...")
        
        while round_num <= max_rounds:
            print(f"Coletando rodada {round_num}...", end='\r')
            url = f"https://www.sofascore.com/api/v1/unique-tournament/{tournament_id}/season/{season_id}/events/round/{round_num}"
            data = self._fetch_api(url)
            
            if data and 'events' in data and len(data['events']) > 0:
                matches.extend(data['events'])
                empty_rounds = 0
            else:
                empty_rounds += 1
                if empty_rounds >= empty_rounds_limit:
                    print(f"\nSem eventos por {empty_rounds_limit} rodadas consecutivas. Parando na rodada {round_num}.")
                    break
            
            round_num += 1
            
        print(f"\nTotal de partidas coletadas nesta execução: {len(matches)}")
        return matches

    def get_match_stats(self, match_id: int) -> dict:
        url = f"https://www.sofascore.com/api/v1/event/{match_id}/statistics"
        data = self._fetch_api(url)
        
        stats = {
            'corners_home_ft': 0, 'corners_away_ft': 0,
            'corners_home_ht': 0, 'corners_away_ht': 0,
            'shots_ot_home_ft': 0, 'shots_ot_away_ft': 0,
            'shots_ot_home_ht': 0, 'shots_ot_away_ht': 0,
            'possession_home': 0, 'possession_away': 0,
            'total_shots_home': 0, 'total_shots_away': 0,
            'fouls_home': 0, 'fouls_away': 0,
            'yellow_cards_home': 0, 'yellow_cards_away': 0,
            'red_cards_home': 0, 'red_cards_away': 0,
            'big_chances_home': 0, 'big_chances_away': 0,
            'expected_goals_home': 0.0, 'expected_goals_away': 0.0,
            'momentum_home': 0.0, 'momentum_away': 0.0,
            'momentum_peak_home': 0, 'momentum_peak_away': 0
        }

        if not data or 'statistics' not in data:
            return stats

        def extract_val(groups: list, keywords: list, is_home: bool, return_float: bool = False, extract_total: bool = False):
            if not groups:
                return 0.0 if return_float else 0
            for g in groups:
                if 'statisticsItems' not in g:
                    continue
                for item in g['statisticsItems']:
                    item_name_lower = item.get('name', '').lower()
                    for keyword in keywords:
                        if keyword.lower() in item_name_lower:
                            try:
                                value = item.get('home' if is_home else 'away')
                                if value is not None:
                                    # Handle "6/11 (55%)" format for Crosses, Duels, etc.
                                    if isinstance(value, str) and '/' in value:
                                        parts = value.split('/')
                                        if extract_total:
                                            # "11 (55%)" -> "11"
                                            val_str = parts[1].strip().split(' ')[0]
                                            return int(val_str)
                                        else:
                                            # "6"
                                            return int(parts[0].strip())

                                    if isinstance(value, str) and '%' in value:
                                        return int(value.replace('%', ''))
                                    
                                    if return_float:
                                        return float(value)
                                    else:
                                        return int(value)
                            except (ValueError, TypeError, IndexError):
                                return 0.0 if return_float else 0
            return 0.0 if return_float else 0

        all_stats = next((p['groups'] for p in data['statistics'] if p['period'] == 'ALL'), [])
        
        stats['corners_home_ft'] = extract_val(all_stats, ['corner kicks'], True)
        stats['corners_away_ft'] = extract_val(all_stats, ['corner kicks'], False)
        stats['shots_ot_home_ft'] = extract_val(all_stats, ['shots on target'], True)
        stats['shots_ot_away_ft'] = extract_val(all_stats, ['shots on target'], False)
        stats['total_shots_home'] = extract_val(all_stats, ['total shots'], True)
        stats['total_shots_away'] = extract_val(all_stats, ['total shots'], False)
        stats['possession_home'] = extract_val(all_stats, ['ball possession'], True)
        stats['possession_away'] = extract_val(all_stats, ['ball possession'], False)
        stats['fouls_home'] = extract_val(all_stats, ['fouls'], True)
        stats['fouls_away'] = extract_val(all_stats, ['fouls'], False)
        stats['yellow_cards_home'] = extract_val(all_stats, ['yellow cards'], True)
        stats['yellow_cards_away'] = extract_val(all_stats, ['yellow cards'], False)
        stats['red_cards_home'] = extract_val(all_stats, ['red cards'], True)
        stats['red_cards_away'] = extract_val(all_stats, ['red cards'], False)
        stats['big_chances_home'] = extract_val(all_stats, ['big chances'], True)
        stats['big_chances_away'] = extract_val(all_stats, ['big chances'], False)
        stats['expected_goals_home'] = extract_val(all_stats, ['expected goals'], True, return_float=True)
        stats['expected_goals_away'] = extract_val(all_stats, ['expected goals'], False, return_float=True)

        ht_stats = next((p['groups'] for p in data['statistics'] if p['period'] == '1ST'), [])
        stats['corners_home_ht'] = extract_val(ht_stats, ['corner kicks'], True)
        stats['corners_away_ht'] = extract_val(ht_stats, ['corner kicks'], False)
        stats['shots_ot_home_ht'] = extract_val(ht_stats, ['shots on target'], True)
        stats['shots_ot_away_ht'] = extract_val(ht_stats, ['shots on target'], False)

        # Dangerous Attacks (New Quant Feature) - Tenta em ALL e soma partes se necessário
        da_home = extract_val(all_stats, ['dangerous attacks', 'ataques perigosos', 'dangerous attack'], True)
        da_away = extract_val(all_stats, ['dangerous attacks', 'ataques perigosos', 'dangerous attack'], False)
        
        if da_home == 0:
            # Tenta somar 1T + 2T se estiver faltando no ALL
            at1_h = extract_val(ht_stats, ['dangerous attacks', 'ataques perigosos'], True)
            at1_a = extract_val(ht_stats, ['dangerous attacks', 'ataques perigosos'], False)
            da_home = at1_h
            da_away = at1_a
        
        # Explicitly fetch Momentum Graph (Regra #38)
        # We fetch it directly instead of relying on passive interception to ensure we have data.
        graph_url = f"https://www.sofascore.com/api/v1/event/{match_id}/graph"
        graph_data = self._fetch_api(graph_url)
        if graph_data and 'graphPoints' in graph_data:
            self.last_momentum_data = graph_data['graphPoints']
        
        # Fallback: Use Momentum Graph Data for DA if stats are zero
        if (da_home == 0 and da_away == 0) and self.last_momentum_data:
            if self.verbose: print("⚠️ Stats zeradas. Usando dados do Gráfico de Momentum...")
            # Calculate pressure minutes from graph
            # Value > 0: Home Pressure, Value < 0: Away Pressure
            da_home = sum(1 for p in self.last_momentum_data if p.get('value', 0) > 0)
            da_away = sum(1 for p in self.last_momentum_data if p.get('value', 0) < 0)
        
        stats['dangerous_attacks_home'] = da_home
        stats['dangerous_attacks_away'] = da_away
        
        # Tactical Metrics (Gap Analysis)
        stats['blocked_shots_home'] = extract_val(all_stats, ['blocked shots', 'chutes travados'], True)
        stats['blocked_shots_away'] = extract_val(all_stats, ['blocked shots', 'chutes travados'], False)
        
        # Crosses: Extract TOTAL attempted crosses (denominator) for volume/pressure analysis
        stats['crosses_home'] = extract_val(all_stats, ['crosses', 'cruzamentos'], True, extract_total=True)
        stats['crosses_away'] = extract_val(all_stats, ['crosses', 'cruzamentos'], False, extract_total=True)
        
        stats['tackles_home'] = extract_val(all_stats, ['tackles', 'desarmes', 'total tackles'], True)
        stats['tackles_away'] = extract_val(all_stats, ['tackles', 'desarmes', 'total tackles'], False)
        
        stats['interceptions_home'] = extract_val(all_stats, ['interceptions', 'interceptações'], True)
        stats['interceptions_away'] = extract_val(all_stats, ['interceptions', 'interceptações'], False)
        
        stats['clearances_home'] = extract_val(all_stats, ['clearances', 'cortes'], True)
        stats['clearances_away'] = extract_val(all_stats, ['clearances', 'cortes'], False)
        
        stats['recoveries_home'] = extract_val(all_stats, ['recoveries', 'bolas recuperadas'], True)
        stats['recoveries_away'] = extract_val(all_stats, ['recoveries', 'bolas recuperadas'], False)

        # Attack Momentum Processing (Regra #38: PhD Metrics)
        if self.last_momentum_data:
            mom_stats = self._process_momentum(self.last_momentum_data)
            stats.update(mom_stats)
            if self.verbose and (mom_stats['momentum_home'] > 0 or mom_stats['momentum_away'] > 0):
                print(f"   📈 Momentum Processado: Home {mom_stats['momentum_home']:.1f} | Away {mom_stats['momentum_away']:.1f}")

        return stats

    def _process_momentum(self, graph_points: list) -> dict:
        """
        Calcula métricas avançadas baseadas no gráfico de Attack Momentum.
        
        Regra de Negócio:
            O gráfico de momentum é uma série temporal onde valores > 0 indicam pressão do Mandante
            e valores < 0 indicam pressão do Visitante.
            Calculamos a "Área Sob a Curva" (Soma Absoluta) como proxy de volume de jogo.
            
        Returns:
            dict: {momentum_home, momentum_away, momentum_peak_home, momentum_peak_away}
        """
        momentum_home = 0.0
        momentum_away = 0.0
        peak_home = 0
        peak_away = 0
        
        for point in graph_points:
            val = point.get('value', 0)
            
            # Home Pressure (> 0)
            if val > 0:
                momentum_home += val
                if val > peak_home:
                    peak_home = val
                    
            # Away Pressure (< 0)
            elif val < 0:
                abs_val = abs(val)
                momentum_away += abs_val
                if abs_val > peak_away:
                    peak_away = abs_val
                    
        return {
            'momentum_home': round(momentum_home, 2),
            'momentum_away': round(momentum_away, 2),
            'momentum_peak_home': int(peak_home),
            'momentum_peak_away': int(peak_away)
        }

    def get_match_details(self, match_id: int) -> dict:
        """Busca detalhes completos de uma partida incluindo minuto do jogo."""
        url = f"https://www.sofascore.com/api/v1/event/{match_id}"
        data = self._fetch_api(url)
        
        if not data or 'event' not in data:
            return None
            
        ev = data['event']
        
        # Extract match minute for live matches
        status_info = ev.get('status', {})
        match_minute = None
        status_description = status_info.get('description', '')
        
        # Get minute from status description (e.g., "45+2", "HT", "78")
        if status_info.get('type') == 'inprogress':
            match_minute = status_description
            
            # 🚀 LIVE MINUTE CALCULATION (Fix for generic "1st half" / "2nd half")
            # If description is just the period name, we calculate the minute manually.
            if status_description in ["1st half", "2nd half", "1º Tempo", "2º Tempo"] or not status_description:
                try:
                    current_period_start = ev.get('currentPeriodStartTimestamp') or ev.get('time', {}).get('currentPeriodStartTimestamp')
                    
                    if current_period_start:
                        import time as t_module
                        now = int(t_module.time())
                        diff_seconds = now - current_period_start
                        minute_calc = int(diff_seconds / 60)
                        
                        if "2nd" in status_description or "2º" in status_description:
                            minute_calc += 45
                            
                        # Format as "35'"
                        match_minute = f"{minute_calc}'"
                        
                        # Check for injury time (added to display?)
                        # Usually SofaScore updates description to "45+2" automatically, but if API fails:
                        # We stick to calculated minute.
                except Exception as e:
                    # Fallback to generic description
                    # print(f"DEBUG Error calc minute: {e}")
                    pass
            
            # REMOVED: if not match_minute: match_minute = "Live"
            # We return None so DB protection can preserve existing value.
        
        details = {
            'id': ev['id'],
            'tournament': ev.get('tournament', {}).get('name', 'Unknown'),
            'tournament_id': ev.get('tournament', {}).get('uniqueTournament', {}).get('id', 0),
            'season_id': ev.get('season', {}).get('id', 0),
            'round': ev.get('roundInfo', {}).get('round', 0),
            'status': status_info.get('type', 'unknown'),
            'status_description': status_description,
            'match_minute': match_minute,
            'timestamp': ev.get('startTimestamp', 0),
            'home_id': ev['homeTeam']['id'],
            'home_name': ev['homeTeam']['name'],
            'away_id': ev['awayTeam']['id'],
            'away_name': ev['awayTeam']['name'],
            'home_score': ev.get('homeScore', {}).get('display', 0),
            'away_score': ev.get('awayScore', {}).get('display', 0),
            'home_position': ev.get('homeTeam', {}).get('ranking', None),
            'away_position': ev.get('awayTeam', {}).get('ranking', None),
            'odds_home': 0.0,
            'odds_draw': 0.0,
            'odds_away': 0.0
        }
        
        # --- ZERO-COST ODDS SCRAPER (Improved) ---
        # Tenta extrair odds do campo 'winningOdds' ou equivalente
        if 'winningOdds' in data:
             wo = data['winningOdds'] # {home: X, draw: Y, away: Z}
             # Às vezes vem como lista ou dict. 
             # Estrutura esperada: [{'choice': 'home', 'fractionalValue': ..., 'decimalValue': ...}, ...]
             # Mas a API pública simplificada pode variar. Vamos verificar 'vote' stats se winOdds falhar.
             pass
             
        # Tenta buscar odds padrão do evento (API v1)
        # Url dedicada de odds pode ser necessária, mas tente extraction direta
        # Estrutura típica: ev['customId'] é chave para odds provider?
        # Vamos tentar um endpoint específico de odds SE não tiver no main.
        # "https://www.sofascore.com/api/v1/event/{id}/odds/1/all" (1=Decimal)
        
        try:
             odds_url = f"https://www.sofascore.com/api/v1/event/{match_id}/odds/1/all"
             odds_data = self._fetch_api(odds_url)
             if odds_data and 'markets' in odds_data:
                  # Procura mercado Winner (Full Time) e Corners
                  for market in odds_data['markets']:
                       m_name = market.get('marketName', '')
                       
                       if m_name == 'Full time':
                            for choice in market.get('choices', []):
                                 if choice['name'] == '1':
                                      details['odds_home'] = float(choice['fractionalValue'].split('/')[0])/float(choice['fractionalValue'].split('/')[1]) + 1
                                 elif choice['name'] == 'X':
                                      details['odds_draw'] = float(choice['fractionalValue'].split('/')[0])/float(choice['fractionalValue'].split('/')[1]) + 1
                                 elif choice['name'] == '2':
                                      details['odds_away'] = float(choice['fractionalValue'].split('/')[0])/float(choice['fractionalValue'].split('/')[1]) + 1
                       
                       # Corners (Total)
                       # A API geralmente chama de "Total corners" ou "Corners"
                       elif 'corners' in m_name.lower() or 'escanteios' in m_name.lower():
                            if 'corner_odds' not in details:
                                details['corner_odds'] = {}
                            
                            # Logica para Over/Under
                            # Choices geralmente é [{'name': 'Over 9.5', ...}, {'name': 'Under 9.5', ...}]
                            for choice in market.get('choices', []):
                                 c_name = choice['name'] # Ex: "Over 9.5"
                                 try:
                                      odd_val = float(choice['fractionalValue'].split('/')[0])/float(choice['fractionalValue'].split('/')[1]) + 1
                                      details['corner_odds'][c_name] = odd_val
                                 except:
                                      pass
        except Exception:
             pass # Falha silenciosa se não tiver odds (pré-live distante)

        # 📊 FETCH MATCH STATISTICS (Unified Call)
        # Instead of partial reimplementation, we call the specialized method
        try:
            full_stats = self.get_match_stats(match_id)
            details['stats'] = full_stats
        except Exception as e:
            if self.verbose: print(f"   ⚠️ Error fetching extended stats: {e}")
            details['stats'] = {}

        return details

    def get_standings(self, tournament_id: int, season_id: int) -> dict:
        """Busca a tabela de classificação (Total). Retorna ditado {team_id: position}."""
        # Type 'total' is usually the main table
        url = f"https://www.sofascore.com/api/v1/unique-tournament/{tournament_id}/season/{season_id}/standings/total"
        if self.verbose: print(f"   📊 Buscando classificação (Torneio {tournament_id})...")
        data = self._fetch_api(url)
        
        standings_map = {}
        if data and 'standings' in data:
            for group in data['standings']:
                if group.get('type') == 'total':
                    for row in group.get('rows', []):
                        team_id = row['team']['id']
                        standings_map[team_id] = {
                            'position': row['position'],
                            'matches': row['matches'],
                            'wins': row['wins'],
                            'draws': row['draws'],
                            'losses': row['losses'],
                            'scoresFor': row['scoresFor'],
                            'scoresAgainst': row['scoresAgainst']
                        }
        return standings_map

    def get_team_last_games(self, team_id: int, limit: int = 5) -> list:
        """Busca últimos jogos de um time."""
        url = f"https://www.sofascore.com/api/v1/team/{team_id}/events/last/0" # Page 0
        data = self._fetch_api(url)
        
        last_games = []
        if data and 'events' in data:
            # Filter distinct tournaments if needed, but for form we take all
            # Take only finished games
            finished = [e for e in data['events'] if e['status']['type'] == 'finished']
            last_games = finished[:limit]
            
        return last_games