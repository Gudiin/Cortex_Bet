"""
Scanner Rápido - Apenas Coleta de Dados (Sem IA)

Uso:
    python scripts/quick_scan.py [--loop] [--interval SECONDS]
    
Opções:
    --loop          Executa em loop contínuo
    --interval N    Intervalo entre scans em segundos (padrão: 60)
    
Função:
    - Busca jogos agendados
    - Enriquece com posições (standings)
    - Salva no banco
    - NÃO faz predições (economiza tempo)
"""
import sys
import os
import time
import argparse
import subprocess

# FIX: Force UTF-8 encoding for Windows Consoles to support emojis
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except AttributeError:
        pass # Older python versions might not need/support this

from datetime import datetime
from pathlib import Path
from datetime import datetime, timedelta

# Fix for Windows Unicode output (emojis)
sys.stdout.reconfigure(encoding='utf-8')


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.database.db_manager import DBManager
from src.scrapers.sofascore import SofaScoreScraper
from src.analysis.prediction_validator import PredictionValidator
from src.analysis.bet_resolver import resolve_pending_bets
from scripts.cleanup_canceled import cleanup_canceled_matches

LAST_TRAIN_MARKER = Path("data/last_auto_train.txt")
TRAIN_SCRIPT = Path("scripts/train_model.py")


def maybe_trigger_auto_training() -> None:
    """Trigger background training if 15 days have passed since last automatic run."""
    now = datetime.now()
    should_train = True
    if LAST_TRAIN_MARKER.exists():
        try:
            last_train = datetime.fromisoformat(LAST_TRAIN_MARKER.read_text(encoding="utf-8").strip())
            should_train = (now - last_train).days >= 15
        except Exception:
            should_train = True

    if not should_train:
        return

    try:
        print("🧠 Auto-training: mais de 15 dias desde o último treino automático. Iniciando...")
        subprocess.Popen([sys.executable, str(TRAIN_SCRIPT)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        LAST_TRAIN_MARKER.parent.mkdir(parents=True, exist_ok=True)
        LAST_TRAIN_MARKER.write_text(now.isoformat(), encoding="utf-8")
    except Exception as exc:
        print(f"⚠️ Falha ao iniciar auto-training: {exc}")

def run_scan(db, scraper, loop_mode=False):
    try:
        # Get today's matches
        target_date = datetime.now().strftime('%Y-%m-%d')
        current_time = datetime.now().strftime('%H:%M:%S')
        
        print(f"[{current_time}] 📅 Scanning for: {target_date}")
        if not loop_mode:
            print()
        
        # Fetch scheduled matches - API returns ALL matches (scheduled, live, finished)
        TOP_LEAGUES = [325, 17, 8, 31, 35, 34, 23, 83, 390]
        matches = scraper.get_scheduled_matches(target_date, league_ids=TOP_LEAGUES)
        
        # REMOVED: Yesterday scanning logic as per user request
        
        if not matches:
            print(f"[{current_time}] ❌ No matches found")
            return
        
        if not loop_mode:
            print(f"📊 Found {len(matches)} matches")
            print()
        
        # Group by tournament for standings
        tournaments = {}
        for match in matches:
            t_id = match.get('tournament_id')
            s_id = match.get('season_id')
            if t_id and s_id and s_id != 0:  # Skip if season_id is invalid
                key = f"{t_id}_{s_id}"
                if key not in tournaments:
                    tournaments[key] = {
                        'tournament_id': t_id,
                        'season_id': s_id,
                        'name': match.get('tournament'),
                        'matches': []
                    }
                tournaments[key]['matches'].append(match)
        
        # Fetch standings (only if not running too frequently or optimize later)
        # For now, fetch every time to keep positions updated
        if not loop_mode:
            print("📊 Fetching standings...")
            
        for key, data in tournaments.items():
            try:
                standings = scraper.get_standings(data['tournament_id'], data['season_id'])
                if standings:
                    for match in data['matches']:
                        h_info = standings.get(match.get('home_id'))
                        a_info = standings.get(match.get('away_id'))
                        
                        if h_info:
                            match['home_position'] = h_info['position']
                        if a_info:
                            match['away_position'] = a_info['position']
                    
                    if not loop_mode:
                        print(f"   ✅ {data['name']}: {len(standings)} teams")
            except Exception as e:
                # Silently fail in loop mode to reduce noise
                if not loop_mode:
                    print(f"   ⚠️ Failed for {data['name']}: {e}")
        
        if not loop_mode:
            print()
            print("💾 Saving to database...")
        
        # Save all matches to DB WITH detailed statistics (no predictions)
        saved = 0
        live_count = 0
        
        for match in matches:
            try:
                match_id = match['match_id']
                
                # 🔥 FETCH DETAILED STATS (corners, shots, etc.)
                # This replaces basic data with full match details
                if not loop_mode:
                    print(f"   📊 Fetching details for {match['home_team']} vs {match['away_team']}...")
                
                details = scraper.get_match_details(match_id)
                
                if not details:
                    # Fallback to basic data if API fails
                    if not loop_mode:
                        print(f"   ⚠️ Could not fetch details, using basic data")
                    details = match
                
                # Merge standings positions into details (from earlier enrichment)
                details['home_position'] = match.get('home_position')
                details['away_position'] = match.get('away_position')
                
                match_data = {
                    'id': details.get('id') or match_id,
                    'tournament': details.get('tournament', match['tournament']),
                    'tournament_id': details.get('tournament_id', match.get('tournament_id', 0)),
                    'season_id': details.get('season_id', match.get('season_id', 0)),
                    'round': details.get('round', match.get('round', 0)),
                   'status': details.get('status', match.get('status', 'notstarted')),
                    'timestamp': details.get('timestamp', match['timestamp']),
                    'home_id': details.get('home_id', match['home_id']),
                    'home_name': details.get('home_name', match['home_team']),
                    'away_id': details.get('away_id', match['away_id']),
                    'away_name': details.get('away_name', match['away_team']),
                    'home_score': details.get('home_score', match.get('home_score', 0)),
                    'away_score': details.get('away_score', match.get('away_score', 0)),
                    'match_minute': details.get('match_minute', match.get('status_description')),
                    'home_position': details.get('home_position'),
                    'away_position': details.get('away_position'),
                    'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
                
                # DEBUG: Print status for every match to diagnose "Not Live" issue
                # print(f"DEBUG: {match['home_team']} vs {match['away_team']} | ID: {match_id} | Status: {match_data.get('status')} | Minute: {match_data.get('match_minute')}")

                
                db.save_match(match_data)
                
                # 📊 SAVE DETAILED STATISTICS (corners, shots, etc.)
                if 'stats' in details or 'statistics' in details:
                    # Extract stats and save to match_stats table
                    stats_data = details.get('stats') or details.get('statistics', {})
                    
                    # Save corners and other stats
                    db.save_stats(match_id, stats_data)
                
                saved += 1
                
                if details.get('status') == 'inprogress':
                    live_count += 1
                
                # Display in verbose mode
                if not loop_mode:
                    h_pos = f"[{match.get('home_position')}°]" if match.get('home_position') else "[--]"
                    a_pos = f"[{match.get('away_position')}°]" if match.get('away_position') else "[--]"
                    status_str = match_data.get('status', 'unknown')
                    minute_str = match_data.get('match_minute', '')
                    print(f"   ✅ {match['home_team']} {h_pos} vs {a_pos} {match['away_team']} | {status_str} {minute_str}")
                else:
                     # Shorter log for loop mode to confirm activity without spam
                     if match_data.get('status') == 'inprogress':
                         print(f"   ⚽ LIVE: {match['home_team']} vs {match['away_team']} ({match_data.get('match_minute')})")
                
            except Exception as e:
                if not loop_mode:
                    print(f"   ❌ Error saving match {match['match_id']}: {e}")
        
        if loop_mode:
            print(f"[{current_time}] ✅ Updated {saved} matches ({live_count} Live). Waiting...")
        else:
            print()
            print("=" * 80)
            print(f"✅ Quick Scan Complete!")
            print(f"   Saved: {saved}/{len(matches)} matches")
            print(f"   Time saved: ~{len(matches) * 10}s (no AI predictions)")
            print()
            print("💡 To add predictions for these matches, run:")
            print("   python scripts/run_scanner.py")
            print("=" * 80)
            
        # Validate and Settle User Bets
        try:
            if not loop_mode:
                print("💰 Resolving pending bets...")
            resolved_bets = resolve_pending_bets()
            if resolved_bets > 0 and loop_mode:
                 print(f"💰 {resolved_bets} bets settled!")
        except Exception as e:
            print(f"⚠️ Bet resolution error: {e}")

        # Auto-Cleanup of Canceled Matches
        try:
            if not loop_mode:
                print("🧹 Running auto-cleanup...")
            cleanup_canceled_matches()
        except Exception as e:
            print(f"⚠️ Cleanup error: {e}")

        return matches
            
    except Exception as e:
        print(f"Error during scan: {e}")
        return []

def main():
    parser = argparse.ArgumentParser(description='Cortex Bet Quick Scanner')
    parser.add_argument('--single', action='store_true', help='Run only once and exit (Disable Loop)')
    parser.add_argument('--interval', type=int, default=60, help='Interval between scans in seconds')
    
    args = parser.parse_args()
    
    print("CORTEX QUICK SCAN - Data Collection Only")
    print("=" * 80)
    
    # Initialize
    db = DBManager()
    scraper = SofaScoreScraper(headless=True, verbose=True)
    validator = PredictionValidator()
    scraper.start()
    
    # Default to loop mode if not specified otherwise
    # Logic: Always loop unless --single is passed
    should_loop = not args.single
    
    try:
        if should_loop:
            print(f"Loop Mode Active (Smart Interval)")
            print("=" * 80)
            while True:
                maybe_trigger_auto_training()
                matches = run_scan(db, scraper, loop_mode=True)
                
                # Validate predictions periodically
                validated = validator.validate_pending_predictions()
                if validated > 0:
                    print(f"   ✅ Graded {validated} predictions (Green/Red updated)")
                
                has_live = False
                # Smart Interval Logic
                if not matches:
                    interval = 3600 # No matches today? Sleep 1h
                else:
                    now = datetime.now()
                    has_live = any(m['status'] == 'inprogress' for m in matches)
                    
                    # Find time to next match
                    next_start = None
                    for m in matches:
                        if m['status'] == 'notstarted':
                            try:
                                ts = m.get('timestamp')
                                if ts:
                                    candidate = datetime.fromtimestamp(int(ts))
                                    if candidate >= now and (next_start is None or candidate < next_start):
                                        next_start = candidate
                            except Exception:
                                continue
                    
                    # Decide interval
                    if has_live:
                        interval = 60 # Live game? Fast update
                    else:
                        interval = args.interval 

                        upcoming = [m for m in matches if m['status'] == 'notstarted']
                        if upcoming:
                            interval = 300 # 5 mins default if no live games
                            if next_start is not None:
                                mins_to_next = (next_start - now).total_seconds() / 60
                                if mins_to_next <= 30:
                                    interval = 60
                                elif mins_to_next <= 120:
                                    interval = 180
                        
                        if all(m['status'] == 'finished' for m in matches):
                            interval = 3600 # All done? Sleep 1h
                            print(f"[Smart Loop] All matches finished. Sleeping for 1 hour.")

                if has_live:
                    print(f"[Smart Loop] 🔴 Live matches active. Next scan in 60s.")
                    interval = 60
                elif interval == 300:
                    print(f"[Smart Loop] No live matches. Next scan in 5m.")
                    
                time.sleep(interval)
        else:
            run_scan(db, scraper, loop_mode=False)
            print("   🔍 Validating predictions...")
            validated = validator.validate_pending_predictions()
            if validated > 0:
                print(f"   ✅ Graded {validated} predictions")
            
    except KeyboardInterrupt:
        print("\n🛑 Stopping scanner...")
    finally:
        scraper.stop()
        db.close()

if __name__ == "__main__":
    main()
