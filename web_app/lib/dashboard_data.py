"""
API Bridge for Next.js Dashboard
Connects to real SQLite database and returns predictions with AI reasoning.
"""

import sys
import os
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.database.db_manager import DBManager

class DashboardDataProvider:
    def __init__(self):
        self.db = DBManager()
        # [OPTIMIZATION] Removed Heavy Models (Predictor/Neural) from Dashboard.
        # The Dashboard is now a READ-ONLY viewer. 
        # All calculations must be done by the Scanner/Engine beforehand.

    def _calculate_fair_odd(self, neural_lambda, line, market_type='Over'):
        """
        Calculate Fair Odd based on Neural Expectation (Lambda).
        Formula: Odd = 1 / Probability
        """
        if neural_lambda is None or neural_lambda <= 0:
            return 0.0
            
        try:
            # Poisson Probability
            if market_type == 'Over':
                # Prob(X > line) = 1 - CDF(line)
                # Note: 'Over 10.5' means X >= 11. CDF(10.5) covers 0..10.
                prob = 1 - poisson.cdf(line, neural_lambda)
            else: # Under
                # Prob(X < line) = Prob(X <= line - epsilon) ??
                # Usually Under 10.5 means X <= 10. CDF(10.5) covers 0..10.
                prob = poisson.cdf(line, neural_lambda)
                
            if prob <= 0.01: return 100.0 # Cap high odds
            return round(1.0 / prob, 2)
        except:
            return 0.0
    
    def get_predictions_with_reasoning(self, date_str='today', league='all', status='all', top7_only=False, sort_by='confidence'):
        """
        Get predictions from database with full AI reasoning.
        
        Returns:
            list: Predictions with match data and AI reasoning
        """
        # Parse date
        # Parse date
        if date_str == 'today':
            # Force UTC-3 calculation (Server might be UTC)
            # 01:00 UTC (next day) should be 22:00 UTC-3 (current day)
            now_br = datetime.now() - timedelta(hours=3)
            target_date = now_br.strftime('%Y-%m-%d')
        elif date_str == 'tomorrow':
            now_br = datetime.now() - timedelta(hours=3)
            target_date = (now_br + timedelta(days=1)).strftime('%Y-%m-%d')
        else:
            target_date = date_str
        
        conn = self.db.connect()
        cursor = conn.cursor()
        
        # Build query
        query = """
            SELECT 
                p.id as pred_id,
                p.match_id,
                p.prediction_value,
                p.prediction_label,
                p.confidence,
                p.market_group,
                p.is_correct,
                p.status as pred_status,
                m.home_team_name,
                m.away_team_name,
                m.start_timestamp,
                m.status as match_status,
                m.tournament_name,
                m.home_score,
                m.away_score,
                s.corners_home_ft,
                s.corners_away_ft,
                s.corners_home_ht,
                s.corners_away_ht,
                m.home_league_position,
                m.away_league_position,
                m.match_minute,
                s.possession_home,
                s.possession_away,
                s.dangerous_attacks_home,
                s.dangerous_attacks_away,
                s.expected_goals_home,
                s.expected_goals_away,
                s.total_shots_home,
                s.total_shots_away,
                p.fair_odds,
                m.home_team_id,
                m.away_team_id,
                p.category
            FROM predictions p
            JOIN matches m ON p.match_id = m.match_id
            LEFT JOIN match_stats s ON m.match_id = s.match_id
            WHERE DATE(datetime(m.start_timestamp, 'unixepoch', '-3 hours')) = ?
        """
        
        params = [target_date]
        
        # Apply filters
        if league != 'all':
            league_map = {
                'premier_league': 'Premier League',
                'la_liga': 'LaLiga',
                'serie_a': 'Serie A',
                'bundesliga': 'Bundesliga',
                'ligue1': 'Ligue 1',
                'brasileirao_a': 'Brasileirão Série A',
                'brasileirao_b': 'Brasileirão Série B'
            }
            mapped_league = league_map.get(league, league)
            query += " AND m.tournament_name LIKE ?"
            params.append(f"%{mapped_league}%")
        
        if status != 'all':
            status_map = {
                'scheduled': 'notstarted',
                'live': 'inprogress', 
                'finished': 'finished'
            }
            query += " AND m.status = ?"
            params.append(status_map.get(status, status))
        
        # Sort
        if sort_by == 'confidence':
            query += " ORDER BY p.confidence DESC"
        elif sort_by == 'time':
            query += " ORDER BY m.start_timestamp ASC"
        elif sort_by == 'league':
            query += " ORDER BY m.tournament_name, p.confidence DESC"
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        
        # Group by match
        matches_dict = {}
        for row in rows:
            # Skip non-betting items
            if row[3] and ('Tactical Analysis' in row[3] or 'Data' in row[3]):
                continue
                
            match_id = int(row[1])
            if match_id not in matches_dict:
                # Get AI reasoning for this match
                reasoning = self._get_ai_reasoning(match_id, row[8], row[9])
                
                # Calculate live stats
                live_stats = None
                if row[11] in ['inprogress', 'live', 'finished']: # status
                    live_stats = {
                        'homeScore': row[13] if row[13] is not None else 0,
                        'awayScore': row[14] if row[14] is not None else 0,
                        'homeCorners': row[15] if row[15] is not None else 0,
                        'awayCorners': row[16] if row[16] is not None else 0,
                        'homeCornersHT': row[17] if row[17] is not None else 0,
                        'awayCornersHT': row[18] if row[18] is not None else 0,
                        'totalCorners': (row[15] or 0) + (row[16] or 0),
                        'possessionHome': row[22] if row[22] is not None else 0,
                        'possessionAway': row[23] if row[23] is not None else 0,
                        'attacksHome': row[24] if row[24] is not None else 0,
                        'attacksAway': row[25] if row[25] is not None else 0,
                        'xgHome': row[26] if row[26] is not None else 0.0,
                        'xgAway': row[27] if row[27] is not None else 0.0,
                        'shotsHome': row[28] if row[28] is not None else 0,
                        'shotsAway': row[29] if row[29] is not None else 0
                    }

                matches_dict[match_id] = {
                    'id': str(match_id),
                    'homeTeam': row[8], 
                    'awayTeam': row[9],
                    'homeTeamId': row[31] if len(row) > 31 else None, # Added ID
                    'awayTeamId': row[32] if len(row) > 32 else None, # Added ID
                    'awayScore': row[14] if row[14] is not None else 0,
                    'kickoff': datetime.fromtimestamp(row[10]).strftime('%H:%M'), 
                    'status': row[11] or 'notstarted',
                    'matchMinute': row[21],
                    'league': row[12],
                    'homePosition': row[19] if len(row) > 19 else None,
                    'awayPosition': row[20] if len(row) > 20 else None,
                    'predictions': [],
                    'aiReasoning': reasoning,
                    'liveStats': live_stats
                }
            
            # Add prediction
            confidence = row[4] if row[4] is not None else 0.0
            fair_odd = float(row[30]) if len(row) > 30 and row[30] is not None else 0.0
            # row[33] = p.category (added in Fix B2)
            pred_category = row[33] if len(row) > 33 and row[33] is not None else 'Alternative'
            
            # Fallback calculation if DB fair_odd is 0 (legacy data)
            if fair_odd == 0 and confidence > 0:
                 if confidence > 1.0:
                     fair_odd = 100.0 / confidence
                 else:
                     fair_odd = 1.0 / confidence

            matches_dict[match_id]['predictions'].append({
                'type': row[3],  # prediction_label
                'line': self._extract_line(row[3]),
                'prediction': float(row[2]) if row[2] is not None else 0.0,
                'confidence': float(confidence),
                'fairOdd': round(fair_odd, 2),
                'marketGroup': row[5],
                'isCorrect': bool(row[6]) if row[6] is not None else None,
                'status': row[7],
                'category': pred_category  # Fix B2: expose category to frontend
            })
        
        # Convert to list and select main bet for each match
        matches = []
        for match_data in matches_dict.values():
            # Sort predictions by confidence
            match_data['predictions'].sort(key=lambda x: x['confidence'], reverse=True)
            
            # Set main bet (highest confidence)
            if match_data['predictions']:
                # Deduplicate predictions by type (keep highest confidence for each type)
                seen = {}
                deduped_predictions = []
                for pred in match_data['predictions']:
                    pred_key = pred['type']
                    if pred_key not in seen:
                        seen[pred_key] = True
                        deduped_predictions.append(pred)
                
                match_data['predictions'] = deduped_predictions
                
                # Fix B2: mainBet must come from Main/Top7 category.
                # Alternative Market picks (e.g. 'Over 3.5') have high Poisson probability
                # by design (low line) — they should NOT override the real Main analysis.
                main_picks = [p for p in deduped_predictions
                              if p.get('category') in ('Main', 'Top7')]
                alt_picks = [p for p in deduped_predictions
                             if p.get('category') not in ('Main', 'Top7')]
                
                # Fallback: if no explicit Main saved yet, use highest confidence overall
                best_for_main = main_picks if main_picks else deduped_predictions
                match_data['mainBet'] = best_for_main[0]
                match_data['alternativeBets'] = alt_picks[:8]
                
                # --- NEURAL VALIDATOR ENRICHMENT ---
                # [OPTIMIZATION] Logic removed. Dashboard now relies on Pre-calculated metadata.
                # TODO: Parse 'feedback_text' or 'neural_params' from DB if available.
                pass
            
            matches.append(match_data)

            # --- GENERAL PREDICTION (Total Corners) ---
            # Search for the "Global Expected Value" (Lambda) for Total Corners
            total_corners_pred = 0.0
            found_general = False
            
            # Strategy 1: Look for explicit 'Corners' market group (usually contains Full Match Total lines)
            # The 'prediction' field contains the underlying expected value (e.g., 9.57) regardless of the line (Over 10.5 / Under 10.5)
            main_corners_preds = [p for p in match_data['predictions'] if p['marketGroup'] == 'Corners']
            if main_corners_preds:
                total_corners_pred = main_corners_preds[0]['prediction']
                found_general = True
                
            # Strategy 2: If no explicit 'Corners' group, look for "Total" bets that are NOT team specific
            if not found_general:
                # Filter for types that look like "Over X", "Total Over X" but exclude "Home", "Away", "Vis", "Casa", "1T", "2T"
                candidates = []
                for p in match_data['predictions']:
                    t = p['type'].lower()
                    # Exclude team/half specifics
                    if any(x in t for x in ['home', 'away', 'vis', 'casa', '1t', '2t', 'ht', 'st']):
                        continue
                    # Must have "over" or "total" or "mais"
                    if any(x in t for x in ['over', 'total', 'mais']):
                        candidates.append(p)
                
                if candidates:
                    total_corners_pred = candidates[0]['prediction']
                    found_general = True

            # Strategy 3: Fallback to Sum of Averages
            if not found_general and match_data['aiReasoning']:
                try:
                    h_avg = match_data['aiReasoning']['recentForm']['homeOverall']['avg']
                    a_avg = match_data['aiReasoning']['recentForm']['awayOverall']['avg']
                    total_corners_pred = h_avg + a_avg
                except:
                    pass
            
            match_data['generalPrediction'] = round(total_corners_pred, 1)
        
        # Filter Top 7 if requested
        if top7_only:
            # Get all predictions across all matches, sort, take top 7
            all_predictions = []
            for match in matches:
                for pred in match['predictions']:
                    all_predictions.append({
                        'match': match,
                        'prediction': pred,
                        'confidence': pred['confidence']
                    })
            
            all_predictions.sort(key=lambda x: x['confidence'], reverse=True)
            top7_matches = list({p['match']['id']: p['match'] for p in all_predictions[:7]}.values())
            matches = top7_matches
        
        return matches
    
    def _get_ai_reasoning(self, match_id, home_team, away_team):
        """Get AI reasoning data for a match"""
        conn = self.db.connect()
        cursor = conn.cursor()
        
        # Get recent form for both teams (Specific and Overall)
        home_recent = self._get_recent_form(cursor, home_team, 'home')
        away_recent = self._get_recent_form(cursor, away_team, 'away')
        home_overall = self._get_recent_form(cursor, home_team, 'all')
        away_overall = self._get_recent_form(cursor, away_team, 'all')
        
        # Get H2H
        h2h = self._get_h2h(cursor, home_team, away_team)
        
        # Feature importance (simplified - would need SHAP/feature importance from model)
        feature_importance = [
            {'name': 'Home Recent Form', 'value': 82},
            {'name': 'Away Dangerous Attacks', 'value': 71},
            {'name': 'Home Momentum', 'value': 65},
            {'name': 'H2H History', 'value': 48}
        ]
        
        # Risk factors
        risk_factors = [
            {'type': 'Weather', 'description': 'Clear', 'status': 'good'},
            {'type': 'Match Importance', 'description': 'Regular season', 'status': 'good'}
        ]
        
        return {
            'recentForm': {
                'homeSpecific': home_recent,
                'awaySpecific': away_recent,
                'homeOverall': home_overall,
                'awayOverall': away_overall
            },
            'h2h': h2h,
            'featureImportance': feature_importance,
            'riskFactors': risk_factors
        }
    
    def _get_recent_form(self, cursor, team_name, home_or_away):
        """Get recent form for a team.

        Important business rule:
        - `home` and `away` forms are derived from the same *recent timeline* as `all`,
          instead of pulling independent "last 5 home" or "last 5 away" across long history.
          This keeps values consistent with the visible recent form context.
        """
        query = """
            SELECT s.corners_home_ft, s.corners_away_ft, m.home_team_name,
                   s.corners_home_ht, s.corners_away_ht
            FROM matches m
            JOIN match_stats s ON m.match_id = s.match_id
            WHERE (m.home_team_name = ? OR m.away_team_name = ?)
              AND m.status = 'finished'
            ORDER BY m.start_timestamp DESC
            LIMIT 15
        """
        cursor.execute(query, [team_name, team_name])
        all_recent_rows = cursor.fetchall()

        if home_or_away == 'all':
            rows = all_recent_rows[:5]
        elif home_or_away == 'home':
            rows = [r for r in all_recent_rows if r[2] == team_name][:5]
        elif home_or_away == 'away':
            rows = [r for r in all_recent_rows if r[2] != team_name][:5]
        else:
            rows = []
        
        games_ft = []  # Full time
        games_ht = []  # Half time
        games_st = []  # Second half (FT - HT)
        
        for row in rows:
            is_home_team = (row[2] == team_name)
            
            ft = row[0] if is_home_team else row[1]
            ht = row[3] if is_home_team else row[4]
            
            # Handle possible NULLs from DB
            ft = ft if ft is not None else 0
            ht = ht if ht is not None else 0
            
            st = ft - ht
            
            games_ft.append(ft)
            games_ht.append(ht)
            games_st.append(st)
        
        # Calculate averages
        avg_ft = sum(games_ft) / len(games_ft) if games_ft else 0
        avg_ht = sum(games_ht) / len(games_ht) if games_ht else 0
        avg_st = sum(games_st) / len(games_st) if games_st else 0
        
        # Calculate variance (standard deviation)
        import statistics
        std_ft = statistics.stdev(games_ft) if len(games_ft) > 1 else 0
        std_ht = statistics.stdev(games_ht) if len(games_ht) > 1 else 0
        std_st = statistics.stdev(games_st) if len(games_st) > 1 else 0
        
        # Simple trend (linear regression slope would be better, but simple diff is okay for now)
        trend = 0
        if len(games_ft) >= 2:
            trend = games_ft[0] - games_ft[-1] # Positive means improving recent form (more corners)

        return {
            'avg': round(avg_ft, 1),
            'std': round(std_ft, 1),
            'trend': trend,
            'games': games_ft,
            'ht': {
                'avg': round(avg_ht, 1),
                'std': round(std_ht, 1),
                'games': games_ht
            },
            'st': {
                'avg': round(avg_st, 1),
                'std': round(std_st, 1),
                'games': games_st
            }
        }
    
    def _get_h2h(self, cursor, home_team, away_team):
        """Get head-to-head history"""
        query = """
            SELECT s.corners_home_ft, s.corners_away_ft
            FROM matches m
            JOIN match_stats s ON m.match_id = s.match_id
            WHERE ((m.home_team_name = ? AND m.away_team_name = ?)
               OR (m.home_team_name = ? AND m.away_team_name = ?))
              AND m.status = 'finished'
            ORDER BY m.start_timestamp DESC
            LIMIT 3
        """
        
        cursor.execute(query, [home_team, away_team, away_team, home_team])
        rows = cursor.fetchall()
        
        games = [row[0] + row[1] for row in rows]
        avg = sum(games) / len(games) if games else 0
        
        return {
            'avg': round(avg, 1),
            'games': games
        }
    
    def _extract_line(self, prediction_label):
        """Extract line from prediction label (e.g., 'Over 11.5' -> 11.5)"""
        import re
        match = re.search(r'(\d+\.?\d*)', prediction_label)
        return float(match.group(1)) if match else 0.0

    def get_ai_stats(self):
        """Get overall AI performance statistics"""
        stats = self.db.get_win_rate_stats()
        
        # Get historical win rate for chart
        history = self._get_win_rate_history()
        
        return {
            'overallWinRate': round(stats['win_rate'] * 100, 1),
            'top7WinRate': round(stats['win_rate_top7'] * 100, 1),
            'totalPredictions': stats['total'],
            'correctPredictions': stats['correct'],
            'pendingPredictions': stats['pending'],
            'metrics': {
                'rps': 0.075, # Mock for now, would be calculated from model
                'mae': 2.68,
                'ece': 0.15
            },
            'history': history,
            'marketPerformance': [
                {'market': 'Total Corners', 'winRate': 72.5, 'total': 45},
                {'market': 'Asian Corners', 'winRate': 78.2, 'total': 38},
                {'market': '1st Half Corners', 'winRate': 81.4, 'total': 22}
            ]
        }

    def get_bankroll_stats(self):
        """Get real user bankroll and bets tracking"""
        # Note: This would normally come from a 'user_bets' table
        # For now, we simulate with data that would be in the DB
        return {
            'balance': 1127.50,
            'startingBalance': 1000.00,
            'netProfit': 127.50,
            'roi': 12.7,
            'recentBets': [
                {
                    'id': 'b1',
                    'matchName': 'Bournemouth vs Tottenham',
                    'prediction': 'Over 10.5',
                    'stake': 50.0,
                    'odd': 1.85,
                    'status': 'win',
                    'profit': 42.50,
                    'date': '2026-01-07'
                }
            ],
            'performance': {
                'totalBets': 34,
                'wins': 21,
                'losses': 13,
                'avgStake': 37.20,
                'followingAIWinRate': 67.8,
                'customWinRate': 33.3
            }
        }

    def _get_win_rate_history(self):
        """Calculate win rate over time (last 30 days)"""
        # Simplified simulation of historical data
        return [
            {'date': '2025-12-15', 'overall': 65, 'top7': 72},
            {'date': '2025-12-22', 'overall': 68, 'top7': 75},
            {'date': '2025-12-29', 'overall': 67, 'top7': 78},
            {'date': '2026-01-05', 'overall': 70, 'top7': 82},
        ]
        
    def get_system_status(self):
        """Get system status including last update time"""
        conn = self.db.connect()
        cursor = conn.cursor()
        
        try:
            # Get latest update time from matches table
            cursor.execute("SELECT MAX(last_updated) FROM matches")
            last_updated = cursor.fetchone()[0]
            
            # Correction: Convert UTC (from DB) to UTC-3 (Brazil) for display
            if last_updated:
                try:
                    # Assuming format 'YYYY-MM-DD HH:MM:SS' or similiar
                    # It might be returning a string
                    dt = datetime.strptime(str(last_updated), '%Y-%m-%d %H:%M:%S.%f')
                except ValueError:
                    try:
                        dt = datetime.strptime(str(last_updated), '%Y-%m-%d %H:%M:%S')
                    except:
                        dt = datetime.now() # Fallback
                
                # Subtract 3 hours
                dt_br = dt - timedelta(hours=3)
                last_updated = dt_br.strftime('%Y-%m-%d %H:%M:%S')
            
            # Count live matches
            cursor.execute("SELECT COUNT(*) FROM matches WHERE status='inprogress'")
            live_count = cursor.fetchone()[0]
            
            return {
                'status': 'online',
                'last_updated': last_updated,
                'live_matches': live_count,
                'system_time': (datetime.now() - timedelta(hours=3)).strftime('%Y-%m-%d %H:%M:%S')
            }
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e)
            }

# Singleton instance
_provider = None

def get_dashboard_data(type='predictions', **kwargs):
    """Get dashboard data - use this from API routes"""
    global _provider
    if _provider is None:
        _provider = DashboardDataProvider()
    
    if type == 'stats':
        return _provider.get_ai_stats()
    elif type == 'bankroll':
        return _provider.get_bankroll_stats()
    elif type == 'system-status':
        return _provider.get_system_status()
    else:
        return _provider.get_predictions_with_reasoning(
            kwargs.get('date', 'today'),
            kwargs.get('league', 'all'),
            kwargs.get('status', 'all'),
            kwargs.get('top7_only', False),
            kwargs.get('sort_by', 'confidence')
        )
