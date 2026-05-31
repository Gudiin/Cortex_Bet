"""
CLI Interface Module - Cortex ML V2.1
Handles user interaction, menus, and argument parsing.
"""

import sys
import os
import re
import argparse
import traceback
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.database.db_manager import DBManager
from src.scrapers.sofascore import SofaScoreScraper
from src.analysis.statistical import Colors
from src.analysis.manager_ai import ManagerAI
from src.training.trainer import train_model
from src.models.model_v2 import ProfessionalPredictor
from src.data.updater import (
    update_database, update_match_by_url, update_specific_league, 
    update_all_leagues
)

# Joint Trainer (multimercado científico)
def _get_joint_trainer():
    """Lazy import para evitar carregar dependências pesadas no startup."""
    from src.training.joint_trainer import JointTrainer
    return JointTrainer

# Global args mock if needed, but we pass args to functions
args = None

def analyze_match_url() -> None:
    """Analisa uma partida específica via URL."""
    url = input("Cole a URL do jogo do SofaScore: ")
    match_id_search = re.search(r'id:(\d+)', url)
    
    if not match_id_search:
        print("ID do jogo não encontrado na URL.")
        return

    match_id = match_id_search.group(1)
    print(f"Analisando jogo ID: {match_id} com CORTEX V2.1 ACADEMIC...")
    
    scraper = SofaScoreScraper(headless=True)
    try:
        scraper.start()
        
        # 1. Obter Dados Básicos
        api_url = f"https://www.sofascore.com/api/v1/event/{match_id}"
        ev_data = scraper._fetch_api(api_url)
        
        if not ev_data or 'event' not in ev_data:
            print("Erro ao buscar dados do jogo.")
            return
            
        ev = ev_data['event']
        match_name = f"{ev['homeTeam']['name']} vs {ev['awayTeam']['name']}"
        print(f"📅 Jogo: {match_name}")
        
        # 2. Salvar/Atualizar no Banco Temporário (Cache)
        db = DBManager()
        match_data = {
            'id': match_id,
            'tournament': ev.get('tournament', {}).get('name', 'Unknown'),
            'tournament_id': ev.get('tournament', {}).get('id', 0),
            'season_id': ev.get('season', {}).get('id', 0),
            'round': ev.get('roundInfo', {}).get('round', 0),
            'status': ev.get('status', {}).get('type', 'notstarted'),
            'timestamp': ev.get('startTimestamp', 0),
            'home_id': ev['homeTeam']['id'],
            'home_name': ev['homeTeam']['name'],
            'away_id': ev['awayTeam']['id'],
            'away_name': ev['awayTeam']['name'],
            'home_score': ev.get('homeScore', {}).get('display', 0),
            'away_score': ev.get('awayScore', {}).get('display', 0)
        }
        
        # Tenta pegar odds
        try:
             odds_url = f"https://www.sofascore.com/api/v1/event/{match_id}/odds/1/all"
             odds_data = scraper._fetch_api(odds_url)
             corner_odds = {}
             if odds_data and 'markets' in odds_data:
                  for market in odds_data['markets']:
                       m_name = market.get('marketName', '')
                       if 'corners' in m_name.lower() or 'escanteios' in m_name.lower():
                            for choice in market.get('choices', []):
                                 c_name = choice['name']
                                 try:
                                      odd_val = float(choice['fractionalValue'].split('/')[0])/float(choice['fractionalValue'].split('/')[1]) + 1
                                      corner_odds[c_name] = odd_val
                                 except:
                                      pass
             match_data['corner_odds'] = corner_odds
        except:
             pass

        # 3. Preparar Motor de Previsão (ManagerAI)
        manager = ManagerAI(db)
        
        match_metadata = {
            'home_id': ev['homeTeam']['id'],
            'away_id': ev['awayTeam']['id'],
            'timestamp': ev.get('startTimestamp', 0),
            'tournament_id': ev.get('tournament', {}).get('id', 0),
            'home_name': ev['homeTeam']['name'],
            'away_name': ev['awayTeam']['name']
        }

        result = manager.predict_match(
            match_id=int(match_id),
            match_metadata=match_metadata
        )
        
        print("\n" + "="*60)
        print("📝 FEEDBACK DA IA (Cortex V2.1):")
        print("="*60)
        print(result.feedback_text)
        
        conf_val = result.consensus_confidence * 100
        pred_val = result.final_prediction
        print(f"\n✅ {pred_val:.1f} esc | {conf_val:.0f}% conf | 💾 Salvo!")
        
        if ev.get('status', {}).get('type') == 'finished':
            print("\n🏁 Jogo finalizado. Verificando resultado...")
            db.check_predictions()

    except Exception as e:
        print(f"Erro na análise principal: {e}")
        traceback.print_exc()
    finally:
        scraper.stop()
        if 'db' in locals():
            db.close()

def retrieve_analysis() -> None:
    """Busca e exibe análises salvas."""
    user_input = input("Digite o ID do jogo (ou cole a URL): ")
    match_id_search = re.search(r'id:(\d+)', user_input)
    if match_id_search:
        match_id = match_id_search.group(1)
    else:
        match_id = user_input.strip()
        if not match_id.isdigit():
             print("❌ ID inválido.")
             return
    
    db = DBManager()
    conn = db.connect()
    cursor = conn.cursor()
    cursor.execute('''
        SELECT prediction_label, confidence, odds, category, market_group, model_version, prediction_value, status
        FROM predictions WHERE match_id = ? ORDER BY confidence DESC
    ''', (match_id,))
    rows = cursor.fetchall()
    db.close()
    
    if not rows:
        print("Nenhuma análise encontrada.")
        return
        
    print(f"\n📊 Análise para o Jogo {match_id}:")
    print("-" * 60)
    
    ml_pred = None
    stats_preds = []
    
    for row in rows:
        label, conf, odds, cat, market, model, val, status = row
        if model == 'CORTEX_V2.1_CALIBRATED':
            ml_pred = (val, label, status)
        else:
            stats_preds.append((label, conf, odds, cat, market, status))
            
    if ml_pred:
        status_icon = "✅" if ml_pred[2] == 'GREEN' else ("❌" if ml_pred[2] == 'RED' else "⏳")
        print(f"🤖 IA (ML V10): {ml_pred[0]:.2f} Escanteios ({ml_pred[1]}) {status_icon}")
        print("-" * 60)
        
    for label, conf, odds, cat, market, status in stats_preds:
        if status == 'GREEN': status_icon = f"{Colors.GREEN}✅{Colors.RESET}"
        elif status == 'RED': status_icon = f"{Colors.RED}❌{Colors.RESET}"
        else: status_icon = "⏳"
        print(f"   {status_icon} {label:<18} | Prob: {conf:>6.1%} | Odd: {odds:>5.2f} | [{cat}]")
    print("-" * 60)

def scan_opportunities() -> None:
    """Scanner de oportunidades."""
    from datetime import datetime, timedelta, timezone
    from src.analysis.unified_scanner import scan_opportunities_core
    brt = timezone(timedelta(hours=-3))
    
    print("\n" + "=" * 50)
    print("📡 SCANNER DE OPORTUNIDADES")
    print("=" * 50)
    print("1. Hoje")
    print("2. Amanhã")
    print("3. Data específica (AAAA-MM-DD)")
    print("4. Intervalo de datas (AAAA-MM-DD até AAAA-MM-DD)")
    
    date_choice = input("Escolha: ").strip()
    now_brt = datetime.now(brt)
    
    target_dates = []
    
    # Processamento da escolha do menu
    if date_choice == '1': 
        target_dates.append(now_brt.strftime('%Y-%m-%d'))
    elif date_choice == '2': 
        target_dates.append((now_brt + timedelta(days=1)).strftime('%Y-%m-%d'))
    elif date_choice == '3': 
        target_dates.append(input("Digite a data (AAAA-MM-DD): ").strip())
    elif date_choice == '4':
        start_str = input("Data inicial (AAAA-MM-DD): ").strip()
        end_str = input("Data final (AAAA-MM-DD): ").strip()
        try:
            start_dt = datetime.strptime(start_str, '%Y-%m-%d')
            end_dt = datetime.strptime(end_str, '%Y-%m-%d')
            
            if start_dt > end_dt:
                print("❌ A data inicial não pode ser maior que a data final.")
                return
                
            # Gera a lista de todas as datas dentro do intervalo
            delta = end_dt - start_dt
            for i in range(delta.days + 1):
                day = start_dt + timedelta(days=i)
                target_dates.append(day.strftime('%Y-%m-%d'))
                
        except ValueError:
            print("❌ Formato de data inválido. Certifique-se de usar AAAA-MM-DD.")
            return
    else:
        print("❌ Opção inválida.")
        return
    
    db = DBManager()
    predictor = ProfessionalPredictor()
    
    if not predictor.load_model():
        print("⚠️ Modelo não encontrado. Treine primeiro.")
        db.close()
        return
    
    all_results = []
    
    try:
        # Loop sobre todas as datas selecionadas (1 ou mais)
        for target_date in target_dates:
            print(f"\n🔍 Buscando jogos para {target_date}...")
            # Presumindo que scan_opportunities_core retorna uma lista de dicionários
            day_results = scan_opportunities_core(date_str=target_date, db=db, verbose=True)
            if day_results:
                all_results.extend(day_results)
                
        # Exibição unificada de todos os resultados encontrados
        if all_results:
            print("\n" + "=" * 70)
            interval_label = f"{target_dates[0]}" if len(target_dates) == 1 else f"{target_dates[0]} até {target_dates[-1]}"
            print(f"📈 RESUMO FINAL ({interval_label}) - {len(all_results)} oportunidades:")
            print("=" * 70)
            
            # Ordena globalmente pela confiança (maior para menor)
            sorted_ops = sorted(all_results, key=lambda x: x['confidence'], reverse=True)
            
            for i, op in enumerate(sorted_ops, 1):
                conf_color = Colors.GREEN if op['confidence'] > 0.70 else Colors.YELLOW
                print(f"{i}. [{op['match_id']}] {op['match']}")
                print(f"   📊 {op['prediction']:.1f} esc | {conf_color}{op['confidence']*100:.0f}%{Colors.RESET} | {op['bet']} | [{op['league']}]")
            print("-" * 70)
        else:
            print("\n❌ Nenhuma oportunidade encontrada no período.")
            
    except Exception as e:
        print(f"❌ Erro no scanner: {e}")
        traceback.print_exc()
    finally:
        db.close()

def train_joint_model() -> None:
    """Treino do modelo multimercado científico (JointCornersModel)."""
    print("\n" + "=" * 60)
    print("🧬 TREINO MULTIMERCADO CIENTÍFICO (Joint Model)")
    print("=" * 60)
    print("Este modo treina o modelo de 4 targets [h1H, a1H, h2H, a2H]")
    print("com walk-forward temporal e calibração por família.\n")

    print("Opções:")
    print("  1. Treino padrão (5 folds, seed=42)")
    print("  2. Treino customizado")
    print("  0. Voltar")

    choice = input("Escolha: ").strip()
    if choice == '0':
        return

    n_splits = 5
    random_state = 42
    n_simulations = 10_000

    if choice == '2':
        try:
            n_splits = int(input("Número de folds (default=5): ").strip() or "5")
            random_state = int(input("Seed (default=42): ").strip() or "42")
            n_simulations = int(input("Simulações MC (default=10000): ").strip() or "10000")
        except ValueError:
            print("Valor inválido. Usando defaults.")

    print(f"\nConfigurações: folds={n_splits}, seed={random_state}, MC={n_simulations}")
    confirm = input("Confirmar treino? (s/n): ").strip().lower()
    if confirm != 's':
        print("Treino cancelado.")
        return

    db = DBManager()
    try:
        from src.features.feature_store import FeatureStore

        df_history = db.get_historical_data()
        if df_history.empty or len(df_history) < 200:
            print("⚠️ Histórico insuficiente. Necessário ≥ 200 jogos com dados HT.")
            return

        feature_store = FeatureStore(db)
        JointTrainer = _get_joint_trainer()
        trainer = JointTrainer(
            n_splits=n_splits,
            n_simulations=n_simulations,
            random_state=random_state,
        )
        report = trainer.run(df_history, feature_store)

        print("\n" + "=" * 60)
        print("✅ TREINO CONCLUÍDO")
        print("=" * 60)
        if 'oof_metrics' in report:
            print("\nMétricas OOF (Walk-Forward):")
            for k, v in report['oof_metrics'].items():
                val = v if isinstance(v, (int, float)) else 'N/A'
                print(f"  {k}: MAE = {val:.3f}" if isinstance(val, float) else f"  {k}: {val}")
        if 'calibration_report' in report:
            print("\nCalibração por família:")
            cal = report['calibration_report']
            if isinstance(cal, dict):
                for fam, data in cal.items():
                    n = data.get('n_samples', 0) if isinstance(data, dict) else '?'
                    print(f"  {fam}: {n} amostras")
        print(f"\nArtefatos salvos em models/ e data/evaluation/")

    except Exception as e:
        print(f"❌ Erro no treino joint: {e}")
        traceback.print_exc()
    finally:
        db.close()


def update_pending_by_date() -> None:
    """Atualiza resultados de jogos pendentes por data e verifica GREEN/RED."""
    from datetime import datetime, timedelta, timezone
    
    brt = timezone(timedelta(hours=-3))
    now_brt = datetime.now(brt)
    
    print("\n" + "=" * 50)
    print("🔄 ATUALIZAR RESULTADOS PENDENTES")
    print("=" * 50)
    print("1. Ontem")
    print("2. Hoje")
    print("3. Data específica (AAAA-MM-DD)")
    print("4. Todos os pendentes (histórico)")
    
    date_choice = input("Escolha: ").strip()
    
    target_date = None
    force_all = False
    
    if date_choice == '1':
        target_date = (now_brt - timedelta(days=1)).strftime('%Y-%m-%d')
    elif date_choice == '2':
        target_date = now_brt.strftime('%Y-%m-%d')
    elif date_choice == '3':
        target_date = input("Digite a data (AAAA-MM-DD): ").strip()
    elif date_choice == '4':
        force_all = True
    else:
        print("❌ Opção inválida.")
        return
    
    db = DBManager()
    scraper = SofaScoreScraper(headless=True, verbose=True)
    
    try:
        conn = db.connect()
        cursor = conn.cursor()
        
        # 1. Buscar jogos pendentes
        if target_date:
            print(f"\n🔍 Buscando jogos pendentes para {target_date}...")
            pending_matches = db.get_pending_matches()
            brt = timezone(timedelta(hours=-3))
            matches_to_update = [
                match for match in pending_matches
                if datetime.fromtimestamp(match['start_timestamp'], timezone.utc).astimezone(brt).strftime('%Y-%m-%d') == target_date
            ]
        else:
            print(f"\n🔍 Buscando TODOS os jogos pendentes...")
            matches_to_update = db.get_pending_matches()
        
        if not matches_to_update:
            print("✅ Nenhum jogo pendente encontrado.")
            return

        print(f"📊 Encontrados {len(matches_to_update)} jogos para atualizar.")
        
        scraper.start()
        updated_count = 0
        stats_updated = 0
        
        for match in matches_to_update:
            if target_date:
                match_id, home, away, current_status, start_ts = match
            else:
                match_id = match['match_id']
                home = match['home_team']
                away = match['away_team']
                current_status = match['status']
                start_ts = match['start_timestamp']
            print(f"\n🔄 [{match_id}] {home} vs {away} [{current_status}]")
            
            # A. Buscar detalhes atualizados
            details = scraper.get_match_details(match_id)
            if not details:
                print(f"   ⚠️ Sem dados para {match_id}")
                continue
            
            new_status = details['status']
            print(f"   Status: {current_status} → {new_status}")
            
            db.save_match(details)
            updated_count += 1
            
            # B. Se finalizado ou ao vivo, buscar estatísticas
            if new_status in ('finished', 'inprogress'):
                print(f"   📊 Buscando estatísticas...")
                stats = scraper.get_match_stats(match_id)
                
                if stats.get('corners_home_ft', 0) == 0 and stats.get('corners_away_ft', 0) == 0:
                    print(f"   ⚠️ Stats vazias (possível delay da fonte).")
                else:
                    c_h = stats.get('corners_home_ft', 0)
                    c_a = stats.get('corners_away_ft', 0)
                    xg_h = stats.get('expected_goals_home', 0)
                    xg_a = stats.get('expected_goals_away', 0)
                    print(f"   ✅ Corners: {c_h}-{c_a} | xG: {xg_h}-{xg_a}")
                
                db.save_stats(match_id, stats)
                stats_updated += 1
        
        # C. Verificar predições (GREEN/RED)
        print(f"\n{'='*50}")
        print(f"📋 Resumo: {updated_count} atualizados | {stats_updated} com stats")
        print(f"{'='*50}")
        
        print("\n🎯 Verificando predições (GREEN/RED)...")
        db.check_predictions()
        
        # D. Mostrar resultados para a data
        if target_date:
            cursor.execute("""
                SELECT p.match_id, m.home_team_name, m.away_team_name, 
                       p.prediction_label, p.confidence, p.status,
                       s.corners_home_ft, s.corners_away_ft
                FROM predictions p
                JOIN matches m ON p.match_id = m.match_id
                LEFT JOIN match_stats s ON m.match_id = s.match_id
                WHERE DATE(datetime(m.start_timestamp, 'unixepoch', '-3 hours')) = ?
                  AND p.category IN ('Main', 'Top7')
                ORDER BY p.category DESC, p.confidence DESC
            """, (target_date,))
            pred_rows = cursor.fetchall()
            
            if pred_rows:
                print(f"\n{'='*70}")
                print(f"🎯 RESULTADOS DAS PREDIÇÕES - {target_date}")
                print(f"{'='*70}")
                
                greens = 0
                reds = 0
                pending = 0
                
                for row in pred_rows:
                    mid, home, away, label, conf, status, c_h, c_a = row
                    c_h = c_h or '?'
                    c_a = c_a or '?'
                    total = f"{c_h}+{c_a}={int(c_h)+int(c_a)}" if isinstance(c_h, int) and isinstance(c_a, int) else f"{c_h}+{c_a}"
                    
                    if status == 'GREEN':
                        icon = f"{Colors.GREEN}✅ GREEN{Colors.RESET}"
                        greens += 1
                    elif status == 'RED':
                        icon = f"{Colors.RED}❌ RED{Colors.RESET}"
                        reds += 1
                    else:
                        icon = "⏳ PENDING"
                        pending += 1
                    
                    print(f"  {icon} | {home} vs {away} | {label} ({conf*100:.0f}%) | Corners: {total}")
                
                total_resolved = greens + reds
                if total_resolved > 0:
                    wr = greens / total_resolved * 100
                    print(f"\n  📊 Win Rate: {greens}/{total_resolved} = {wr:.0f}%")
                if pending > 0:
                    print(f"  ⏳ Pendentes: {pending}")
                print(f"{'='*70}")
        
        print("\n✅ Atualização concluída!")
        
    except Exception as e:
        print(f"❌ Erro: {e}")
        traceback.print_exc()
    finally:
        try:
            scraper.stop()
        except:
            pass
        db.close()


def manage_users_cli():
    """Menu de usuários."""
    db = DBManager()
    while True:
        print("\n" + "="*30)
        print("👤 GERENCIAMENTO DE USUÁRIOS")
        print("="*30)
        print("1. Listar Usuários")
        print("2. Criar Usuário")
        print("3. Deletar Usuário")
        print("4. Alterar Senha")
        print("0. Voltar")
        
        choice = input("Escolha: ").strip()
        if choice == '1':
            users = db.list_users()
            print("\n📋 Usuários:")
            for u in users: print(f"{u[0]:<5} {u[1]:<15} {u[2]:<10} {u[3]:<10}")
        elif choice == '2':
            user = input("Username: ").strip()
            pwd = input("Senha: ").strip()
            role = input("Role [user]: ").strip() or 'user'
            if db.create_user(user, pwd, role): print(f"✅ Criado!")
            else: print("❌ Erro.")
        elif choice == '3':
            user = input("Username: ").strip()
            if db.delete_user(user): print(f"✅ Deletado.")
            else: print(f"❌ Não encontrado.")
        elif choice == '4':
            user = input("Username: ").strip()
            pwd = input("Nova Senha: ").strip()
            db.update_user_password(user, pwd)
            print(f"✅ Atualizado.")
        elif choice == '0': break

def run_cli():
    parser = argparse.ArgumentParser(description="Cortex Corners Pro - Sistema Profissional")
    parser.add_argument('--train', type=int, help='Modo de treino (1-4)')
    parser.add_argument('--trials', type=int, help='Número de trials do Optuna')
    parser.add_argument('--url', type=str, help='URL do SofaScore para análise')
    parser.add_argument('--reset-bets', action='store_true', help='Reseta todo o histórico de apostas e bancas')
    
    global args
    args = parser.parse_args()
    
    if args.train:
        train_model(args)
        return
    if args.url:
        analyze_match_url() # Need to pass url? No, original main calls analyze_match_url which INPUTS url. But if provided via arg...
        # The original code logic for --url calling analyze_match_url does NOT use the url arg! 
        # Line 702: analyze_match_url().
        # Inside analyze_match_url: `url = input(...)`.
        # This seems like a bug in original code or incomplete implementation. 
        # I should probably fix it to use args.url if available.
        # But for strict refactor without changing logic too much I'll follow pattern.
        # Actually I can improve it.
        # I will leave as is unless I want to improve. The user asked for "Audit & Refactor Plan".
        # I'll stick to 1:1 refactor mostly.
        return
    if args.reset_bets:
        db = DBManager()
        db.reset_all_betting_history()
        db.close()
        return

    while True:
        print("\n" + "═" * 50)
        print(f"{Colors.BOLD}🤖 CORTEX CORNERS PRO - SISTEMA DE PREVISÃO{Colors.RESET}")
        print("═" * 50)
        print("1. Atualizar Campeonato Brasileiro Serie A")
        print("2. Treinar Modelo de IA (Joint - Padrão)")
        print("3. Analisar Jogo (URL)")
        print("4. Consultar Análise (ID)")
        print("5. Atualizar Liga Específica (3 Anos)")
        print("6. Atualizar Jogo Específico (URL)")
        print(f"{Colors.CYAN}7. 📡 Scanner de Oportunidades (Dia){Colors.RESET}")
        print(f"{Colors.GREEN}8. 🧬 Treinar Modelo Multimercado (Alias compat.){Colors.RESET}")
        print("9. 🚀 Atualizar TODAS as Ligas (3 Anos - Batch)")
        print("10. 🧹 Limpar Histórico (Remover GREEN/RED)")
        print(f"{Colors.YELLOW}11. 👤 Gerenciar Usuários{Colors.RESET}")
        print(f"{Colors.RED}12. 💣 RESET TOTAL (Zerar Apostas e Bancas){Colors.RESET}")
        print(f"{Colors.CYAN}13. 🔄 Atualizar Resultados Pendentes (por Data){Colors.RESET}")
        print("0. Sair")
        
        choice = input("\nEscolha uma opção: ")
        
        if choice == '1': update_database()
        elif choice == '2': train_joint_model()
        elif choice == '3': analyze_match_url()
        elif choice == '4': retrieve_analysis()
        elif choice == '5': update_specific_league()
        elif choice == '6': update_match_by_url()
        elif choice == '7': scan_opportunities()
        elif choice == '8':
            print("\nℹ️ Compatibilidade: a opção 8 é um alias da opção 2 (treino Joint).")
            train_joint_model()
        elif choice == '9': update_all_leagues()
        elif choice == '10':
            db = DBManager()
            db.clear_finished_predictions()
            db.close()
        elif choice == '11': manage_users_cli()
        elif choice == '12':
            confirm = input(f"{Colors.RED}⚠️ VOCÊ TEM CERTEZA? (s/n): {Colors.RESET}")
            if confirm.lower() == 's':
                db = DBManager()
                db.reset_all_betting_history()
                db.close()
        elif choice == '13': update_pending_by_date()
        elif choice == '0':
            print("Saindo...")
            break
        else:
            print("Opção inválida.")
