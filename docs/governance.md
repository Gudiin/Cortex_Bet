# Governança Científica — Cortex Bet
> Branch: `refactor/multimercado-cientifico` | Versão: 1.0 | Data: 2026-03-31

---

## 1. Princípio Central

O sistema produz **probabilidades calibradas** para 9 mercados de escanteios.  
Não realiza afirmações de "value bet" sem odd real informada pelo usuário.

---

## 2. Papéis de Modelos

### 2.1 Champion (produção)
- **Atual**: `ProfessionalPredictor` (Ensemble LGBM + RF, objetivo Tweedie) — `ensemble_v1`
- **Próximo**: `JointCornersModel` (Bivariate NB por período, 4 saídas) após Walk-Forward aprovado
- **Regra**: Um único champion no output. Nunca blend com challenger.
- **Feature flag**: `CHAMPION_ONLY_MODE=1` (env var) — ativado por default

### 2.2 Challenger (shadow mode)
- **Atual (legado)**: `NeuralChallenger` (MLPRegressor — FT total apenas)
- **Próximo**: `NeuralMultiHead` (4 cabeças, lê vetor latente)
- **Regra**: Logs em `data/shadow_logs/challenger_shadow.jsonl`. Nunca visível no output.

---

## 3. Critério de Promoção do Champion

O challenger **só pode ser promovido** se atender TODOS os critérios abaixo:

| Critério | Requisito |
|---|---|
| Walk-Forward | Brier Score < champion em ≥ 4 dos 5 folds |
| Calibração (ECE) | ECE < champion em FT total, HT total e HT2 total |
| Estabilidade | Desvio padrão do Brier por liga ≤ champion |
| Família Critical | Sem degradação material (Δ Brier > 0.01) em nenhuma família |
| Reprodutibilidade | Resultado reproduzível com seed fixo (42) |

**"Ganho global com degradação em 1T ou 2T" NÃO é promoção automática.**

### 3.1 Processo de Promoção
1. Executar `JointTrainer.run()` com `n_splits=5`
2. Executar `WalkForwardValidator.run()` para ambos (champion e challenger)
3. Comparar `ModelEvaluationReport.summary_dataframe()` por família
4. Se todos os critérios acima Ok: atualizar `ModelRegistry` via `model_registry_cli.py`
5. Documentar decisão em `docs/decision_log.md`

---

## 4. Política de Odds

### 4.1 Sem Odd Informada pelo Usuário
- O sistema exibe **probabilidade calibrada** e **odd justa do modelo**
- **Proibido**: afirmar "mercado X é o melhor economicamente"
- **Permitido**: ranquear por `P_calibrada`, estabilidade por liga, ECE local

### 4.2 Com Odd Informada pelo Usuário
- Chamada a `MarketTranslator.compare_with_user_odd(prob_calibrated, user_odd)`
- Exibe: `fair_odd`, `implied_prob_market`, `edge_vs_market`
- **Não é output automático** — requer ação explícita do usuário

---

## 5. Métricas de Avaliação

### 5.1 Primárias (critério de promoção)
| Métrica | Interpretação |
|---|---|
| Brier Score | Erro quadrático em probabilidade. **Menor = melhor** |
| Log Loss | Entropia cruzada. Pela formulação binária. **Menor = melhor** |
| RPS | Ranked Probability Score. Para distribuição discreta. **Menor = melhor** |
| ECE | Expected Calibration Error. Alinhamento prob vs frequência. **< 0.05 = calibrado** |

### 5.2 Secundárias (contexto)
| Métrica | Papel |
|---|---|
| MAE do E[X] | Qualidade da estimativa de contagem (não de probabilidade) |
| Sharpness | Quão "confiante" é o modelo (não = correto) |
| Estabilidade (liga/temporada) | std(Brier) — invariância por contexto |
| Cobertura CI 90% | Frequência de X real dentro do intervalo de credibilidade |
| Hit Rate | **APENAS auxiliar** — nunca como critério principal |

### 5.3 Proibições de Métricas
- **Proibido**: usar acurácia simples como métrica principal
- **Proibido**: comparar modelos apenas por hit rate
- **Proibido**: reportar apenas FT total — sempre por família
- **Proibido**: considerar promoção com degradação em qualquer família

---

## 6. Calibração

### 6.1 Calibradores por Família
9 calibradores independentes gerenciados pelo `PerMarketCalibrator`:

```
ft_total | ht_total | ht2_total | ft_home | ft_away | ht_home | ht_away | ht2_home | ht2_away
```

### 6.2 Pooling Hierárquico
Quando `n_local < 100` amostras:
```
P_cal = α * P_local + (1 - α) * P_global
α = n_local / (n_local + 100)
P_global = calibrador ft_total (família com mais amostras)
```

### 6.3 Threshold de Produção
- `n < 10`: calibrador não ajustado → pass-through (aviso emitido)
- `10 ≤ n < 30`: temperature scaling
- `n ≥ 30`: isotonic regression

---

## 7. Coerência do Vetor Latente

### 7.1 Restrições Obrigatórias
$$\text{home\_ft} = \text{home\_1H} + \text{home\_2H}$$
$$\text{away\_ft} = \text{away\_1H} + \text{away\_2H}$$
$$\text{total\_ft} = \text{home\_ft} + \text{away\_ft}$$

### 7.2 Verificação Automática
`JointCornersModel.predict_lambda()` garante coerência por construção.  
`MarketTranslator` deriva todos os 9 mercados da **mesma simulação Monte Carlo**.

### 7.3 Proibições de Derivação
- **Proibido**: `2T = FT - 1T` como derivação de previsão de modelo
- **Proibido**: usar médias históricas separadas para HT e FT sem vínculo

---

## 8. Ablação Obrigatória

Qualquer heurística deve passar por ablação temporal antes de ser mantida.  
Candidatas identificadas no audit (`docs/audit/02_diagnostico.md`):

| Heurística | Arquivo | Status |
|---|---|---|
| `clip(lower=3.0)` em corners | `feature_store.py` | Pendente ablação |
| `agreement = 1.1` (challenger boost) | `manager_ai.py` | Removido em CHAMPION_ONLY_MODE |
| `Top 7` como critério de seleção | `unified_scanner.py` | Pendente ablação |
| `Safety_1 / Safety_2` lines | `manager_ai._find_best_line` | Pendente ablação |
| Blend `0.5 / 0.5` champion+challenger | `manager_ai.py` | **Eliminado** (Fase 1) |

Para cada heurística: comparar Brier Score com vs. sem em walk-forward temporal.

---

## 9. Rastreabilidade

### 9.1 Shadow Log
- Localização: `data/shadow_logs/challenger_shadow.jsonl`
- Formato: JSON Lines (uma linha por predição)
- Campos: `ts, match_id, champion_raw, challenger_raw, champion_conf, challenger_conf, line_val, is_over, champion_only_mode`

### 9.2 Walk-Forward Reports
- Localização: `data/evaluation/walkforward_<model_id>_<timestamp>.json`
- Gerado por: `WalkForwardValidator.run()`

### 9.3 Model Registry
- Localização: `data/model_registry.json`
- Gerenciado por: `ModelRegistry` (src/models/model_registry.py)
- CLI: `scripts/model_registry_cli.py`

---

## 10. Histórico de Decisões

| Data | Decisão | Arquivo |
|---|---|---|
| 2026-03-31 | Ativar CHAMPION_ONLY_MODE=True | `manager_ai.py` |
| 2026-03-31 | Criar scaffold JointCornersModel | `src/ml/joint_model.py` |
| 2026-03-31 | Criar scaffold NeuralMultiHead | `src/models/neural_multihead.py` |
| 2026-03-31 | Criar PerMarketCalibrator (9 famílias) | `src/ml/per_market_calibrator.py` |
| 2026-03-31 | Criar MarketTranslator (distribuição conjunta) | `src/ml/market_translator.py` |
| 2026-03-31 | Criar WalkForwardValidator | `src/training/walk_forward_validator.py` |
| 2026-03-31 | Criar SciEvaluator (Brier, ECE, RPS) | `src/evaluation/sci_evaluator.py` |
| 2026-03-31 | Criar JointTrainer (pipeline completo) | `src/training/joint_trainer.py` |
| 2026-03-31 | Adicionar create_joint_targets() | `src/ml/features_v2.py` |
| 2026-03-31 | CLI opção 2 → treino Joint padrão | `src/interface/cli.py` |
| 2026-03-31 | CLI opção 8 → alias de compatibilidade | `src/interface/cli.py` |
| 2026-03-31 | Web /api/model/train → JointTrainer | `src/web/server.py` |
| 2026-03-31 | Mover AutoML para /api/legacy/model/optimize | `src/web/server.py` |
| 2026-03-31 | Alias /api/model/optimize → deprecated + legacy | `src/web/server.py` |

Ver detalhes: `docs/decision_log.md`

---

## 11. Contrato Operacional (CLI + Web)

### 11.1 Treino Canônico
- CLI: opção `2` = treino Joint padrão
- CLI: opção `8` = **alias de compatibilidade** da opção 2
- Web: `POST /api/model/train` = treino Joint padrão

Parâmetros canônicos:
- `n_splits` (default `5`)
- `random_state` (default `42`)
- `n_simulations` (default `10000`)

### 11.2 Namespace Legacy
- Fluxos legados devem ficar em `/api/legacy/*`
- Endpoint legado atual: `POST /api/legacy/model/optimize`
- Alias antigo mantido por compatibilidade: `POST /api/model/optimize` com aviso de depreciação

### 11.3 Top 7 Científico
- Seleção Top 7 oficial: `ScientificSelectionStrategy` em `src/analysis/unified_scanner.py`
- Proibido voltar para ranking heurístico baseado apenas em confiança
