# Cortex Bet

Cortex Bet is a soccer pre-live corner prediction system focused on a scientifically rigorous, testable and auditable architecture. The system models the joint distribution of corners across 9 derived markets:

| Market family | Markets |
|---|---|
| Full-time | FT total, FT home, FT away |
| 1st half | 1T total, 1T home, 1T away |
| 2nd half | 2T total, 2T home, 2T away |

All markets are derived from a single latent vector `Y = [home_1H, away_1H, home_2H, away_2H]`, guaranteeing mathematical coherence: FT = 1H + 2H for every team and total.

## Key Features

- **Multimercado Joint Model**: single probabilistic model produces all 9 markets from the same distribution.
- **Scientific Top 7**: match opportunities ranked by calibrated probability, uncertainty penalty and stability per market family — not heuristics.
- **Champion-only output**: one champion in production, one challenger in shadow mode. No multi-AI voting on final picks.
- **Per-family calibration**: 9 independent calibrators with hierarchical pooling for small samples.
- **Official HTTP Backend**: FastAPI server in `src/api/server.py`.
- **Official Frontend**: Next.js app in `web_app` with scientific data panels (score, E[corners], uncertainty, CI 90%, calibrated probability, stability).
- **Canonical Feature Source**: unified feature extraction through `FeatureStore`.
- **Model Governance**: JSON-backed model registry with walk-forward promotion policy.

## Architecture

- **Frontend (official)**: `web_app` (Next.js) — match cards with scientific metrics, Top 7 Scientific panel.
- **Backend (official)**: `src/api/server.py` (FastAPI).
- **Joint Model**: `src/ml/joint_model.py` — 4x LGBM Poisson regressors for the latent vector.
- **Market Translator**: `src/ml/market_translator.py` — bivariate Poisson MC simulation producing 9 market distributions.
- **Per-Market Calibrators**: `src/ml/per_market_calibrator.py` — isotonic regression per family.
- **Scientific Scorer**: `src/domain/strategies/scientific_scorer.py` — Top 7 ranking engine.
- **Neural Challenger**: `src/models/neural_multihead.py` — multi-head shadow model (never in production output).
- **Core Domain**: Python modules under `src/` with contract tests.
- **Operational Automation**: `scripts/`.
- **Research-only Experiments**: `research/`.

## Getting Started

1. **Setup environment**: use the workspace virtual environment `.venv`.
2. **Train the multimercado joint model** (required for scientific Top 7):
   ```bash
   # Via CLI interactive menu (option 2 - Joint default; option 8 is compatibility alias):
   python src/main.py
   # Or via script:
   python scripts/train_model.py
   ```
3. **Run scanner**: `python scripts/run_scanner.py --date tomorrow`.
4. **Start the dashboard**: `python start_system.py` or `START_DASHBOARD.bat`.
5. **Check model registry**: `python scripts/model_registry_cli.py status`.
6. **Check model health**: `python scripts/check_model_health.py`.

See `README_ML.md` for the complete training guide and methodology.

## Operational Notes (CLI + Web)

- **Single training truth**: Joint training is the default in CLI option `2` and in web endpoint `POST /api/model/train`.
- **Compatibility alias**: CLI option `8` calls the same Joint flow and is kept only to avoid operator breakage.
- **Legacy namespace**: old AutoML optimization is now explicitly legacy at `POST /api/legacy/model/optimize`.
- **Deprecated alias**: `POST /api/model/optimize` remains as compatibility alias and returns deprecation metadata.
- **Top 7 method**: scanner ranking uses `ScientificSelectionStrategy` in `src/analysis/unified_scanner.py`.

## Documentation

- `README_ML.md`: machine learning methodology and training guide.
- `docs/governance.md`: promotion criteria, odds policy, ablation schedule.
- `docs/architecture.md`: current architecture and target principles.
- `docs/decision_log.md`: refactor decisions by phase.
- `docs/refactor_audit.md`: incremental audit trail and residual risks.
- `docs/audit/`: inventory, diagnostics, migration plan.
